"""One environment class for the whole crutch curriculum.

Replaces the four sibling classes in sconegym_crutch_v3, which were file copies
of each other. Stage differences live in stages.STAGES.

Three corrections to v3 behaviour are baked in here and are not optional:

* prev_action is part of the observation. The action rate limiter makes the
  executed torque a function of the previous action; in v3 that buffer was
  hidden from the policy, so returns depended on state the critic could not
  see. See include_prev_action to A/B it.
* The crutch contact force API is probed at construction and raises if a stage
  weights a crutch term but the API does not work. v3 caught the failure,
  printed once, and returned 0.0 forever.
* Reward terms are all in [0, 1] and composed as a weighted geometric mean, so
  the per-step reward is non-negative and no term can be farmed in isolation.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import gym
import numpy as np
from sconegym.gaitgym import GaitGym

from .rewards import gaussian, smoothstep
from .stages import StageSpec, get_stage

CANE_BODY_NAMES = ("Crutch_r", "Crutch_l")
FOOT_BODY_NAMES = ("calcn_r", "calcn_l")
PELVIS_BODY_NAME = "pelvis"

DEFAULT_MODEL = "Rajagopal_crutch_v3_A0_walk_003.scone"


class CrutchCurriculumGym(GaitGym):
    """Torque-actuated 2D Rajagopal model with welded crutches."""

    ACTUATOR_NAMES = (
        "lumbar_extension",
        "hip_flexion_r",
        "knee_angle_r",
        "hip_flexion_l",
        "knee_angle_l",
        "arm_flex_r",
        "elbow_flex_r",
        "arm_flex_l",
        "elbow_flex_l",
    )

    TORQUE_SCALES_DEFAULT = np.asarray(
        [80.0, 120.0, 120.0, 120.0, 120.0, 80.0, 60.0, 80.0, 60.0],
        dtype=np.float32,
    )

    LOCKED_DOFS = ("ankle_angle_r", "mtp_angle_r", "ankle_angle_l", "mtp_angle_l")

    RANDOMIZED_Q = (
        "pelvis_tilt",
        "pelvis_ty",
        "lumbar_extension",
        "hip_flexion_r",
        "knee_angle_r",
        "hip_flexion_l",
        "knee_angle_l",
    )

    N_ACT = 9
    N_DOF = 16

    def __init__(
        self,
        stage: str = "A",
        model_file: Optional[str] = None,
        include_prev_action: bool = True,
        strict_crutch: bool = True,
        torque_scales: Optional[Sequence[float]] = None,
        gamma_for_safety_check: float = 0.99,
        gaitgym_kwargs: Optional[dict] = None,
        **overrides,
    ):
        self.spec_key = str(stage).upper()
        self.stage_spec: StageSpec = get_stage(self.spec_key).with_overrides(**overrides)
        self.curriculum_stage = self.stage_spec.name

        # Must exist before super().__init__, which calls
        # _setup_action_observation_spaces -> _get_obs.
        self._include_prev_action = bool(include_prev_action)
        self.prev_action = np.zeros(self.N_ACT, dtype=np.float32)
        self.current_action = np.zeros(self.N_ACT, dtype=np.float32)

        self.torque_scales = (
            self.TORQUE_SCALES_DEFAULT.copy()
            if torque_scales is None
            else np.asarray(torque_scales, dtype=np.float32)
        )
        if self.torque_scales.shape != (self.N_ACT,):
            raise ValueError("torque_scales must have shape (%d,)" % self.N_ACT)

        if model_file is None:
            model_file = str(
                Path(__file__).resolve().parents[2] / "models" / "scone" / DEFAULT_MODEL
            )
        if not Path(model_file).is_file():
            raise FileNotFoundError("model file not found: %s" % model_file)

        spec = self.stage_spec
        super().__init__(
            model_file=model_file,
            left_leg_idxs=[],
            right_leg_idxs=[],
            root_body_name=PELVIS_BODY_NAME,
            foot_body_name="calcn",
            target_vel=spec.target_vel,
            leg_switch=False,
            clip_actions=False,
            obs_type="2D",
            min_com_height=float(spec.min_com_height),
            min_head_height=float(spec.min_head_height),
            fall_recovery_time=0.0,
            **(gaitgym_kwargs or {}),
        )

        self.init_load = float(spec.init_load)
        self.step_size = 0.01
        self._max_episode_steps = int(spec.episode_steps)

        self._dof_names = [self._name_of(d) for d in self.model.dofs()]
        self._dof_index = {n: i for i, n in enumerate(self._dof_names)}
        self._actuator_names = [self._name_of(a) for a in self.model.actuators()]
        self._validate_model()

        self._base_q = np.asarray(self.init_dof_pos, dtype=np.float64).copy()
        self._base_dq = np.asarray(self.init_dof_vel, dtype=np.float64).copy()
        self._base_pelvis_y = float(self._base_q[self._dof_index["pelvis_ty"]])

        self._pelvis_body = self._find_body(PELVIS_BODY_NAME)
        self._foot_bodies = [self._find_body(n) for n in FOOT_BODY_NAMES]
        self._cane_bodies = [self._find_body(n) for n in CANE_BODY_NAMES]

        self._body_weight_n = self._resolve_body_weight()
        self._crutch_force_ok = False
        self._crutch_probe_error: Optional[str] = None
        self._probe_crutch_api(strict=bool(strict_crutch))

        self.rwd_dict: Optional[Dict[str, float]] = None
        self.term_values: Dict[str, float] = {}
        self._rng = np.random.RandomState(0)

        self.stage_spec.reward.warn_if_unsafe(
            gamma=float(gamma_for_safety_check), label=self.curriculum_stage
        )

    # -- introspection ----------------------------------------------------

    @staticmethod
    def _name_of(obj) -> str:
        try:
            return str(obj.name())
        except TypeError:
            return str(obj.name)

    def _find_body(self, name: str):
        try:
            return self.model.find_body(name)
        except Exception:
            for b in self.model.bodies():
                if self._name_of(b) == name:
                    return b
        return None

    @staticmethod
    def _vec_x(vec) -> float:
        try:
            return float(vec.x)
        except AttributeError:
            return float(np.asarray(vec, dtype=np.float64)[0])

    @staticmethod
    def _vec_y(vec) -> float:
        try:
            return float(vec.y)
        except AttributeError:
            return float(np.asarray(vec, dtype=np.float64)[1])

    def _validate_model(self) -> None:
        if len(self.model.dofs()) != self.N_DOF:
            raise RuntimeError(
                "expected %d dofs, found %d" % (self.N_DOF, len(self.model.dofs()))
            )
        if len(self.model.actuators()) != self.N_ACT:
            raise RuntimeError(
                "expected %d actuators, found %d"
                % (self.N_ACT, len(self.model.actuators()))
            )
        if len(self.model.muscles()) != 0:
            raise RuntimeError(
                "expected a torque-only model, found %d muscles"
                % len(self.model.muscles())
            )
        if tuple(self._actuator_names) != self.ACTUATOR_NAMES:
            raise RuntimeError(
                "unexpected actuator order\nexpected: %r\nfound:    %r"
                % (self.ACTUATOR_NAMES, tuple(self._actuator_names))
            )
        missing = [
            n
            for n in tuple(self.ACTUATOR_NAMES) + self.LOCKED_DOFS + ("pelvis_tx",)
            if n not in self._dof_index
        ]
        if missing:
            raise RuntimeError("missing dofs: %s" % ", ".join(missing))

    def _resolve_body_weight(self) -> Optional[float]:
        try:
            mass = float(self.model.mass())
        except Exception:
            return None
        return mass * 9.81 if mass > 0 else None

    def _probe_crutch_api(self, strict: bool) -> None:
        """Establish up front whether crutch sensing actually works.

        v3 discovered this lazily inside a try/except during stepping, printed
        one message, and then silently contributed 0.0 to the reward for the
        rest of training. Any stage that weights a crutch term now refuses to
        construct unless the API is confirmed.
        """
        spec = self.stage_spec
        problems: List[str] = []

        missing_bodies = [
            n for n, b in zip(CANE_BODY_NAMES, self._cane_bodies) if b is None
        ]
        if missing_bodies:
            problems.append("crutch bodies not found: %s" % ", ".join(missing_bodies))

        if not problems:
            try:
                total = 0.0
                for body in self._cane_bodies:
                    total += self._vec_y(body.contact_force())
                self._crutch_force_ok = np.isfinite(total)
                if not self._crutch_force_ok:
                    problems.append("contact_force() returned a non-finite value")
            except Exception as exc:
                problems.append("contact_force() failed: %r" % (exc,))

        if spec.needs_crutch_force and self._body_weight_n is None:
            problems.append("model.mass() unavailable, cannot normalise crutch load")

        needed = spec.needs_crutch_force or spec.needs_crutch_pose
        if problems:
            self._crutch_probe_error = "; ".join(problems)
            message = (
                "stage %s needs crutch sensing but the API is not working: %s"
                % (self.curriculum_stage, self._crutch_probe_error)
            )
            if needed and strict:
                raise RuntimeError(
                    message
                    + "\nPass strict_crutch=False to train anyway (the crutch terms"
                    " will read 0.0 and the reward will be meaningless)."
                )
            if needed:
                warnings.warn(message, RuntimeWarning, stacklevel=2)

    # -- spaces and observation ------------------------------------------

    def _setup_action_observation_spaces(self) -> None:
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.N_ACT, dtype=np.float32),
            high=np.ones(self.N_ACT, dtype=np.float32),
            dtype=np.float32,
        )
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-10000.0, high=10000.0, shape=obs.shape, dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """Base observation, with the rate limiter's state appended.

        Without prev_action the MDP is not Markov in the observation: the same
        observed state maps to different executed torques depending on a buffer
        the policy cannot see.
        """
        obs = np.asarray(super()._get_obs(), dtype=np.float32).ravel()
        if not self._include_prev_action:
            return obs
        return np.concatenate([obs, self.prev_action.astype(np.float32)])

    # -- episode lifecycle -----------------------------------------------

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        self._rng = np.random.RandomState(int(seed))
        return [int(seed)]

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options=None):
        if seed is not None:
            self.seed(seed)
        spec = self.stage_spec

        self.episode_number = self._rng.randint(0, 1_000_000)
        self.model.reset()
        self.has_reset = True
        self.time = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.fall_time = -1.0
        self.prev_action[:] = 0.0
        self.current_action[:] = 0.0
        self.rwd_dict = None
        self.term_values = {}

        self.model.set_store_data(self.store_next)

        q = self._base_q.copy()
        dq = np.zeros_like(self._base_dq)

        for name in self.RANDOMIZED_Q:
            i = self._dof_index[name]
            std = (
                spec.pelvis_height_position_std
                if name == "pelvis_ty"
                else spec.reset_position_std
            )
            q[i] += self._rng.normal(0.0, std)
            dq[i] = self._rng.normal(0.0, spec.reset_velocity_std)

        q[self._dof_index["pelvis_tx"]] = 0.0
        for name in self.LOCKED_DOFS:
            q[self._dof_index[name]] = 0.0
            dq[self._dof_index[name]] = 0.0

        vx0 = self._rng.normal(
            spec.initial_forward_velocity, spec.initial_forward_velocity_std
        )
        dq[self._dof_index["pelvis_tx"]] = max(0.0, float(vx0))

        self.model.set_dof_positions(q)
        self.model.set_dof_velocities(dq)
        self.model.init_state_from_dofs()
        if self.init_load > 0:
            self.model.adjust_state_for_load(self.init_load)

        obs = self._get_obs()
        return (obs, {}) if return_info else obs

    def _rate_limit(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.N_ACT,):
            raise ValueError(
                "expected action shape (%d,), got %r" % (self.N_ACT, action.shape)
            )
        action = np.clip(action, -1.0, 1.0)
        limit = self.stage_spec.action_rate_limit
        delta = np.clip(action - self.prev_action, -limit, limit)
        limited = np.clip(self.prev_action + delta, -1.0, 1.0)
        self.prev_action = limited.astype(np.float32)
        return self.prev_action

    def step(self, action):
        if not self.has_reset:
            raise RuntimeError("call reset() before step()")

        normalized = self._rate_limit(action)
        self.current_action = normalized.copy()
        self.model.set_actuator_inputs(normalized * self.torque_scales)
        self.model.advance_simulation_to(self.time + self.step_size)

        reward = self._get_rew()
        obs = self._get_obs()
        done = self._get_done()
        reward = self._apply_termination_cost(reward, done)

        self.time += self.step_size
        self.total_reward += reward

        info = {"curriculum_stage": self.curriculum_stage}
        info.update(self.rwd_dict or {})

        if done:
            if self.store_next:
                self.model.write_results(
                    self.output_dir, "%05d_%.3f" % (self.episode, self.total_reward)
                )
                self.store_next = False
            self.episode += 1

        return obs, float(reward), bool(done), info

    # -- reward terms, all in [0, 1] -------------------------------------

    def _dof(self) -> np.ndarray:
        return np.asarray(self.model.dof_position_array(), dtype=np.float64)

    def _pelvis_body_x(self) -> Optional[float]:
        """World x of the pelvis body COM.

        Used by the relative-geometry terms (crutch_forward, pelvis_lag) so
        they compare like with like. v3 mixed model.com_pos().x with the
        pelvis_tx dof between these two terms.
        """
        if self._pelvis_body is None:
            return None
        try:
            return self._vec_x(self._pelvis_body.com_pos())
        except Exception:
            return None

    def _travel_x(self) -> float:
        """Forward travel since reset, from the pelvis_tx dof (zeroed at reset)."""
        return float(self._dof()[self._dof_index["pelvis_tx"]])

    def _term_height(self) -> float:
        y = self._dof()[self._dof_index["pelvis_ty"]]
        drop = max(0.0, self._base_pelvis_y - float(y))
        return gaussian(drop, self.stage_spec.terms.height_drop_sigma)

    def _term_posture(self) -> float:
        t = self.stage_spec.terms
        q = self._dof()
        tilt = q[self._dof_index["pelvis_tilt"]]
        lumbar = q[self._dof_index["lumbar_extension"]]
        trunk = float(
            np.exp(-((tilt / max(t.pelvis_tilt_sigma, 1e-9)) ** 2)
                   - ((lumbar / max(t.lumbar_sigma, 1e-9)) ** 2))
        )
        dev_sq = 0.0
        for name in ("hip_flexion_r", "knee_angle_r", "hip_flexion_l", "knee_angle_l"):
            i = self._dof_index[name]
            dev_sq += float(q[i] - self._base_q[i]) ** 2
        stance = float(np.exp(-dev_sq / max(t.hip_knee_sigma**2, 1e-12)))
        return float(np.clip(trunk * stance, 0.0, 1.0))

    def crutch_contact_force(self) -> float:
        """Total upward contact force on both crutch tips, in newtons."""
        if not self._crutch_force_ok:
            return 0.0
        total = 0.0
        for body in self._cane_bodies:
            try:
                total += max(0.0, self._vec_y(body.contact_force()))
            except Exception:
                return 0.0
        return float(total)

    def _term_crutch(self) -> float:
        t = self.stage_spec.terms
        if self._body_weight_n is None or self._body_weight_n <= 0.0:
            return 0.0
        force = self.crutch_contact_force()
        # Smooth contact gate instead of v3's hard 20 N cliff. A crutch that is
        # not loaded scores 0 rather than collecting the Gaussian's tail.
        gate = smoothstep(force, t.cane_gate_lo_n, t.cane_gate_hi_n)
        frac = force / self._body_weight_n
        match = gaussian(frac - t.cane_target_load_fraction, t.cane_load_sigma_fraction)
        return float(np.clip(gate * match, 0.0, 1.0))

    def _term_crutch_forward(self) -> float:
        t = self.stage_spec.terms
        pelvis_x = self._pelvis_body_x()
        if pelvis_x is None:
            return 0.0
        scores = []
        for body in self._cane_bodies:
            if body is None:
                continue
            try:
                offset = self._vec_x(body.com_pos()) - pelvis_x
            except Exception:
                continue
            if not np.isfinite(offset):
                continue
            # Deviation from where the crutch sits in the neutral pose, not
            # from the pelvis itself. The crutches are welded to the forearms
            # and rest ~5 cm ahead of the pelvis, so measuring against zero
            # made this term a constant 1.0 that discriminated nothing.
            shortfall = (t.crutch_offset_ref - offset) - t.crutch_forward_margin
            scores.append(gaussian(max(0.0, shortfall), t.crutch_forward_sigma))
        return float(np.mean(scores)) if scores else 0.0

    def _term_velocity(self) -> float:
        v = self._vec_x(self.model.com_vel())
        return gaussian(v - self.stage_spec.target_vel, self.stage_spec.velocity_sigma)

    def _term_backward(self) -> float:
        """1.0 when not moving backward, decaying as backward speed grows.

        v3 made this an unbounded subtractive penalty (up to -2.25/step against
        a +1.55 maximum), which made falling immediately the optimal policy.
        """
        v = self._vec_x(self.model.com_vel())
        if v >= 0.0:
            return 1.0
        return gaussian(v, self.stage_spec.terms.backward_sigma)

    def _term_displacement(self) -> float:
        t = self.stage_spec.terms
        excess = max(0.0, abs(self._travel_x()) - t.displacement_cap)
        return gaussian(excess, t.displacement_sigma)

    def _term_pelvis_forward(self) -> float:
        cap = self.stage_spec.terms.displacement_cap
        return float(np.clip(self._travel_x() / max(cap, 1e-9), 0.0, 1.0))

    def _term_pelvis_lag(self) -> float:
        """1.0 when the pelvis trails the feet no more than it does at rest.

        The calcn bodies sit ~0.10 m ahead of the pelvis body COM in the
        neutral pose, so measuring raw lag against zero scored 0.017 at rest and
        left the term pinned regardless of what the policy did.
        """
        t = self.stage_spec.terms
        pelvis_x = self._pelvis_body_x()
        if pelvis_x is None:
            return 1.0
        xs = []
        for body in self._foot_bodies:
            if body is None:
                continue
            try:
                xs.append(self._vec_x(body.com_pos()))
            except Exception:
                continue
        if not xs:
            return 1.0
        excess_lag = (min(xs) - pelvis_x) - t.pelvis_foot_offset_ref
        return gaussian(max(0.0, excess_lag), t.pelvis_lag_sigma)

    _TERM_FNS = {
        "height": _term_height,
        "posture": _term_posture,
        "crutch": _term_crutch,
        "crutch_forward": _term_crutch_forward,
        "velocity": _term_velocity,
        "backward": _term_backward,
        "displacement": _term_displacement,
        "pelvis_forward": _term_pelvis_forward,
        "pelvis_lag": _term_pelvis_lag,
    }

    def compute_terms(self) -> Dict[str, float]:
        """Evaluate only the terms this stage actually weights."""
        out: Dict[str, float] = {}
        for name in self.stage_spec.reward.required_terms:
            try:
                fn = self._TERM_FNS[name]
            except KeyError:
                raise KeyError(
                    "stage %s requests unknown reward term %r; known terms: %s"
                    % (self.curriculum_stage, name, ", ".join(sorted(self._TERM_FNS)))
                ) from None
            out[name] = float(fn(self))
        return out

    def _get_rew(self) -> float:
        self.steps += 1
        self.term_values = self.compute_terms()
        total, breakdown = self.stage_spec.reward.compose(self.term_values)
        self.rwd_dict = breakdown
        return float(total)

    def get_rwd_dict(self) -> Dict[str, float]:
        if self.rwd_dict is None:
            self._get_rew()
        return dict(self.rwd_dict or {})

    # -- termination ------------------------------------------------------

    def _is_fall(self) -> bool:
        return bool(
            self._vec_y(self.model.com_pos()) < self.min_com_height
            or self._vec_y(self.head_body.com_pos()) < self.min_head_height
        )

    def _get_done(self) -> bool:
        return bool(self._is_fall() or self.steps >= self.stage_spec.episode_steps)

    def _apply_termination_cost(self, reward: float, done: bool) -> float:
        if done and self._is_fall():
            reward -= self.stage_spec.reward.fall_penalty
        return reward

    @property
    def horizon(self) -> int:
        return int(self.stage_spec.episode_steps)
