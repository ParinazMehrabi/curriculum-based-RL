from __future__ import annotations

"""
 Stage 2c (crutch-forward correction).
"""

from pathlib import Path
from typing import Callable, Optional

import gym
import numpy as np
from sconegym.gaitgym import GaitGym


class RajagopalCrutchStage2DPostureFixGym(GaitGym):
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

    RANDOMIZED_Q = (
        "pelvis_tilt",
        "pelvis_ty",
        "lumbar_extension",
        "hip_flexion_r",
        "knee_angle_r",
        "hip_flexion_l",
        "knee_angle_l",
    )

    RANDOMIZED_DQ = RANDOMIZED_Q

    REWARD_KEYS = (
        "alive",
        "height",
        "posture",
        "crutch",
        "crutch_forward",
        "velocity",
        "backward",
        "displacement",
        "total",
        "pelvis_forward",
        "pelvis_lag",
    )

    CANE_BODY_NAMES = ("Crutch_r", "Crutch_l")

    def __init__(
        self,
        model_file: Optional[str] = None,
        alive_reward_coeff: float = 0.15,
        height_reward_coeff: float = 0.20,
        posture_reward_coeff: float = 0.45,
        crutch_reward_coeff: float = 0.20,
        velocity_reward_coeff: float = 0.25,
        target_vel: float = 0.03,
        velocity_sigma_fraction: float = 0.7,
        displacement_reward_coeff: float = 0.05,
        target_displacement_cap: float = 0.5,
        pelvis_forward_reward_coeff: float = 0.15,
        pelvis_lag_penalty_coeff: float = 0.20,
        fall_penalty: float = 5.0,
        pelvis_tilt_sigma: float = 0.12,
        lumbar_sigma: float = 0.12,
        hip_knee_posture_sigma: float = 0.35,
        height_drop_sigma: float = 0.18,
        cane_target_load_fraction: float = 0.15,
        cane_load_sigma_fraction: float = 0.15,
        backward_penalty_coeff: float = 0.45,
        crutch_forward_reward_coeff: float = 0.20,
        crutch_forward_margin: float = 0.05,
        crutch_forward_sigma: float = 0.10,
        initial_forward_velocity: float = 0.0,
        initial_forward_velocity_std: float = 0.0,
        reset_position_std: float = 0.005,
        reset_velocity_std: float = 0.005,
        pelvis_height_position_std: float = 0.002,
        init_load: float = 0.5,
        min_com_height: float = 0.55,
        min_head_height: float = 0.75,
        episode_steps: int = 1000,
        action_rate_limit: float = 0.05,
        torque_scales=None,
        **kwargs,
    ):
        if model_file is None:
            project_root = Path(__file__).resolve().parents[1]
            model_file = str(
                project_root
                / "models"
                / "scone"
                / "Rajagopal_crutch_v3_A0_walk_003.scone"
            )

        self.curriculum_stage = "2c-CRUTCH-FORWARD-FIX"

        self.alive_reward_coeff = float(alive_reward_coeff)
        self.height_reward_coeff = float(height_reward_coeff)
        self.posture_reward_coeff = float(posture_reward_coeff)
        self.crutch_reward_coeff = float(crutch_reward_coeff)
        self.velocity_reward_coeff = float(velocity_reward_coeff)
        self.backward_penalty_coeff = float(backward_penalty_coeff)
        self.displacement_reward_coeff = float(displacement_reward_coeff)
        self.crutch_forward_reward_coeff = float(crutch_forward_reward_coeff)
        self.crutch_forward_margin = float(crutch_forward_margin)
        self.crutch_forward_sigma = float(crutch_forward_sigma)
        self.fall_penalty = float(fall_penalty)

        self.pelvis_forward_reward_coeff = float(
            pelvis_forward_reward_coeff
        )

        self.pelvis_lag_penalty_coeff = float(
            pelvis_lag_penalty_coeff
        )
        self.pelvis_tilt_sigma = float(pelvis_tilt_sigma)
        self.lumbar_sigma = float(lumbar_sigma)
        self.hip_knee_posture_sigma = float(hip_knee_posture_sigma)
        self.height_drop_sigma = float(height_drop_sigma)
        self.cane_target_load_fraction = float(cane_target_load_fraction)
        self.cane_load_sigma_fraction = float(cane_load_sigma_fraction)

        self.target_vel_v3 = float(target_vel)
        self.velocity_sigma_fraction = float(velocity_sigma_fraction)
        self.velocity_sigma = max(
            self.target_vel_v3 * self.velocity_sigma_fraction, 1e-6
        )

        self.target_displacement_cap = float(target_displacement_cap)

        self.initial_forward_velocity = float(initial_forward_velocity)
        self.initial_forward_velocity_std = float(initial_forward_velocity_std)
        self.reset_position_std = float(reset_position_std)
        self.reset_velocity_std = float(reset_velocity_std)
        self.pelvis_height_position_std = float(pelvis_height_position_std)
        self.init_load_v3 = float(init_load)

        self.episode_steps_v3 = int(episode_steps)
        self.action_rate_limit = float(action_rate_limit)

        if self.target_vel_v3 <= 0.0:
            raise ValueError("target_vel must be positive for this stage")
        if self.velocity_sigma_fraction <= 0.0:
            raise ValueError("velocity_sigma_fraction must be positive")
        if self.target_displacement_cap <= 0.0:
            raise ValueError("target_displacement_cap must be positive")
        if self.crutch_forward_sigma <= 0.0:
            raise ValueError("crutch_forward_sigma must be positive")

        self.torque_scales = (
            self.TORQUE_SCALES_DEFAULT.copy()
            if torque_scales is None
            else np.asarray(torque_scales, dtype=np.float32)
        )
        if self.torque_scales.shape != (9,):
            raise ValueError("torque_scales must have shape (9,)")

        self._crutch_force_fn: Optional[Callable[[], float]] = (
            self._get_crutch_contact_force
        )
        self._crutch_warned = False

        super().__init__(
            model_file=model_file,
            left_leg_idxs=[],
            right_leg_idxs=[],
            root_body_name="pelvis",
            foot_body_name="calcn",
            target_vel=self.target_vel_v3,
            leg_switch=False,
            clip_actions=False,
            obs_type="2D",
            min_com_height=float(min_com_height),
            min_head_height=float(min_head_height),
            fall_recovery_time=0.0,
            **kwargs,
        )

        self.init_load = self.init_load_v3
        self.step_size = 0.01
        self._max_episode_steps = self.episode_steps_v3

        self._dof_names = [self._name_of(d) for d in self.model.dofs()]
        self._dof_index = {n: i for i, n in enumerate(self._dof_names)}
        self._actuator_names = [self._name_of(a) for a in self.model.actuators()]

        self._validate_model()

        self._base_q = np.asarray(self.init_dof_pos, dtype=np.float64).copy()
        self._base_dq = np.asarray(self.init_dof_vel, dtype=np.float64).copy()
        self._base_pelvis_y = float(self._base_q[self._dof_index["pelvis_ty"]])

        try:
            total_mass = float(self.model.mass())
        except Exception:
            total_mass = None
        self._body_weight_n = (
            total_mass * 9.81 if total_mass is not None else None
        )

        self.prev_action = np.zeros(9, dtype=np.float32)
        self.current_action = np.zeros(9, dtype=np.float32)
        self.rwd_dict = None
        self._rng = np.random.RandomState(0)

        self.action_space = gym.spaces.Box(
            low=-np.ones(9, dtype=np.float32),
            high=np.ones(9, dtype=np.float32),
            dtype=np.float32,
        )

    @staticmethod
    def _name_of(obj):
        try:
            return str(obj.name())
        except Exception:
            return str(obj.name)

    @staticmethod
    def _vec3_array(vec):
        try:
            return np.asarray(vec.array(), dtype=np.float64)
        except Exception:
            values = []
            for attr in ("x", "y", "z"):
                try:
                    values.append(float(getattr(vec, attr)))
                except Exception:
                    values.append(0.0)
            return np.asarray(values, dtype=np.float64)

    def _find_body(self, name):
        try:
            return self.model.find_body(name)
        except Exception:
            for b in self.model.bodies():
                if self._name_of(b) == name:
                    return b
        return None

    def _get_crutch_contact_force(self):
        total_force = 0.0
        for name in self.CANE_BODY_NAMES:
            body = self._find_body(name)
            if body is None:
                continue
            try:
                force = body.contact_force()
                fy = float(force.y) if hasattr(force, "y") else float(force[1])
                total_force += max(0.0, fy)
            except Exception as e:
                if not self._crutch_warned:
                    print("[Crutch force error]", repr(e))
                    self._crutch_warned = True
        return total_force

    def _crutch_forward_offsets(self):
        pelvis_x = None
        try:
            pelvis_x = float(self.model.com_pos().x)
        except Exception:
            return []

        offsets = []
        for name in self.CANE_BODY_NAMES:
            body = self._find_body(name)
            if body is None:
                continue
            try:
                pos = self._vec3_array(body.com_pos())
                if pos.shape == (3,) and np.isfinite(pos[0]):
                    offsets.append(float(pos[0]) - pelvis_x)
            except Exception:
                continue
        return offsets

    def _validate_model(self):
        if len(self.model.dofs()) != 16:
            raise RuntimeError(f"Expected 16 DOFs, found {len(self.model.dofs())}")
        if len(self.model.actuators()) != 9:
            raise RuntimeError(
                f"Expected 9 actuators, found {len(self.model.actuators())}"
            )
        if len(self.model.muscles()) != 0:
            raise RuntimeError(
                f"Expected torque-only model with 0 muscles, found {len(self.model.muscles())}"
            )
        missing = [n for n in self.ACTUATOR_NAMES if n not in self._dof_index]
        if missing:
            raise RuntimeError(f"Missing controlled DOFs: {missing}")
        if tuple(self._actuator_names) != self.ACTUATOR_NAMES:
            raise RuntimeError(
                "Unexpected actuator order.\n"
                f"Expected: {self.ACTUATOR_NAMES}\n"
                f"Found:    {tuple(self._actuator_names)}"
            )
        for n in ("ankle_angle_r", "mtp_angle_r", "ankle_angle_l", "mtp_angle_l"):
            if n not in self._dof_index:
                raise RuntimeError(f"Missing locked DOF: {n}")

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        self._rng = np.random.RandomState(int(seed))
        return [int(seed)]

    def _pelvis_forward_term(self):

        q = np.asarray(
            self.model.dof_position_array(),
            dtype=np.float64
        )

        pelvis_x = q[self._dof_index["pelvis_tx"]]

        return float(
            np.clip(
                pelvis_x / self.target_displacement_cap,
                0,
                1
            )
        )


    def _pelvis_lag_penalty(self):

        try:
            pelvis_x = float(
                self.model.com_pos().x
            )

            r = self._find_body("calcn_r")
            l = self._find_body("calcn_l")

            if r is None or l is None:
                return 0.0

            rx = self._vec3_array(r.com_pos())[0]
            lx = self._vec3_array(l.com_pos())[0]

            foot_x = min(rx, lx)

            lag = foot_x - pelvis_x

            return float(max(0.0, lag))

        except Exception:
            return 0.0
    def _setup_action_observation_spaces(self):
        num_act = len(self.model.actuators())
        self.action_space = gym.spaces.Box(
            low=-np.ones(num_act, dtype=np.float32),
            high=np.ones(num_act, dtype=np.float32),
            dtype=np.float32,
        )
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-10000, high=10000, shape=obs.shape, dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self.seed(seed)

        self.episode_number = self._rng.randint(0, 1_000_000)
        self.model.reset()
        self.has_reset = True
        self.time = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.fall_time = -1.0
        self.prev_action[:] = 0.0
        self.current_action[:] = 0.0

        self.model.set_store_data(self.store_next)

        q = self._base_q.copy()
        dq = np.zeros_like(self._base_dq)

        for name in self.RANDOMIZED_Q:
            i = self._dof_index[name]
            std = (
                self.pelvis_height_position_std
                if name == "pelvis_ty"
                else self.reset_position_std
            )
            q[i] += self._rng.normal(0.0, std)

        q[self._dof_index["pelvis_tx"]] = 0.0

        for name in (
            "ankle_angle_r", "mtp_angle_r",
            "ankle_angle_l", "mtp_angle_l",
        ):
            q[self._dof_index[name]] = 0.0
            dq[self._dof_index[name]] = 0.0

        for name in self.RANDOMIZED_DQ:
            dq[self._dof_index[name]] = self._rng.normal(
                0.0, self.reset_velocity_std
            )
        vx0 = self._rng.normal(
            self.initial_forward_velocity,
            self.initial_forward_velocity_std,
        )
        dq[self._dof_index["pelvis_tx"]] = max(0.0, float(vx0))

        self.model.set_dof_positions(q)
        self.model.set_dof_velocities(dq)
        self.model.init_state_from_dofs()
        if self.init_load > 0:
            self.model.adjust_state_for_load(self.init_load)

        obs = self._get_obs()
        if return_info:
            return obs, {}
        return obs

    def _rate_limit(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (9,):
            raise ValueError(f"Expected action shape (9,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        delta = np.clip(
            action - self.prev_action,
            -self.action_rate_limit,
            self.action_rate_limit,
        )
        limited = np.clip(self.prev_action + delta, -1.0, 1.0)
        self.prev_action = limited.copy()
        return limited

    def step(self, action):
        if not self.has_reset:
            raise RuntimeError("Call reset() before step().")

        normalized = self._rate_limit(action)
        self.current_action = normalized.copy()
        torques = normalized * self.torque_scales

        self.model.set_actuator_inputs(torques)
        self.model.advance_simulation_to(self.time + self.step_size)

        reward = self._get_rew()
        obs = self._get_obs()
        done = self._get_done()

        reward = self._apply_termination_cost(reward, done)
        self.time += self.step_size
        self.total_reward += reward

        info = {
            "curriculum_stage": self.curriculum_stage,
            **dict(self.rwd_dict),
        }

        if done:
            if self.store_next:
                self.model.write_results(
                    self.output_dir,
                    f"{self.episode:05d}_{self.total_reward:.3f}",
                )
                self.store_next = False
            self.episode += 1

        return obs, float(reward), bool(done), info

    def _height_term(self):
        q = np.asarray(self.model.dof_position_array(), dtype=np.float64)
        y = q[self._dof_index["pelvis_ty"]]
        drop = max(0.0, self._base_pelvis_y - y)
        return float(np.exp(-(drop / self.height_drop_sigma) ** 2))

    def _posture_term(self):
        q = np.asarray(self.model.dof_position_array(), dtype=np.float64)

        tilt = q[self._dof_index["pelvis_tilt"]]
        lumbar = q[self._dof_index["lumbar_extension"]]
        trunk_term = np.exp(
            -(tilt / self.pelvis_tilt_sigma) ** 2
            - (lumbar / self.lumbar_sigma) ** 2
        )

        hip_knee_names = (
            "hip_flexion_r", "knee_angle_r",
            "hip_flexion_l", "knee_angle_l",
        )
        dev_sq_sum = 0.0
        for name in hip_knee_names:
            idx = self._dof_index[name]
            dev_sq_sum += (q[idx] - self._base_q[idx]) ** 2
        stance_term = np.exp(-dev_sq_sum / (self.hip_knee_posture_sigma ** 2))

        return float(trunk_term * stance_term)

    def _crutch_term(self):
        if self._crutch_force_fn is None or self._body_weight_n is None:
            return 0.0
        if self._body_weight_n <= 0:
            return 0.0

        cane_force = float(self._crutch_force_fn())
        if cane_force < 20.0:
            return 0.0

        frac = cane_force / self._body_weight_n
        error = abs(frac - self.cane_target_load_fraction)
        sigma = max(self.cane_load_sigma_fraction, 1e-6)
        return float(np.exp(-(error / sigma) ** 2))

    def _velocity_term(self):
        v = float(self.model.com_vel().x)
        error = v - self.target_vel_v3
        return float(np.exp(-0.5 * (error / self.velocity_sigma) ** 2))

    def _backward_term(self):
        v = float(self.model.com_vel().x)
        if v >= 0:
            return 0.0
        denom = max(self.target_vel_v3, 1e-6)
        penalty = abs(v) / denom
        return float(np.clip(penalty, 0.0, 5.0))

    def _displacement_term(self):
        q = np.asarray(self.model.dof_position_array(), dtype=np.float64)
        dx = abs(q[self._dof_index["pelvis_tx"]])
        excess = max(0.0, dx - self.target_displacement_cap)
        return float(excess)

    # ------------------------------------------------------------------
    # FIX #2: the new, direct crutch-forward-placement term
    # ------------------------------------------------------------------

    def _crutch_forward_term(self):
        offsets = self._crutch_forward_offsets()
        if not offsets:
            return 0.0

        scores = []
        for offset in offsets:
            # offset > 0 (crutch ahead of pelvis) or within the allowed
            # trailing margin -> full score. Beyond the margin behind the
            # pelvis, score falls off with a Gaussian.
            lag = max(0.0, -offset - self.crutch_forward_margin)
            scores.append(float(np.exp(-(lag / self.crutch_forward_sigma) ** 2)))

        return float(np.mean(scores))

    def _update_rwd_dict(self):
        self.rwd_dict = {
            "alive": self.alive_reward_coeff,
            "height": self.height_reward_coeff * self._height_term(),
            "posture": self.posture_reward_coeff * self._posture_term(),
            "crutch": self.crutch_reward_coeff * self._crutch_term(),
            "crutch_forward": self.crutch_forward_reward_coeff
            * self._crutch_forward_term(),
            "velocity": self.velocity_reward_coeff * self._velocity_term(),
            "backward": -self.backward_penalty_coeff * self._backward_term(),
            "displacement": -self.displacement_reward_coeff
            * self._displacement_term(),
            "pelvis_forward":
                self.pelvis_forward_reward_coeff
                * self._pelvis_forward_term(),

            "pelvis_lag":
                -self.pelvis_lag_penalty_coeff
                * self._pelvis_lag_penalty(),
        }
        self.rwd_dict["total"] = float(sum(self.rwd_dict.values()))
        return self.rwd_dict

    def _get_rew(self):
        self.steps += 1
        self._update_rwd_dict()
        return float(self.rwd_dict["total"])

    def get_rwd_dict(self):
        if self.rwd_dict is None:
            self._update_rwd_dict()
        return dict(self.rwd_dict)

    def _is_fall(self):
        return bool(
            self.model.com_pos().y < self.min_com_height
            or self.head_body.com_pos().y < self.min_head_height
        )

    def _get_done(self):
        if self._is_fall():
            return True
        if self.steps >= self.episode_steps_v3:
            return True
        return False

    def _apply_termination_cost(self, reward, done):
        if done and self._is_fall():
            reward -= self.fall_penalty
        return reward

    @property
    def horizon(self):
        return self.episode_steps_v3
