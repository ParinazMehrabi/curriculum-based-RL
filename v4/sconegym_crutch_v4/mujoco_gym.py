"""A stand-in for `sconegym.gaitgym.GaitGym`, backed by MuJoCo.

`CrutchCurriculumGym` subclasses GaitGym for the plumbing -- gym spaces, the
base observation, episode bookkeeping, the `self.model` handle -- and supplies
the curriculum on top. This class provides that same plumbing over
`mujoco_backend.MujocoModel`, so the curriculum code above it is unchanged.

It accepts GaitGym's constructor keywords so `env.py`'s `super().__init__(...)`
call needs no branching, and it validates rather than ignores the ones that
would change behaviour if they were wrong.

The observation is not GaitGym's
--------------------------------
sconegym's 2D observation layout is defined inside sconegym, which is not a
dependency of this backend and is not installed on every machine that will run
it. Rather than guess at the layout and be subtly wrong, this class defines its
own, documented in `OBS_LAYOUT` and pinned by a test.

The consequence is the one that always follows an observation change, and it is
the same one the v4 README records for the v3 -> v4 transition: **checkpoints
do not transfer between the SCONE and MuJoCo backends.** The actor's input
layer has a different width and a different meaning per element. Train a
curriculum on one backend or the other, not half on each.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np

from .mujoco_backend import GRAVITY, MujocoModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = REPO_ROOT / "models" / "mjcf" / "rajagopal_crutch_2d.xml"
DEFAULT_INIT_STATE = (
    REPO_ROOT / "models" / "init_states" / "InitState_A0_walk_003_v2.zml"
)

# The nine dofs that carry an actuator, in actuator order.
ACTUATED_DOFS = (
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
ROOT_DOFS = ("pelvis_tx", "pelvis_ty", "pelvis_tilt")

# Bodies whose vertical contact load goes into the observation.
CONTACT_BODIES = ("calcn_r", "calcn_l", "Crutch_r", "Crutch_l")

OBS_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("pelvis_height", 1),           # absolute, metres
    ("pelvis_tilt_sin_cos", 2),     # angle as sin/cos, so it cannot wrap
    ("actuated_q", len(ACTUATED_DOFS)),
    ("root_dq", len(ROOT_DOFS)),
    ("actuated_dq", len(ACTUATED_DOFS)),
    ("com_vel_xy", 2),
    ("contact_load", len(CONTACT_BODIES)),  # vertical force / body weight
)
OBS_SIZE = sum(n for _, n in OBS_LAYOUT)

# pelvis_tx is deliberately absent: absolute forward position is not part of
# the task, and including it would let the policy read the episode clock. The
# reward's travel terms read it off the model directly instead.


class MujocoGaitGym(gym.Env):
    """GaitGym's interface over a MuJoCo model."""

    metadata = {"render.modes": ["human", "rgb_array"], "render_modes": ["human", "rgb_array"]}

    BACKEND = "mujoco"
    OBS_LAYOUT = OBS_LAYOUT
    OBS_SIZE = OBS_SIZE

    def __init__(
        self,
        model_file=None,
        left_leg_idxs=None,
        right_leg_idxs=None,
        root_body_name: str = "pelvis",
        foot_body_name: str = "calcn",
        target_vel: float = 0.0,
        leg_switch: bool = False,
        clip_actions: bool = False,
        obs_type: str = "2D",
        min_com_height: float = 0.55,
        min_head_height: float = 0.75,
        fall_recovery_time: float = 0.0,
        head_body_name: str = "torso",
        init_state_file=None,
        output_dir=None,
        **unused,
    ):
        if obs_type != "2D":
            raise ValueError(
                "the mujoco backend models a sagittal-plane skeleton and only "
                "supports obs_type='2D', got %r" % obs_type
            )
        if leg_switch:
            raise ValueError("leg_switch is not implemented by the mujoco backend")
        if unused:
            # Quietly dropping a keyword here would mean training against a
            # setting nobody chose, which is the failure v4 already fixed once
            # for stage overrides.
            raise TypeError(
                "unexpected keyword(s) for the mujoco backend: %s"
                % ", ".join(sorted(unused))
            )

        model_file = Path(model_file) if model_file else DEFAULT_MJCF
        if model_file.suffix.lower() != ".xml":
            raise ValueError(
                "the mujoco backend needs an MJCF .xml, got %s.\nGenerate one with "
                "`python v4/scripts/hfd_to_mjcf.py`." % model_file.name
            )
        init_state_file = Path(init_state_file) if init_state_file else DEFAULT_INIT_STATE

        self.model = MujocoModel(model_file, init_state_path=init_state_file)
        self.model_file = model_file

        self.target_vel = float(target_vel)
        self.clip_actions = bool(clip_actions)
        self.min_com_height = float(min_com_height)
        self.min_head_height = float(min_head_height)
        self.fall_recovery_time = float(fall_recovery_time)

        self.root_body = self.model.find_body(root_body_name)
        self.head_body = self.model.find_body(head_body_name)
        if self.root_body is None:
            raise RuntimeError("root body %r not found" % root_body_name)
        if self.head_body is None:
            raise RuntimeError("head body %r not found" % head_body_name)
        if head_body_name == "torso":
            # The Rajagopal crutch model merges the skull into `torso`, so
            # there is no separate head body to measure. min_head_height is
            # therefore read against the torso COM (~1.35 m in the init pose),
            # not a head. Worth knowing if you port a threshold from elsewhere.
            pass

        self._dof_names = self.model.dof_names()
        self._dof_index = {n: i for i, n in enumerate(self._dof_names)}
        self._obs_q_idx = [self._dof_index[n] for n in ACTUATED_DOFS]
        self._obs_dq_idx = [self._dof_index[n] for n in ROOT_DOFS + ACTUATED_DOFS]
        self._contact_bodies = [self.model.find_body(n) for n in CONTACT_BODIES]
        missing = [n for n, b in zip(CONTACT_BODIES, self._contact_bodies) if b is None]
        if missing:
            raise RuntimeError("contact bodies not found: %s" % ", ".join(missing))

        self.init_dof_pos = self.model.init_q.copy()
        self.init_dof_vel = self.model.init_dq.copy()

        self.episode = 0
        self.episode_number = 0
        self.steps = 0
        self.time = 0.0
        self.total_reward = 0.0
        self.fall_time = -1.0
        self.has_reset = False
        self.store_next = False
        self.output_dir = str(output_dir) if output_dir else str(REPO_ROOT / "results_mujoco")

        self._renderer = None
        self._viewer = None

        self._setup_action_observation_spaces()

    # -- spaces and observation ------------------------------------------

    def _setup_action_observation_spaces(self) -> None:
        n_act = self.model.m.nu
        self.action_space = gym.spaces.Box(
            low=-np.ones(n_act, dtype=np.float32),
            high=np.ones(n_act, dtype=np.float32),
            dtype=np.float32,
        )
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-10000.0, high=10000.0, shape=obs.shape, dtype=np.float32
        )

    def contact_loads(self) -> np.ndarray:
        """Vertical contact force per contact body, as a fraction of body weight."""
        weight = self.model.body_weight
        return np.asarray(
            [max(0.0, b.contact_force().y) / weight for b in self._contact_bodies],
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        q = self.model.dof_position_array()
        dq = self.model.dof_velocity_array()
        tilt = float(q[self._dof_index["pelvis_tilt"]])
        com_vel = self.model.com_vel()
        obs = np.concatenate(
            [
                [q[self._dof_index["pelvis_ty"]]],
                [math.sin(tilt), math.cos(tilt)],
                q[self._obs_q_idx],
                dq[self._obs_dq_idx],
                [com_vel.x, com_vel.y],
                self.contact_loads(),
            ]
        )
        return np.asarray(obs, dtype=np.float32)

    # -- gym.Env ----------------------------------------------------------

    def reset(self, **kwargs):
        self.model.reset()
        self.has_reset = True
        self.time = 0.0
        self.steps = 0
        self.total_reward = 0.0
        self.fall_time = -1.0
        return self._get_obs()

    def step(self, action):  # pragma: no cover - CrutchCurriculumGym overrides it
        raise NotImplementedError(
            "MujocoGaitGym supplies plumbing only; step() belongs to the curriculum env"
        )

    # -- rendering --------------------------------------------------------

    def render(self, mode: str = "rgb_array", width: int = 640, height: int = 480):
        import mujoco

        if mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model.m, height=height, width=width)
            self._renderer.update_scene(self.model.d, camera="side")
            return self._renderer.render()
        if mode == "human":
            import mujoco.viewer

            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model.m, self.model.d)
            self._viewer.sync()
            return None
        raise ValueError("unsupported render mode %r" % mode)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # -- description ------------------------------------------------------

    def describe_backend(self) -> str:
        return "mujoco %s | %s | %d dofs, %d actuators, %.2f kg" % (
            _mujoco_version(),
            self.model_file.name,
            len(self._dof_names),
            self.model.m.nu,
            self.model.mass(),
        )


def _mujoco_version() -> str:
    import mujoco

    return getattr(mujoco, "__version__", "?")
