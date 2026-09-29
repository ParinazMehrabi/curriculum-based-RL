"""A muscle-actuated full-body locomotion environment.

Drives MyoSuite's MyoFullBody model (`myo_sim/body/myobody.xml`, Apache-2.0):
26 bodies, 53 qpos, 290 Hill-type muscles, 82 kg. Unlike v4's planar
9-torque skeleton this is a 3-D musculoskeletal body, so the environment
differs from v4's in four ways that are worth stating up front.

**Muscle activation is state, and it is in the observation.** MuJoCo integrates
an activation variable per muscle (`data.act`, 290 of them here) with
first-order dynamics, so the force a muscle produces this step depends on
activations the policy set several steps ago. Leaving `act` out of the
observation would make the MDP non-Markov in exactly the way v4's hidden
`prev_action` buffer did -- but with 290 hidden variables instead of 9. It
dominates the observation vector for that reason.

**Only 17 joints are independent.** The model has 46 non-root joints, 29 of
which are driven by equality constraints: the knee's rolling contact
(`knee_angle_*_translation*`, `_rotation*`, `_beta_*`) follows `knee_angle_*`,
and the lumbar levels (`L1_L2_*` through `L4_L5_*`) distribute the three trunk
angles. Those are read, never written -- writing them at reset would fight the
solver rather than pose the model. `INDEPENDENT_JOINTS` lists the 17.

**Actions are activations in [0, 1], not torques.** The policy emits [-1, 1]
and the environment maps it, so a zero action is *half* activation, not rest.
The rate limiter from v4 is kept and `prev_action` stays in the observation.

**Out-of-plane failure is possible.** v4's model could not fall sideways or
turn; this one does both, which is why `lateral` and `heading` terms exist and
why termination checks trunk tilt in addition to height.

The reward machinery is v4's, imported rather than copied -- see rewards.py.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("v5 needs gymnasium: pip install gymnasium") from exc

from .rewards import gaussian
from .stages import StageSpec, get_stage

GRAVITY = 9.81

# The 17 joints that are not driven by an equality constraint, in model order.
# Everything else below the root follows from these.
INDEPENDENT_JOINTS: Tuple[str, ...] = (
    "flex_extension",
    "lat_bending",
    "axial_rotation",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
)

FOOT_BODIES = ("calcn_r", "calcn_l")

# Contact geometry, split fore/aft. The rear group sits on the calcn, the
# forward group on the toes; `r_foot_col3` is the rearmost and is what has to
# touch for the stance to be plantigrade rather than up on the forefoot.
#
# Heel and toe loads are reported separately because the difference between
# them *is* gait phase -- heel strike, midstance, toe-off -- and a single
# per-foot total throws that away.
HEEL_GEOMS = {
    "calcn_r": ("r_foot_col1", "r_foot_col3", "r_foot_col4"),
    "calcn_l": ("l_foot_col1", "l_foot_col3", "l_foot_col4"),
}
TOE_GEOMS = {
    "calcn_r": ("r_bofoot_col1", "r_bofoot_col2"),
    "calcn_l": ("l_bofoot_col1", "l_bofoot_col2"),
}

PELVIS_BODY = "pelvis"
TORSO_BODY = "torso"
HEAD_BODY = "head"
HIP_BODIES = ("femur_r", "femur_l")
ROOT_JOINT = "root"

WORLD_UP = np.array([0.0, 0.0, 1.0])

# Posture is derived from body *positions*, not from a body frame's axes.
#
# This model's world is z-up but its body frames are locally y-up -- the torso
# frame's own +y is what points at the sky, a leftover of the OpenSim
# convention the model was converted from. Reading a fixed local axis as
# "up" therefore gives nonsense (the first version of this file measured a
# 89-degree trunk tilt on a model standing perfectly straight, and every
# episode terminated on step 1).
#
# The pelvis-to-head vector and the hip-to-hip vector are unambiguous whatever
# the frame convention, so posture and heading are measured from those.


def _default_model_path() -> Path:
    import myosuite

    return (
        Path(myosuite.__file__).resolve().parent
        / "simhive"
        / "myo_sim"
        / "body"
        / "myobody.xml"
    )


class MyoLocomotionEnv(gym.Env):
    """One environment class for the whole muscle-locomotion curriculum."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 100}

    def __init__(
        self,
        stage: str = "A",
        model_path=None,
        frame_skip: int = 10,
        include_prev_action: bool = True,
        include_activation: bool = True,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        **overrides,
    ):
        self.spec_key = str(stage).upper()
        self.stage_spec: StageSpec = get_stage(self.spec_key).with_overrides(**overrides)
        self.curriculum_stage = self.stage_spec.name

        path = Path(model_path) if model_path else _default_model_path()
        if not path.is_file():
            raise FileNotFoundError("model not found: %s" % path)
        self.model_path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        # Let render() ask for a sensible size without editing the XML.
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 1280)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 960)

        self.frame_skip = int(frame_skip)
        self.dt = float(self.model.opt.timestep) * self.frame_skip
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = int(round(1.0 / self.dt))

        self._include_prev_action = bool(include_prev_action)
        self._include_activation = bool(include_activation)

        self.n_act = int(self.model.nu)
        self.prev_action = np.zeros(self.n_act, dtype=np.float32)

        self._validate_model()
        self._resolve_indices()

        self.body_weight = float(self.model.body_mass.sum()) * GRAVITY

        # Reset-time reference frame; re-measured on every reset.
        self._forward_ref = np.array([1.0, 0.0, 0.0])
        self._right_ref = np.array([0.0, -1.0, 0.0])
        self._heading_ref = 0.0

        # Settle the model once so height and posture are measured against a
        # pose the physics actually supports, rather than against the keyframe,
        # whose toes penetrate the floor by about a centimetre.
        self.stance_residual = float("nan")
        self.seat_residual = float("nan")
        self._neutral_qpos, self._neutral_height = self._solve_stance_pose()

        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None

        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.total_reward = 0.0
        self.term_values: Dict[str, float] = {}
        self.reward_breakdown: Dict[str, float] = {}
        self.rwd_dict: Dict[str, float] = {
            k: 0.0 for k in tuple(self.stage_spec.reward.weights) + ("alive", "total")
        }

        self.stage_spec.reward.warn_if_unsafe(gamma=0.99, label=self.curriculum_stage)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_act,), dtype=np.float32
        )
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    # -- model introspection ---------------------------------------------

    def _resolve_indices(self) -> None:
        m = self.model
        self.joint_qpos_adr: List[int] = []
        self.joint_dof_adr: List[int] = []
        missing = []
        for name in INDEPENDENT_JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                missing.append(name)
                continue
            self.joint_qpos_adr.append(int(m.jnt_qposadr[jid]))
            self.joint_dof_adr.append(int(m.jnt_dofadr[jid]))
        if missing:
            raise RuntimeError(
                "model %s is missing expected joints: %s"
                % (self.model_path.name, ", ".join(missing))
            )

        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT)
        if root < 0 or m.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("expected a free root joint named %r" % ROOT_JOINT)
        self.root_qpos_adr = int(m.jnt_qposadr[root])
        self.root_dof_adr = int(m.jnt_dofadr[root])

        self.pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)
        self.torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
        self.head_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HEAD_BODY)
        self.hip_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in HIP_BODIES
        ]
        self.foot_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in FOOT_BODIES
        ]
        required = {
            PELVIS_BODY: self.pelvis_id,
            TORSO_BODY: self.torso_id,
            HEAD_BODY: self.head_id,
        }
        required.update(dict(zip(HIP_BODIES, self.hip_ids)))
        required.update(dict(zip(FOOT_BODIES, self.foot_ids)))
        absent = sorted(n for n, i in required.items() if i < 0)
        if absent:
            raise RuntimeError(
                "model %s is missing bodies this env needs: %s"
                % (self.model_path.name, ", ".join(absent))
            )

        # Geoms belonging to each foot subtree, for per-foot contact force.
        # Geom ids per load group, ordered heel_r, toe_r, heel_l, toe_l to
        # match contact_loads().
        self._load_groups: List[set] = []
        for foot in FOOT_BODIES:
            for table in (HEEL_GEOMS, TOE_GEOMS):
                ids = set()
                for gname in table[foot]:
                    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
                    if g < 0:
                        raise RuntimeError("contact geom %r not found" % gname)
                    ids.add(g)
                self._load_groups.append(ids)

        self.foot_geoms: List[set] = []
        for bid in self.foot_ids:
            ids = set()
            for g in range(m.ngeom):
                b = int(m.geom_bodyid[g])
                # include the toes, which hang off the calcn
                while b > 0:
                    if b == bid:
                        ids.add(g)
                        break
                    b = int(m.body_parentid[b])
            self.foot_geoms.append(ids)

    def _validate_model(self) -> None:
        """Fail loudly if the model is not the one this env was written for."""
        m = self.model
        n_muscle = int(
            (m.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE).sum()
        )
        if n_muscle == 0:
            raise RuntimeError(
                "%s has no muscle actuators -- this environment is for "
                "muscle-actuated models. Did you point it at a torque model?"
                % self.model_path.name
            )
        if n_muscle != m.nu:
            warnings.warn(
                "%d of %d actuators are muscles; the rest are treated as "
                "activations in [0, 1] too, which may not be what their "
                "ctrlrange means." % (n_muscle, m.nu),
                RuntimeWarning,
                stacklevel=3,
            )
        if m.na != n_muscle:
            warnings.warn(
                "expected one activation state per muscle, got na=%d for %d "
                "muscles" % (m.na, n_muscle),
                RuntimeWarning,
                stacklevel=3,
            )
        self.n_muscle = n_muscle

    # -- the standing stance ----------------------------------------------

    # Joints the stance solve may move, against the residuals it drives to
    # zero: four foot heights, two COM-over-base offsets and two
    # trunk-verticality components, with pelvis height, root pitch and root
    # roll as three further unknowns.
    _STANCE_VARS = (
        "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
        "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
        "flex_extension", "lat_bending",
    )
    _STANCE_ZERO = (
        "hip_adduction_r", "hip_adduction_l", "hip_rotation_r", "hip_rotation_l",
        "subtalar_angle_r", "subtalar_angle_l", "mtp_angle_r", "mtp_angle_l",
        "axial_rotation",
    )

    def _support_centroid(self) -> np.ndarray:
        """Horizontal centre of the base of support.

        The mean of the four contact groups' geom positions -- heel and toe of
        each foot -- rather than the midpoint of the two calcn bodies. The
        calcn COM sits forward of the foot's true support area, so balancing
        the body COM over it left the model pitching onto its toes and
        unloading both heels within four steps.
        """
        pts = []
        for group in self._load_groups:
            for g in group:
                pts.append(np.asarray(self.data.geom_xpos[g]))
        return np.mean(pts, axis=0)

    def _geom_lowest_z(self, name: str) -> float:
        """World z of the lowest point of a contact geom."""
        g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            raise RuntimeError("contact geom %r not found" % name)
        size = self.model.geom_size[g]
        radius = (
            float(size.min())
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_ELLIPSOID
            else float(size[0])
        )
        return float(self.data.geom_xpos[g][2] - radius)

    def _heel_z(self, foot: str) -> float:
        return min(self._geom_lowest_z(n) for n in HEEL_GEOMS[foot])

    def _toe_z(self, foot: str) -> float:
        return min(self._geom_lowest_z(n) for n in TOE_GEOMS[foot])

    _SEAT_VARS = (
        "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
        "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    )

    def _seat_feet(self, iterations: int = 60) -> None:
        """Put both heels and both toes back on the floor after randomisation.

        Reset perturbs the independent joints, which breaks the stance solve in
        two ways: tilting a foot about its ankle lifts the heel (0.02 rad moves
        a 0.2 m foot by 4 mm), and perturbing a hip or knee changes that leg's
        length and lifts the whole foot, by up to 30 mm in practice.

        Solving each leg on its own is not enough. The knee's lower limit is
        full extension, so once a leg is straight it cannot lengthen further,
        and the per-leg Newton stalls a millimetre or two short however many
        iterations it is given. The pelvis has to move too.

        So this shares the stance solve's unknowns -- six leg angles plus
        pelvis height and a root pitch/roll delta -- against six residuals:
        four foot heights and the two horizontal offsets of the COM from the
        base of support. Nine unknowns, six constraints, solved least-norm, so
        the seating disturbs the randomised pose as little as it can while
        guaranteeing the model starts balanced with both feet flat.

        The trunk joints keep whatever randomisation gave them; what it cannot
        do is take a foot off the ground or put the COM outside the base, which
        for a standing reset are the right constraints.
        """
        m, d = self.model, self.data
        adr, lim = [], []
        for joint in self._SEAT_VARS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
            adr.append(int(m.jnt_qposadr[jid]))
            lim.append(m.jnt_range[jid])
        n_var = len(self._SEAT_VARS)
        lo_b = np.array([l[0] for l in lim])
        hi_b = np.array([l[1] for l in lim])
        base_quat = d.qpos[3:7].copy()

        def residual(x):
            for a, v, (lo, hi) in zip(adr, x[:n_var], lim):
                d.qpos[a] = float(np.clip(v, lo, hi))
            d.qpos[2] = x[n_var]
            q = base_quat.copy()
            for angle, axis in (
                (x[n_var + 1], (0.0, 1.0, 0.0)),
                (x[n_var + 2], (1.0, 0.0, 0.0)),
            ):
                dq = np.array(
                    [np.cos(angle / 2.0), *(np.sin(angle / 2.0) * np.asarray(axis))]
                )
                out = np.zeros(4)
                mujoco.mju_mulQuat(out, dq, q)
                q = out
            d.qpos[3:7] = q
            mujoco.mj_forward(m, d)
            com = np.asarray(d.subtree_com[0])
            mid = self._support_centroid()
            return np.array([
                self._heel_z("calcn_r"), self._toe_z("calcn_r"),
                self._heel_z("calcn_l"), self._toe_z("calcn_l"),
                com[0] - mid[0], com[1] - mid[1],
            ])

        x = np.concatenate([[d.qpos[a] for a in adr], [d.qpos[2]], [0.0, 0.0]])
        for _ in range(iterations):
            r = residual(x)
            if np.abs(r).max() < 1e-7:
                break
            jac = np.zeros((r.size, x.size))
            eps = 1e-5
            for k in range(x.size):
                xp = x.copy()
                xp[k] += eps
                jac[:, k] = (residual(xp) - r) / eps
            step = -np.linalg.solve(jac.T @ jac + 1e-9 * np.eye(x.size), jac.T @ r)
            norm = np.linalg.norm(step)
            if norm > 0.05:
                step *= 0.05 / norm
            x = x + step
            x[:n_var] = np.clip(x[:n_var], lo_b, hi_b)
        self.seat_residual = float(np.abs(residual(x)).max())

    def _solve_stance_pose(self):
        """Solve for a plantigrade, balanced, upright standing pose.

        The shipped keyframe is a mid-stride pose, not a stance: the hips
        differ by 0.43 rad, hip_rotation_r is -35 degrees, the feet are 0.23 m
        apart along the facing direction and the trunk is flexed 30 degrees.
        Dropping the model from it lands it on its toes with both heels in the
        air -- a bad place to start a locomotion curriculum. It is
        near-singular, it biases the ankle plantarflexors from step one, and it
        leaves no heel contact for a gait reward to read.

        A symmetric pose cannot fix it either. With identical joint angles and
        level hips the right femur is 23.5 mm shorter than the left, so one
        foot is always off the ground. The solve therefore treats the legs
        independently and lets lat_bending absorb the difference, which is what
        a person with a leg-length discrepancy does.

        Residuals, driven to zero by damped Gauss-Newton:

          * each foot's heel and toe both at z = 0  (plantigrade, both feet)
          * the COM horizontally over the midpoint of the feet  (balanced)
          * the trunk axis vertical  (upright)

        Returns (qpos, pelvis_height), and raises rather than quietly handing
        back a pose that is none of those things.
        """
        m, d = self.model, self.data
        names = self._STANCE_VARS + self._STANCE_ZERO
        qadr = {
            n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in names
        }
        limits = [
            m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self._STANCE_VARS
        ]
        n_var = len(self._STANCE_VARS)
        upright = np.array([1.0, 0.0, 0.0, 0.0])

        mujoco.mj_resetDataKeyframe(m, d, 0)
        for n in self._STANCE_ZERO:
            d.qpos[qadr[n]] = 0.0
        d.qvel[:] = 0.0

        def residual(x):
            for n, v, (lo, hi) in zip(self._STANCE_VARS, x[:n_var], limits):
                d.qpos[qadr[n]] = float(np.clip(v, lo, hi))
            d.qpos[2] = x[n_var]
            q = upright.copy()
            for angle, axis in (
                (x[n_var + 1], (0.0, 1.0, 0.0)),
                (x[n_var + 2], (1.0, 0.0, 0.0)),
            ):
                dq = np.array(
                    [np.cos(angle / 2.0), *(np.sin(angle / 2.0) * np.asarray(axis))]
                )
                out = np.zeros(4)
                mujoco.mju_mulQuat(out, dq, q)
                q = out
            d.qpos[3:7] = q
            mujoco.mj_forward(m, d)
            com = np.asarray(d.subtree_com[0])
            mid = self._support_centroid()
            up = self.trunk_axis()
            return np.array([
                self._heel_z("calcn_r"), self._toe_z("calcn_r"),
                self._heel_z("calcn_l"), self._toe_z("calcn_l"),
                com[0] - mid[0], com[1] - mid[1],
                up[0], up[1],
            ])

        x = np.zeros(n_var + 3)
        x[self._STANCE_VARS.index("knee_angle_r")] = 0.10
        x[self._STANCE_VARS.index("knee_angle_l")] = 0.10
        x[n_var] = 0.95
        lo_b = np.array([l[0] for l in limits])
        hi_b = np.array([l[1] for l in limits])

        # Two passes: the four foot residuals first, then all eight. Starting
        # the full solve from a pose whose feet are already flat is what makes
        # it converge; from the keyframe the COM and foot residuals pull
        # against each other and it wanders into the joint limits instead.
        for n_res in (4, 8):
            for _ in range(200):
                r = residual(x)[:n_res]
                if np.abs(r).max() < 1e-7:
                    break
                jac = np.zeros((n_res, x.size))
                eps = 1e-5
                for k in range(x.size):
                    xp = x.copy()
                    xp[k] += eps
                    jac[:, k] = (residual(xp)[:n_res] - r) / eps
                step = -np.linalg.solve(
                    jac.T @ jac + 1e-9 * np.eye(x.size), jac.T @ r
                )
                norm = np.linalg.norm(step)
                if norm > 0.05:
                    step *= 0.05 / norm
                x = x + step
                x[:n_var] = np.clip(x[:n_var], lo_b, hi_b)

        r = residual(x)
        worst = float(np.abs(r).max())
        if worst > 1e-4:
            raise RuntimeError(
                "could not solve a plantigrade standing stance for %s "
                "(worst residual %.2e). The model's foot geometry or joint "
                "ranges may differ from what this env expects."
                % (self.model_path.name, worst)
            )
        self.stance_residual = worst
        return d.qpos.copy(), float(d.xipos[self.pelvis_id][2])

    # -- observation -------------------------------------------------------

    @property
    def pelvis_height(self) -> float:
        return float(self.data.xipos[self.pelvis_id][2])

    def trunk_axis(self) -> np.ndarray:
        """Unit vector from the pelvis COM to the head COM."""
        v = np.asarray(self.data.xipos[self.head_id]) - np.asarray(
            self.data.xipos[self.pelvis_id]
        )
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else WORLD_UP.copy()

    def trunk_tilt(self) -> float:
        """Angle between the trunk axis and world vertical, in radians.

        About 0.14 rad on the settled standing pose.
        """
        return float(np.arccos(np.clip(float(self.trunk_axis() @ WORLD_UP), -1.0, 1.0)))

    def right_axis(self) -> np.ndarray:
        """Horizontal unit vector pointing to the model's right, from the hips."""
        v = np.asarray(self.data.xipos[self.hip_ids[0]]) - np.asarray(
            self.data.xipos[self.hip_ids[1]]
        )
        v = np.array([v[0], v[1], 0.0])
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.array([0.0, -1.0, 0.0])

    def forward_axis(self) -> np.ndarray:
        """Horizontal unit vector the model currently faces."""
        return np.cross(WORLD_UP, self.right_axis())

    def heading(self) -> float:
        """Current facing direction as a yaw angle, in radians."""
        f = self.forward_axis()
        return float(np.arctan2(f[1], f[0]))

    def heading_error(self) -> float:
        """Signed yaw away from the direction faced at reset, wrapped to +-pi.

        Measured against the reset heading rather than world +x, because this
        model's neutral pose faces about 109 degrees and "walk forward" means
        the way it started, not the way the world axes happen to point.
        """
        err = self.heading() - self._heading_ref
        return float((err + np.pi) % (2.0 * np.pi) - np.pi)

    def planar_velocity(self) -> Tuple[float, float]:
        """COM velocity resolved into (forward, lateral) at the reset heading."""
        v = self.com_velocity()
        return float(v @ self._forward_ref), float(v @ self._right_ref)

    def com_velocity(self) -> np.ndarray:
        mujoco.mj_subtreeVel(self.model, self.data)
        return np.asarray(self.data.subtree_linvel[0]).copy()

    def contact_loads(self) -> np.ndarray:
        """Vertical load on [heel_r, toe_r, heel_l, toe_l], as a fraction of BW.

        Heel and toe are kept apart because the difference between them is
        gait phase -- heel strike loads the rear group, toe-off the forward
        one -- and a single per-foot total cannot tell those apart.
        """
        out = np.zeros(4)
        buf = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            mujoco.mj_contactForce(self.model, self.data, i, buf)
            frame = np.asarray(con.frame).reshape(3, 3)
            fz = float((frame.T @ buf[:3])[2])
            for k, geoms in enumerate(self._load_groups):
                if g2 in geoms:
                    out[k] += fz
                elif g1 in geoms:
                    out[k] -= fz
        return np.maximum(out, 0.0) / self.body_weight

    def foot_contact_loads(self) -> np.ndarray:
        """Vertical load per foot (heel + toe), as a fraction of body weight."""
        loads = self.contact_loads()
        return np.array([loads[0] + loads[1], loads[2] + loads[3]])

    def heel_contact_loads(self) -> np.ndarray:
        """Vertical load under each heel, as a fraction of body weight."""
        loads = self.contact_loads()
        return np.array([loads[0], loads[2]])

    def _get_obs(self) -> np.ndarray:
        d, m = self.data, self.model
        vf, vl = self.planar_velocity()
        parts = [
            [self.pelvis_height],
            self.trunk_axis(),                     # orientation, no wrap-around
            [np.sin(self.heading_error()), np.cos(self.heading_error())],
            d.qvel[self.root_dof_adr : self.root_dof_adr + 6],
            d.qpos[self.joint_qpos_adr],
            d.qvel[self.joint_dof_adr],
            [vf, vl, float(self.com_velocity()[2])],
            self.contact_loads(),
        ]
        if self._include_activation:
            # The 290 muscle activation states. Without these the observation
            # does not determine the next state; see the module docstring.
            parts.append(d.act[: m.na] if m.na else np.zeros(0))
        if self._include_prev_action:
            parts.append(self.prev_action)
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts]).astype(
            np.float32
        )

    def obs_layout(self) -> List[Tuple[str, int]]:
        """Names and widths of the observation blocks, in order."""
        layout = [
            ("pelvis_height", 1),
            ("trunk_axis", 3),
            ("heading_err_sin_cos", 2),
            ("root_vel", 6),
            ("joint_q", len(self.joint_qpos_adr)),
            ("joint_dq", len(self.joint_dof_adr)),
            ("com_vel_fwd_lat_up", 3),
            ("contact_load_heel_toe", 4),
        ]
        if self._include_activation:
            layout.append(("muscle_activation", int(self.model.na)))
        if self._include_prev_action:
            layout.append(("prev_action", self.n_act))
        return layout

    # -- episode -----------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        spec = self.stage_spec

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._neutral_qpos
        self.data.qvel[:] = 0.0

        # Perturb only the independent joints. The constrained ones are left
        # for mj_forward to resolve from these.
        for adr in self.joint_qpos_adr:
            self.data.qpos[adr] += self.rng.normal(0.0, spec.reset_position_std)
        for adr in self.joint_dof_adr:
            self.data.qvel[adr] += self.rng.normal(0.0, spec.reset_velocity_std)

        # Resolve kinematics before touching the root velocity: the facing
        # direction is read off body positions, so it needs the pose settled.
        mujoco.mj_forward(self.model, self.data)

        # The stance solve puts both heels and both toes exactly on the floor,
        # but the randomisation above then tilts the feet: 0.02 rad at the
        # ankle moves a 0.2 m foot by 4 mm, which is enough to lift a heel off
        # a perfectly flat solve entirely. Re-flatten each foot through its own
        # ankle, then re-seat the model vertically. Without this the right heel
        # carried no load at all on some resets, and penetration on others made
        # the reset load reach 1.5x body weight.
        self._seat_feet()

        # Freeze the heading the episode starts from. Everything directional --
        # the velocity, lateral and heading terms -- is measured against this,
        # so "forward" means the way the model was facing when it started.
        self._heading_ref = self.heading()
        self._forward_ref = self.forward_axis()
        self._right_ref = self.right_axis()

        if spec.initial_forward_velocity > 0.0:
            # Along the model's own facing direction, not world +x. This pose
            # faces about 109 degrees, so pushing along +x would launch it
            # mostly sideways -- which is what the first version did, and the
            # lateral term then punished the env's own initial condition.
            speed = max(
                0.0,
                float(
                    self.rng.normal(
                        spec.initial_forward_velocity,
                        spec.initial_forward_velocity_std,
                    )
                ),
            )
            self.data.qvel[self.root_dof_adr : self.root_dof_adr + 3] = (
                speed * self._forward_ref
            )

        if self.model.na:
            self.data.act[:] = spec.initial_activation
        self.data.ctrl[:] = spec.initial_activation
        self.prev_action[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.total_reward = 0.0
        self.term_values = {}
        self.reward_breakdown = {}
        for key in self.rwd_dict:
            self.rwd_dict[key] = 0.0

        return self._get_obs(), {"curriculum_stage": self.curriculum_stage}

    def _rate_limit(self, action) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (self.n_act,):
            raise ValueError(
                "expected action shape (%d,), got %r" % (self.n_act, action.shape)
            )
        limit = self.stage_spec.action_rate_limit
        delta = np.clip(action - self.prev_action, -limit, limit)
        self.prev_action = np.clip(self.prev_action + delta, -1.0, 1.0).astype(np.float32)
        return self.prev_action

    def step(self, action):
        limited = self._rate_limit(action)
        # Policy space [-1, 1] -> activation [0, 1]. A zero action is therefore
        # half activation, not rest.
        self.data.ctrl[:] = 0.5 * (limited.astype(np.float64) + 1.0)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        reward = self._get_reward()
        terminated = self._is_fallen()
        truncated = self.steps >= self.stage_spec.episode_steps
        if terminated:
            reward -= self.stage_spec.reward.fall_penalty
        self.total_reward += reward

        info = {"curriculum_stage": self.curriculum_stage}
        info.update(self.rwd_dict)
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # -- reward terms, all in [0, 1] --------------------------------------

    def _term_height(self) -> float:
        drop = max(0.0, self._neutral_height - self.pelvis_height)
        return gaussian(drop, self.stage_spec.terms.height_drop_sigma)

    def _term_upright(self) -> float:
        return gaussian(self.trunk_tilt(), self.stage_spec.terms.upright_sigma)

    def _term_velocity(self) -> float:
        forward, _ = self.planar_velocity()
        return gaussian(
            forward - self.stage_spec.target_vel, self.stage_spec.terms.velocity_sigma
        )

    def _term_lateral(self) -> float:
        _, lateral = self.planar_velocity()
        return gaussian(lateral, self.stage_spec.terms.lateral_sigma)

    def _term_heading(self) -> float:
        return gaussian(self.heading_error(), self.stage_spec.terms.heading_sigma)

    def _term_effort(self) -> float:
        """Reward low mean activation, but only above the target budget.

        Below `effort_target` the term is flat at 1.0. Rewarding ever-lower
        activation would fight every other term, since the cheapest posture is
        no posture at all.
        """
        if not self.model.na:
            return 1.0
        mean_act = float(np.mean(self.data.act[: self.model.na]))
        excess = max(0.0, mean_act - self.stage_spec.terms.effort_target)
        return gaussian(excess, self.stage_spec.terms.effort_sigma)

    _TERM_FNS = {
        "height": _term_height,
        "upright": _term_upright,
        "velocity": _term_velocity,
        "lateral": _term_lateral,
        "heading": _term_heading,
        "effort": _term_effort,
    }

    def compute_terms(self) -> Dict[str, float]:
        out = {}
        for name in self.stage_spec.reward.required_terms:
            try:
                fn = self._TERM_FNS[name]
            except KeyError:
                raise KeyError(
                    "stage %s wants unknown term %r; known: %s"
                    % (self.curriculum_stage, name, ", ".join(sorted(self._TERM_FNS)))
                ) from None
            out[name] = float(fn(self))
        return out

    def _get_reward(self) -> float:
        self.term_values = self.compute_terms()
        total, breakdown = self.stage_spec.reward.compose(self.term_values)
        self.reward_breakdown = breakdown
        for key in self.rwd_dict:
            self.rwd_dict[key] = float(breakdown.get(key, 0.0))
        return float(total)

    # -- termination -------------------------------------------------------

    def _is_fallen(self) -> bool:
        spec = self.stage_spec
        return bool(
            self.pelvis_height < spec.min_pelvis_height
            or self.trunk_tilt() > spec.max_trunk_tilt
        )

    # -- rendering ---------------------------------------------------------

    def render(self, width: int = 640, height: int = 480, camera=None):
        mode = self.render_mode or "rgb_array"
        if mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer.update_scene(
                self.data, camera=camera if camera is not None else -1
            )
            return self._renderer.render()
        if mode == "human":
            import mujoco.viewer

            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
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

    # -- description -------------------------------------------------------

    def describe(self) -> str:
        return (
            "%s | %s | %d muscles, %d independent joints, %.1f kg | "
            "obs %d, act %d | dt %.3f s"
            % (
                self.curriculum_stage,
                self.model_path.name,
                self.n_muscle,
                len(INDEPENDENT_JOINTS),
                self.body_weight / GRAVITY,
                self.observation_space.shape[0],
                self.n_act,
                self.dt,
            )
        )
