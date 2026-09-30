"""A planar, muscle-actuated locomotion environment.

MyoSuite's MyoFullBody model (`myo_sim/body/myobody.xml`, Apache-2.0) --
26 bodies, 290 Hill-type muscles, 82 kg -- restructured to move in the
sagittal plane only, matching the `.hfd` cane model this project is built
around.

Why planar
----------
The `.hfd` is a 2-D model: a planar root (`pelvis_tx`, `pelvis_ty`,
`pelvis_tilt`) and sagittal joints. MyoFullBody ships as a 3-D body with a
free root, where lateral balance and heading are extra failure modes a policy
has to solve before it can start walking at all. Removing them is the largest
single reduction in problem difficulty available here, and it is what the
source model does.

Three changes make it planar, applied through `MjSpec` at construction:

* **The free root becomes three joints** -- slide x (forward), slide z (up),
  hinge y (sagittal pitch) -- named as in the `.hfd`. The model cannot
  translate sideways, yaw or roll.
* **Eight out-of-plane joints are pinned to zero** by equality constraints:
  hip adduction and rotation, subtalar, lumbar lateral bending and axial
  rotation. Those were measured rather than guessed -- 0.15 rad of hip
  adduction moves bodies 0.127 m out of plane, lateral bending 0.089 m, the
  rest 0.015-0.021 m.
* **The foot contact becomes two balls**, as in the `.hfd`. See below.

Ankle, knee and mtp are left free. They are nominally sagittal but carry small
oblique components (4.6 mm, 1.2 mm and 0.2 mm out of plane per 0.15 rad),
which is real anatomy rather than an artefact. With a planar root those cannot
accumulate into lateral motion, so the model behaves as 2-D while its joints
stay anatomical.

What remains
------------
Twelve independent degrees of freedom: three at the root, `flex_extension` for
the trunk, and hip/knee/ankle/mtp on each leg. Twenty-nine more joints follow
equality constraints -- the knee's rolling contact and the lumbar levels'
distribution of the trunk angles -- and are read, never written; posing them
directly would fight the solver.

Muscle activation is state, and it is observed
----------------------------------------------
MuJoCo integrates an activation variable per muscle (`data.act`, 290 here)
with first-order dynamics, so the force a muscle makes this step depends on
activations set several steps ago. Leaving `act` out of the observation would
make the MDP non-Markov exactly as v4's hidden `prev_action` buffer did, but
with 290 hidden variables instead of 9. It dominates the observation for that
reason.

Actions are activations in [0, 1], not torques: the policy emits [-1, 1] and
the environment maps it, so **a zero action is half activation, not rest**.
v4's action rate limiter is kept and `prev_action` stays in the observation.

The reward machinery is v4's, imported rather than copied -- see rewards.py.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("v5 needs gymnasium: pip install gymnasium") from exc

from .reference import TRACKED_JOINTS, Reference
from .rewards import gaussian, smoothstep
from .stages import StageSpec, get_stage

GRAVITY = 9.81
WORLD_UP = np.array([0.0, 0.0, 1.0])
FORWARD = np.array([1.0, 0.0, 0.0])

# --------------------------------------------------------------------------
# Planar structure
# --------------------------------------------------------------------------

# The root, named as in the .hfd. Forward is +x, up is +z, and the sagittal
# pitch axis is +y, which is the model's own frontal axis at qpos = 0.
ROOT_JOINTS = ("pelvis_tx", "pelvis_ty", "pelvis_tilt")
ROOT_SPEC = (
    ("pelvis_tx", mujoco.mjtJoint.mjJNT_SLIDE, (1.0, 0.0, 0.0)),
    ("pelvis_ty", mujoco.mjtJoint.mjJNT_SLIDE, (0.0, 0.0, 1.0)),
    ("pelvis_tilt", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 1.0, 0.0)),
)

# Pinned to zero. Measured by perturbing each independent joint by 0.15 rad
# and recording how far any body left the sagittal plane:
#   hip_adduction 0.127 m, lat_bending 0.089, hip_rotation 0.021,
#   subtalar 0.019, axial_rotation 0.015
# against ankle 0.0046, knee 0.0012, mtp 0.00016 and flex_extension 3e-6,
# which are left free.
OUT_OF_PLANE_JOINTS = (
    "hip_adduction_r", "hip_adduction_l",
    "hip_rotation_r", "hip_rotation_l",
    "subtalar_angle_r", "subtalar_angle_l",
    "lat_bending", "axial_rotation",
)

# The nine non-root joints the policy and the reset actually drive.
INDEPENDENT_JOINTS: Tuple[str, ...] = (
    "flex_extension",
    "hip_flexion_r", "knee_angle_r", "ankle_angle_r", "mtp_angle_r",
    "hip_flexion_l", "knee_angle_l", "ankle_angle_l", "mtp_angle_l",
)

# Per-leg, in the order the seating solve uses them.
LEG_JOINTS = {
    "calcn_r": ("hip_flexion_r", "knee_angle_r", "ankle_angle_r"),
    "calcn_l": ("hip_flexion_l", "knee_angle_l", "ankle_angle_l"),
}

FOOT_BODIES = ("calcn_r", "calcn_l")
PELVIS_BODY = "pelvis"
TORSO_BODY = "torso"
HEAD_BODY = "head"

# --------------------------------------------------------------------------
# Foot contact: two balls per foot, as in the OpenSim/SCONE cane model
# --------------------------------------------------------------------------
#
# The .hfd gives each foot exactly two contact spheres -- heel and toe, radius
# 0.03. MyoFullBody instead wraps each foot in five capsules and an ellipsoid:
# a rolling sole, with no clean heel/toe split to read gait phase from. The
# MyoSuite geoms are not deleted, only taken out of collision and left in the
# visual group, so the foot still renders as a foot.
MYO_FOOT_GEOMS = (
    "r_foot_col1", "r_foot_col3", "r_foot_col4", "r_bofoot_col1", "r_bofoot_col2",
    "l_foot_col1", "l_foot_col3", "l_foot_col4", "l_bofoot_col1", "l_bofoot_col2",
)

BALL_RADIUS = 0.03

# Derived from this model's own sole geometry in the plantigrade stance -- the
# rearmost and forwardmost points of the MyoSuite sole, dropped to the floor
# and raised by one ball radius -- rather than copied from the .hfd, whose
# calcn frame is scaled and oriented differently. Mirror-symmetric, because
# the model is; `scripts/derive_ball_contacts.py` recomputes them.
#
# The heel ball sits on calcn and the toe ball on toes. The .hfd puts both on
# calcn, but its own comment says why that was free -- "the mtp joint here is
# locked 0..0 anyway, so this changes nothing kinematically". Here mtp is live
# with muscles crossing it, so the equivalent choice is to follow the toes.
BALL_CONTACTS = {
    "heel_r": ("calcn_r", (+0.055194, +0.010372, +0.020376)),
    "toe_r": ("toes_r", (+0.057671, +0.009030, +0.009809)),
    "heel_l": ("calcn_l", (+0.055194, +0.010372, -0.020376)),
    "toe_l": ("toes_l", (+0.057671, +0.009030, -0.009809)),
}

HEEL_GEOMS = {"calcn_r": ("heel_r",), "calcn_l": ("heel_l",)}
TOE_GEOMS = {"calcn_r": ("toe_r",), "calcn_l": ("toe_l",)}


def _default_model_path() -> Path:
    import myosuite

    return (
        Path(myosuite.__file__).resolve().parent
        / "simhive" / "myo_sim" / "body" / "myobody.xml"
    )


class MyoLocomotionEnv(gym.Env):
    """One environment class for the whole planar muscle-locomotion curriculum."""

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
        self.model = self._build_model(path)
        self.data = mujoco.MjData(self.model)

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

        self.stance_residual = float("nan")
        self.seat_residual = float("nan")
        self._neutral_qpos, self._neutral_height = self._solve_stance_pose()

        # Reference gait, loaded only when a stage tracks it.
        self.reference: Optional[Reference] = None
        self.ref_phase = 0.0
        self.tracking_error = 0.0
        self._tracked_qadr = [self.qadr[n] for n in TRACKED_JOINTS]
        if self.stage_spec.track_reference:
            self.reference = Reference()


        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None

        self.rng = np.random.default_rng(seed)
        self._start_tx = 0.0
        self.terminal_bonus = 0.0
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

    # -- model construction ------------------------------------------------

    @staticmethod
    def _build_model(path):
        """Compile a planar, two-ball-contact version of the model.

        Done through MjSpec rather than by editing a copy of MyoSuite's XML,
        so there is no second model file to keep in step with the package and
        no mesh paths to rewrite.
        """
        spec = mujoco.MjSpec.from_file(str(path))

        root = spec.joints[0]
        if root.type != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(
                "expected %s to start with a free root joint, found %r"
                % (Path(path).name, str(root.type))
            )
        root_body = root.parent
        spec.delete(root)
        # The keyframe's qpos was sized for the free root and no longer fits.
        for key in list(spec.keys):
            spec.delete(key)
        for name, jtype, axis in ROOT_SPEC:
            joint = root_body.add_joint()
            joint.name, joint.type, joint.axis = name, jtype, list(axis)

        present = {j.name for j in spec.joints}
        missing = [n for n in OUT_OF_PLANE_JOINTS if n not in present]
        if missing:
            raise RuntimeError(
                "%s is missing out-of-plane joints this env pins: %s"
                % (Path(path).name, ", ".join(missing))
            )
        for name in OUT_OF_PLANE_JOINTS:
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.name1 = name
            eq.name2 = ""
            eq.data = [0.0] * len(eq.data)

        by_geom = {g.name: g for g in spec.geoms}
        absent = [n for n in MYO_FOOT_GEOMS if n not in by_geom]
        if absent:
            raise RuntimeError(
                "%s does not have the MyoSuite foot geoms this env replaces: %s"
                % (Path(path).name, ", ".join(absent))
            )
        for name in MYO_FOOT_GEOMS:
            geom = by_geom[name]
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 1  # still drawn, just not collidable

        by_body = {b.name: b for b in spec.bodies}
        for name, (parent, pos) in BALL_CONTACTS.items():
            if parent not in by_body:
                raise RuntimeError(
                    "body %r not found for contact ball %r" % (parent, name)
                )
            ball = by_body[parent].add_geom()
            ball.name = name
            ball.type = mujoco.mjtGeom.mjGEOM_SPHERE
            ball.size = [BALL_RADIUS, 0.0, 0.0]
            ball.pos = list(pos)
            ball.contype = 1
            ball.conaffinity = 1
            # Group 4 is MyoSuite's collision group, so one toggle shows them.
            ball.group = 4
            # Massless, like the .hfd's contact spheres. At MuJoCo's default
            # density these four added 0.3 kg and shifted the feet's inertia.
            ball.density = 0.0
            ball.rgba = [0.9, 0.5, 0.2, 0.9]
        return spec.compile()

    def _validate_model(self) -> None:
        m = self.model
        n_muscle = int((m.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE).sum())
        if n_muscle == 0:
            raise RuntimeError(
                "%s has no muscle actuators -- this environment is for "
                "muscle-actuated models." % self.model_path.name
            )
        if m.na != n_muscle:
            warnings.warn(
                "expected one activation state per muscle, got na=%d for %d muscles"
                % (m.na, n_muscle),
                RuntimeWarning,
                stacklevel=3,
            )
        self.n_muscle = n_muscle

    def _resolve_indices(self) -> None:
        m = self.model

        def joint_id(name):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError("model is missing joint %r" % name)
            return jid

        names = ROOT_JOINTS + INDEPENDENT_JOINTS
        self.qadr = {n: int(m.jnt_qposadr[joint_id(n)]) for n in names}
        self.dadr = {n: int(m.jnt_dofadr[joint_id(n)]) for n in names}
        self.joint_qpos_adr = [self.qadr[n] for n in INDEPENDENT_JOINTS]
        self.joint_dof_adr = [self.dadr[n] for n in INDEPENDENT_JOINTS]
        self.root_dof_adr = [self.dadr[n] for n in ROOT_JOINTS]

        def body_id(name):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError("model is missing body %r" % name)
            return bid

        self.pelvis_id = body_id(PELVIS_BODY)
        self.torso_id = body_id(TORSO_BODY)
        self.head_id = body_id(HEAD_BODY)
        self.foot_ids = [body_id(n) for n in FOOT_BODIES]

        # Geom ids per load group, ordered heel_r, toe_r, heel_l, toe_l.
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

    # -- measurement -------------------------------------------------------

    @property
    def pelvis_height(self) -> float:
        return float(self.data.xipos[self.pelvis_id][2])

    def trunk_axis(self) -> np.ndarray:
        """Unit vector from the pelvis COM to the torso COM.

        Derived from positions rather than a body frame's axes: this model's
        world is z-up but its body frames are locally y-up, a leftover of the
        OpenSim conversion, so reading a fixed local axis as "up" gives
        nonsense. The torso is used rather than the head because the head
        geometry sits 87 mm off-centre.
        """
        v = np.asarray(self.data.xipos[self.torso_id]) - np.asarray(
            self.data.xipos[self.pelvis_id]
        )
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else WORLD_UP.copy()

    def trunk_tilt(self) -> float:
        """Angle between the trunk axis and vertical, in radians."""
        return float(np.arccos(np.clip(float(self.trunk_axis() @ WORLD_UP), -1.0, 1.0)))

    def com_velocity(self) -> np.ndarray:
        mujoco.mj_subtreeVel(self.model, self.data)
        return np.asarray(self.data.subtree_linvel[0]).copy()

    def forward_velocity(self) -> float:
        """Forward COM speed. In a planar model this is simply +x."""
        return float(self.com_velocity()[0])

    @property
    def travel(self) -> float:
        """Forward distance covered since reset, metres."""
        return float(self.data.qpos[self.qadr["pelvis_tx"]] - self._start_tx)

    def forward_bonus(self) -> float:
        """The terminal forward-progress payment for the episode so far.

        Distance **times** survival. Distance alone is not enough: a trained
        policy learns to dive -- accelerate hard, bank the distance, fall. One
        measured at iteration 180 reached 1.67 m/s, eleven times the
        reference's speed, covered 0.57 m in 0.74 s and collected 425 of 700,
        which was 98.7% of its return. Scaling by the fraction of the episode
        survived drops that same dive to about 31.

        Gating strictly on truncation -- pay only if the full episode is
        survived -- also kills the dive, but pays nothing at all until the
        policy can already last 1000 steps, which removes 70% of the reward
        exactly when it is needed. The product keeps a gradient for partial
        progress while making survival a multiplier rather than a bonus.
        """
        spec = self.stage_spec
        if spec.forward_bonus <= 0.0:
            return 0.0
        progress = smoothstep(self.travel, 0.0, spec.forward_target_distance)
        survived = 1.0
        if spec.forward_requires_survival:
            survived = min(1.0, self.steps / max(1, spec.episode_steps))
        return spec.forward_bonus * spec.episode_steps * progress * survived

    def contact_loads(self) -> np.ndarray:
        """Vertical load on [heel_r, toe_r, heel_l, toe_l], as a fraction of BW.

        Heel and toe are kept apart because the difference between them is
        gait phase -- heel strike loads the rear ball, toe-off the forward one
        -- and a single per-foot total cannot tell those apart.
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

    def _geom_lowest_z(self, name: str) -> float:
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

    def _support_centroid(self) -> np.ndarray:
        """Horizontal centre of the base of support.

        The mean of the four contact balls, not the midpoint of the two calcn
        bodies: the calcn COM sits forward of the foot's real support area, so
        balancing over it leaves the model pitching onto its toes.
        """
        pts = [
            np.asarray(self.data.geom_xpos[g])
            for group in self._load_groups
            for g in group
        ]
        return np.mean(pts, axis=0)

    def foot_stagger(self) -> float:
        """Fore-aft offset between the two feet. Zero is a parallel stance."""
        delta = np.asarray(self.data.xipos[self.foot_ids[0]]) - np.asarray(
            self.data.xipos[self.foot_ids[1]]
        )
        return float(delta @ FORWARD)

    # -- the standing stance ----------------------------------------------

    _STANCE_VARS = ("hip_flexion", "knee_angle", "ankle_angle")

    def _solve_stance_pose(self):
        """Solve for a plantigrade, balanced, upright standing pose.

        The model ships with a mid-stride keyframe rather than a stance, and
        that keyframe is dropped when the free root is replaced, so the pose
        has to be constructed. It is solved **symmetrically** -- one set of leg
        angles applied to both legs -- because the model is exactly
        left/right symmetric, which a test pins.

        Residuals, driven to zero by damped Gauss-Newton:

          * the heel and toe of one foot at z = 0 (plantigrade; symmetry gives
            the other foot for free)
          * the COM horizontally over the base of support (balanced)
          * the trunk axis vertical (upright)

        Unknowns: hip, knee and ankle angle, pelvis height, pelvis tilt and
        trunk flexion -- six against four, solved least-norm.
        """
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)

        def residual(x):
            for side in ("r", "l"):
                for name, value in zip(self._STANCE_VARS, x[:3]):
                    d.qpos[self.qadr["%s_%s" % (name, side)]] = float(value)
            d.qpos[self.qadr["pelvis_ty"]] = x[3]
            d.qpos[self.qadr["pelvis_tilt"]] = x[4]
            d.qpos[self.qadr["flex_extension"]] = x[5]
            mujoco.mj_forward(m, d)
            com = np.asarray(d.subtree_com[0])
            base = self._support_centroid()
            return np.array([
                self._heel_z("calcn_r"),
                self._toe_z("calcn_r"),
                com[0] - base[0],
                float(self.trunk_axis() @ FORWARD),
            ])

        x = np.array([0.0, 0.10, 0.0, 0.0, 0.0, 0.0])
        for _ in range(300):
            r = residual(x)
            if np.abs(r).max() < 1e-10:
                break
            jac = np.zeros((r.size, x.size))
            eps = 1e-6
            for k in range(x.size):
                xp = x.copy()
                xp[k] += eps
                jac[:, k] = (residual(xp) - r) / eps
            step = -np.linalg.solve(jac.T @ jac + 1e-10 * np.eye(x.size), jac.T @ r)
            norm = np.linalg.norm(step)
            if norm > 0.05:
                step *= 0.05 / norm
            x = x + step

        r = residual(x)
        worst = float(np.abs(r).max())
        if worst > 1e-5:
            raise RuntimeError(
                "could not solve a plantigrade standing stance for %s "
                "(worst residual %.2e)." % (self.model_path.name, worst)
            )
        self.stance_residual = worst
        return d.qpos.copy(), float(d.xipos[self.pelvis_id][2])

    def set_pelvis_height(self, height: float) -> None:
        """Place the pelvis COM at an absolute height, exactly.

        `pelvis_ty` is a slide along z, so `pelvis_height = qpos[pelvis_ty] + c`
        where c depends on the rest of the pose -- including `pelvis_tilt`,
        which swings the pelvis COM about the root. So c is measured against
        the pose as it stands now rather than cached from the stance, which
        put the reference frames 0.1 m too high.
        """
        mujoco.mj_forward(self.model, self.data)
        offset = self.pelvis_height - self.data.qpos[self.qadr["pelvis_ty"]]
        self.data.qpos[self.qadr["pelvis_ty"]] = float(height) - offset
        mujoco.mj_forward(self.model, self.data)

    def apply_reference_pose(self, phase: float) -> None:
        """Pose the model at a phase of the reference gait."""
        joints, height = self.reference.pose_at(phase)
        for adr, value in zip(self._tracked_qadr, joints):
            self.data.qpos[adr] = float(value)
        self.set_pelvis_height(height)

    def reference_error(self) -> float:
        """RMS deviation of the tracked joints from the reference, in radians."""
        if self.reference is None:
            return 0.0
        target, _ = self.reference.pose_at(self.ref_phase)
        actual = self.data.qpos[self._tracked_qadr]
        return float(np.sqrt(np.mean((actual - target) ** 2)))

    def _seat_feet(self, iterations: int = 60) -> None:
        """Put both heels and both toes back on the floor after randomisation.

        Reset perturbs the independent joints, which lifts the feet two ways:
        tilting a foot about its ankle raises the heel (0.02 rad moves a 0.2 m
        foot by 4 mm), and perturbing a hip or knee changes that leg's length
        and lifts the whole foot. Each leg is projected back onto "heel and toe
        at z = 0" through its own hip, knee and ankle -- three unknowns against
        two constraints, solved least-norm so the seating disturbs the
        randomised pose as little as it can.
        """
        m, d = self.model, self.data
        worst = 0.0
        for foot, joints in LEG_JOINTS.items():
            adr = [self.qadr[n] for n in joints]
            lim = [
                m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in joints
            ]
            lo_b = np.array([l[0] for l in lim])
            hi_b = np.array([l[1] for l in lim])

            def residual(x):
                for a, v, (lo, hi) in zip(adr, x, lim):
                    d.qpos[a] = float(np.clip(v, lo, hi))
                mujoco.mj_forward(m, d)
                return np.array([self._heel_z(foot), self._toe_z(foot)])

            x = np.array([d.qpos[a] for a in adr], dtype=float)
            for _ in range(iterations):
                r = residual(x)
                if np.abs(r).max() < 1e-9:
                    break
                jac = np.zeros((2, x.size))
                eps = 1e-6
                for k in range(x.size):
                    xp = x.copy()
                    xp[k] += eps
                    jac[:, k] = (residual(xp) - r) / eps
                step = -np.linalg.solve(
                    jac.T @ jac + 1e-10 * np.eye(x.size), jac.T @ r
                )
                norm = np.linalg.norm(step)
                if norm > 0.05:
                    step *= 0.05 / norm
                x = np.clip(x + step, lo_b, hi_b)
            worst = max(worst, float(np.abs(residual(x)).max()))
        self.seat_residual = worst

    # -- observation -------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        d, m = self.data, self.model
        tilt = float(d.qpos[self.qadr["pelvis_tilt"]])
        com = self.com_velocity()
        parts = [
            [self.pelvis_height],
            [np.sin(tilt), np.cos(tilt)],
            d.qpos[self.joint_qpos_adr],
            d.qvel[self.root_dof_adr],
            d.qvel[self.joint_dof_adr],
            [com[0], com[2]],
            self.contact_loads(),
        ]
        if self._include_activation:
            parts.append(d.act[: m.na] if m.na else np.zeros(0))
        if self._include_prev_action:
            parts.append(self.prev_action)
        return np.concatenate(
            [np.asarray(p, dtype=np.float64).ravel() for p in parts]
        ).astype(np.float32)

    def obs_layout(self) -> List[Tuple[str, int]]:
        """Names and widths of the observation blocks, in order.

        `pelvis_tx` is deliberately absent: absolute forward position is not
        part of the task and would let the policy read the episode clock.
        """
        layout = [
            ("pelvis_height", 1),
            ("pelvis_tilt_sin_cos", 2),
            ("joint_q", len(INDEPENDENT_JOINTS)),
            ("root_dq", len(ROOT_JOINTS)),
            ("joint_dq", len(INDEPENDENT_JOINTS)),
            ("com_vel_xz", 2),
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

        if spec.rsi and self.reference is not None:
            # Reference-state initialisation: start at a random phase of the
            # gait. The pose is the reference's, so the feet are *not* seated
            # -- forcing them flat would corrupt a mid-swing frame.
            self.ref_phase = float(self.rng.uniform())
            self.apply_reference_pose(self.ref_phase)
            for name in INDEPENDENT_JOINTS:
                self.data.qpos[self.qadr[name]] += self.rng.normal(
                    0.0, spec.reset_position_std
                )
                self.data.qvel[self.dadr[name]] += self.rng.normal(
                    0.0, spec.reset_velocity_std
                )
            mujoco.mj_forward(self.model, self.data)
            # Never start underground.
            lowest = min(
                min(self._heel_z(foot), self._toe_z(foot)) for foot in FOOT_BODIES
            )
            if lowest < 0.0:
                self.data.qpos[self.qadr["pelvis_ty"]] -= lowest
        else:
            self.ref_phase = 0.0
            for name in INDEPENDENT_JOINTS:
                self.data.qpos[self.qadr[name]] += self.rng.normal(
                    0.0, spec.reset_position_std
                )
                self.data.qvel[self.dadr[name]] += self.rng.normal(
                    0.0, spec.reset_velocity_std
                )
            mujoco.mj_forward(self.model, self.data)
            self._seat_feet()
            lowest = min(
                min(self._heel_z(foot), self._toe_z(foot)) for foot in FOOT_BODIES
            )
            self.data.qpos[self.qadr["pelvis_ty"]] -= lowest

        if spec.initial_forward_velocity > 0.0:
            speed = max(
                0.0,
                float(
                    self.rng.normal(
                        spec.initial_forward_velocity,
                        spec.initial_forward_velocity_std,
                    )
                ),
            )
            self.data.qvel[self.dadr["pelvis_tx"]] = speed

        if self.model.na:
            self.data.act[:] = spec.initial_activation
        self.data.ctrl[:] = spec.initial_activation
        self.prev_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.total_reward = 0.0
        self.term_values = {}
        self.reward_breakdown = {}
        self.tracking_error = self.reference_error()
        self._start_tx = float(self.data.qpos[self.qadr["pelvis_tx"]])
        self.terminal_bonus = 0.0
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
        # Policy space [-1, 1] -> activation [0, 1]: a zero action is half
        # activation, not rest.
        self.data.ctrl[:] = 0.5 * (limited.astype(np.float64) + 1.0)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        if self.reference is not None:
            self.ref_phase = self.reference.advance(self.ref_phase, self.dt)
            self.tracking_error = self.reference_error()
        reward = self._get_reward()
        terminated = self._is_fallen()
        truncated = self.steps >= self.stage_spec.episode_steps
        if terminated:
            reward -= self.stage_spec.reward.fall_penalty
        if terminated or truncated:
            # Forward progress is paid once, here, against distance covered.
            self.terminal_bonus = self.forward_bonus()
            reward += self.terminal_bonus
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
        """Forward speed.

        Tracking stages use a smoothstep that ramps from 0 to the asked-for
        speed and saturates: "reward for moving faster than 0.1 m/s" as a hard
        threshold has no gradient at all from a standing start, which is the
        trap v4's stage D fell into. Non-tracking stages keep the Gaussian
        around a target speed.
        """
        v = self.forward_velocity()
        if self.stage_spec.track_reference:
            return smoothstep(v, 0.0, self.stage_spec.terms.velocity_gate)
        return gaussian(v - self.stage_spec.target_vel,
                        self.stage_spec.terms.velocity_sigma)

    def _term_tracking(self) -> float:
        """How closely the tracked joints follow the reference frame."""
        if self.reference is None:
            return 1.0
        return gaussian(self.tracking_error, self.stage_spec.terms.tracking_sigma)

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
        "effort": _term_effort,
        "tracking": _term_tracking,
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

    def _is_fallen(self) -> bool:
        spec = self.stage_spec
        if (
            self.pelvis_height < spec.min_pelvis_height
            or self.trunk_tilt() > spec.max_trunk_tilt
        ):
            return True
        # Early termination on tracking error matters as much as the reward
        # itself: without it the policy banks alive and velocity return from
        # states that have nothing to do with the reference any more.
        if (
            self.reference is not None
            and spec.max_tracking_error is not None
            and self.tracking_error > spec.max_tracking_error
        ):
            return True
        return False

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

    def describe(self) -> str:
        return (
            "%s | %s | planar, %d muscles, %d dof (%d root + %d joints), "
            "%.1f kg | 2 balls/foot | obs %d, act %d | dt %.3f s%s"
            % (
                self.curriculum_stage,
                self.model_path.name,
                self.n_muscle,
                len(ROOT_JOINTS) + len(INDEPENDENT_JOINTS),
                len(ROOT_JOINTS),
                len(INDEPENDENT_JOINTS),
                self.body_weight / GRAVITY,
                self.observation_space.shape[0],
                self.n_act,
                self.dt,
                "" if self.reference is None else "\n  reference: " + self.reference.describe(),
            )
        )
