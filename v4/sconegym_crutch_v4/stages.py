"""Curriculum stages as data.

In v3 each stage was a ~550-line copy of the previous stage's class, which is
how the velocity term came to use a linear falloff in stage A and a Gaussian in
stages C and D without anyone intending it. Here there is exactly one
environment class and the stages are values.

Adding a stage means adding an entry to STAGES, not a file.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .rewards import GEOMETRIC, RewardSpec

# Terms that need the crutch contact force API to be working.
CRUTCH_FORCE_TERMS = ("crutch",)
# Terms that need crutch body positions.
CRUTCH_POSE_TERMS = ("crutch_forward",)

# Mean forward speed of models/reference/gaitTracking_solution_raw.sto in m/s,
# from its pelvis_tx speed column. This is the gait the curriculum is trying to
# reproduce. Stage C's 0.03 is a deliberate stepping stone; stage D asks for the
# real thing. Re-measure if the reference changes:
#   python -c "import sys; sys.path.insert(0,'.'); from sconegym_crutch_v4.trajectory import load_sto; t=load_sto('../models/reference/gaitTracking_solution_raw.sto',['pelvis_tx']); print(t.dq[:,0].mean())"
REFERENCE_SPEED = 0.142

# The four sub-movements of reciprocal crutch gait, as windows into the
# reference expressed as fractions of the record. Detected by
# scripts/find_keyframes.py, which finds peaks in arm_flex_r/l (the crutches are
# welded to the forearms, so arm flexion is crutch position) and hip_flexion_r/l:
#
#     0.53 s  frame  25  crutch_r     cycle period 3.41 s (160 frames)
#     1.17 s  frame  55  leg_l        right crutch advances with the LEFT leg
#     2.18 s  frame 102  crutch_l     left crutch advances with the RIGHT leg
#     3.03 s  frame 142  leg_r
#     3.82 s  frame 179  crutch_r  <- second cycle
#     4.61 s  frame 216  leg_l
#     5.67 s  frame 266  crutch_l
#
# leg_r recurs at the very end of the record, past the last detectable peak,
# which is why it has one window and the others have two. Half-width 0.15 s.
#
# Re-derive with: python scripts/find_keyframes.py
GAIT_KEYFRAMES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "crutch_r": ((0.0599, 0.1068), (0.5732, 0.6201)),
    "leg_l": ((0.1599, 0.2068), (0.6966, 0.7434)),
    "crutch_l": ((0.3166, 0.3634), (0.8632, 0.9101)),
    "leg_r": ((0.4499, 0.4968),),
}

KEYFRAME_WINDOWS: Tuple[Tuple[float, float], ...] = tuple(
    sorted(w for windows in GAIT_KEYFRAMES.values() for w in windows)
)

# The same windows grouped by sub-movement, so each of the four gets an equal
# share of episodes. Sampling KEYFRAME_WINDOWS uniformly would not: leg_r has
# one window against two for the others, because its second occurrence falls
# past the end of the record, so it would be trained half as often.
KEYFRAME_GROUPS: Tuple[Tuple[Tuple[float, float], ...], ...] = tuple(
    GAIT_KEYFRAMES[name] for name in ("crutch_r", "leg_l", "crutch_l", "leg_r")
)

# Every keyframe's midpoint, in record order, as (fraction, sub-movement). Used
# to answer "which pose comes next" when a stage targets the following keyframe
# rather than the one it started at. Record order already interleaves the two
# cycles correctly: crutch_r, leg_l, crutch_l, leg_r, crutch_r, leg_l, crutch_l.
KEYFRAME_CENTERS: Tuple[Tuple[float, str], ...] = tuple(
    sorted(
        ((lo + hi) / 2.0, name)
        for name, windows in GAIT_KEYFRAMES.items()
        for lo, hi in windows
    )
)


# The canonical order of the gait cycle. The chain follows THIS, not the order
# the windows happen to appear in the record.
#
# That distinction matters: leg_r has a single window because its second
# occurrence falls past the end of the file. Walking the record in order would
# give crutch_r, leg_l, crutch_l, leg_r, crutch_r, leg_l, crutch_l and then wrap
# straight back to crutch_r, silently skipping the right-leg step every other
# cycle -- a limp, not a gait.
CYCLE_ORDER: Tuple[str, ...] = ("crutch_r", "leg_l", "crutch_l", "leg_r")

# Keyframes with their windows attached, in record order, so a fraction can be
# mapped to the window it falls inside.
_KEYFRAME_SEQUENCE: Tuple[Tuple[float, float, float, str], ...] = tuple(
    sorted(
        ((lo + hi) / 2.0, lo, hi, name)
        for name, windows in GAIT_KEYFRAMES.items()
        for lo, hi in windows
    )
)


def _sub_movement_at(fraction: float) -> str:
    """Which sub-movement `fraction` belongs to, or the most recent one behind it."""
    for centre, lo, hi, name in _KEYFRAME_SEQUENCE:
        if lo - 1e-9 <= fraction <= hi + 1e-9:
            return name
    behind = [entry for entry in _KEYFRAME_SEQUENCE if entry[0] <= fraction]
    return behind[-1][3] if behind else _KEYFRAME_SEQUENCE[-1][3]


def next_keyframe(fraction: float) -> Tuple[float, str]:
    """The keyframe that follows `fraction` in the gait cycle.

    The sub-movement is decided by CYCLE_ORDER, so the sequence is always
    crutch_r, leg_l, crutch_l, leg_r regardless of how many windows each has in
    the record. The frame returned is that sub-movement's next occurrence after
    `fraction`, wrapping to its first when there is none -- so targets stay
    local to where the model is whenever the record allows.
    """
    current = _sub_movement_at(fraction)
    following = CYCLE_ORDER[(CYCLE_ORDER.index(current) + 1) % len(CYCLE_ORDER)]
    centres = sorted((lo + hi) / 2.0 for lo, hi in GAIT_KEYFRAMES[following])
    ahead = [c for c in centres if c > fraction + 1e-9]
    return (ahead[0] if ahead else centres[0]), following


@dataclass(frozen=True)
class TermParams:
    """Shaping hyperparameters. One home for every sigma in the project."""

    # posture / height
    pelvis_tilt_sigma: float = 0.12
    lumbar_sigma: float = 0.12
    hip_knee_sigma: float = 0.35
    height_drop_sigma: float = 0.18

    # crutch load sharing
    cane_target_load_fraction: float = 0.15
    cane_load_sigma_fraction: float = 0.25
    # Smooth contact gate, replacing the v3 "if force < 20: return 0" cliff.
    cane_gate_lo_n: float = 10.0
    cane_gate_hi_n: float = 30.0

    # crutch placement, measured relative to the neutral standing pose
    crutch_forward_margin: float = 0.02
    crutch_forward_sigma: float = 0.05

    # locomotion
    velocity_sigma_fraction: float = 0.7
    backward_sigma: float = 0.05
    displacement_cap: float = 0.5
    displacement_sigma: float = 0.10
    pelvis_lag_sigma: float = 0.05

    # Neutral-pose x offsets from the pelvis body COM, in metres, as measured by
    # scripts/calibrate.py on Rajagopal2015_crutch_2D_ankles_locked_mesh_lumbar:
    #
    #   rearmost foot   pelvis + 0.101
    #   rearmost crutch pelvis + 0.052
    #
    # pelvis_lag and crutch_forward measure deviation from these, not from zero.
    # Measuring against zero made both terms constants: pelvis_lag scored 0.017
    # at rest and crutch_forward a flat 1.0. Re-measure with calibrate.py if the
    # model or its init state changes.
    pelvis_foot_offset_ref: float = 0.101
    crutch_offset_ref: float = 0.052


def _check_window(window) -> None:
    lo, hi = window
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(
            "each phase window must satisfy 0 <= lo < hi <= 1, got %r" % (window,)
        )


NEUTRAL = "neutral"
INIT_FRAME = "init"
NEXT_KEYFRAME = "next_keyframe"
CHAIN = "chain"
POSTURE_REFERENCES = (NEUTRAL, INIT_FRAME, NEXT_KEYFRAME, CHAIN)


@dataclass(frozen=True)
class RSIConfig:
    """Reference-state initialisation: reset to a random phase of a trajectory.

    The DeepMimic trick. Resetting only to the neutral pose teaches balance from
    one state; resetting to random phases of a gait cycle teaches it across the
    whole cycle and gives a far wider basin of attraction.
    """

    # Path to the reference, relative to the repository root.
    trajectory: str = "models/reference/gaitTracking_solution_raw.sto"

    # Fraction of the reference's joint velocities to apply at reset.
    #
    # 0.0 drops the model into a walking *pose* at rest, which is the balance
    # task. 1.0 hands it the full mid-stride momentum, which is a catch-and-
    # recover task and much harder. This is the natural thing to fade in: start
    # at 0.0, raise it as the policy stops falling.
    velocity_scale: float = 0.0

    # Restrict sampling to a sub-window of the reference, as fractions.
    phase_range: Tuple[float, float] = (0.0, 1.0)

    # Restrict sampling to a set of disjoint windows instead, which is how a
    # keyframe curriculum is expressed: every episode starts at one of the gait
    # events rather than anywhere in the cycle. Takes precedence over
    # phase_range when set.
    phase_windows: Optional[Tuple[Tuple[float, float], ...]] = None

    # Windows grouped by sub-movement. The group is sampled uniformly first, so
    # each sub-movement gets an equal share even when they recur unequally
    # often in the record. Takes precedence over phase_windows.
    phase_window_groups: Optional[Tuple[Tuple[Tuple[float, float], ...], ...]] = None

    # "init"          -> posture and height measured against the frame the
    #                    episode started from ("hold the pose you were dropped
    #                    in")
    # "neutral"       -> measured against the model's neutral standing pose
    #                    ("recover to standing from wherever you start")
    # "next_keyframe" -> measured against the NEXT gait keyframe ("move from
    #                    this pose to the following one"). This is the
    #                    transition task: B holds the poses, C connects them.
    # "chain"         -> like next_keyframe, but the target ADVANCES to the
    #                    following keyframe each time the model arrives, so one
    #                    episode walks the whole cycle instead of one step of
    #                    it. This is the stage that wires the sub-movements into
    #                    continuous gait.
    posture_reference: str = INIT_FRAME

    # Posture value at which a "chain" episode counts the target as reached and
    # advances to the next keyframe. Too high and the model can never advance;
    # too low and it skips ahead without really arriving. 0.60 is roughly what a
    # trained stage C policy reaches on its target pose.
    chain_advance_threshold: float = 0.60

    # Keep the model at the origin rather than inheriting the reference's x.
    zero_travel: bool = True

    def __post_init__(self):
        if not 0.0 <= self.velocity_scale <= 1.0:
            raise ValueError("velocity_scale must lie in [0, 1]")
        if not 0.0 < self.chain_advance_threshold < 1.0:
            raise ValueError("chain_advance_threshold must lie in (0, 1)")
        if self.posture_reference not in POSTURE_REFERENCES:
            raise ValueError(
                "posture_reference must be one of %r, got %r"
                % (POSTURE_REFERENCES, self.posture_reference)
            )
        lo, hi = self.phase_range
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("phase_range must satisfy 0 <= lo < hi <= 1")
        if self.phase_windows is not None:
            if not self.phase_windows:
                raise ValueError("phase_windows must be non-empty when given")
            for window in self.phase_windows:
                _check_window(window)
        if self.phase_window_groups is not None:
            if not self.phase_window_groups:
                raise ValueError("phase_window_groups must be non-empty when given")
            for group in self.phase_window_groups:
                if not group:
                    raise ValueError("each phase window group must be non-empty")
                for window in group:
                    _check_window(window)


@dataclass(frozen=True)
class StageSpec:
    """Everything that distinguishes one curriculum stage from another."""

    name: str
    reward: RewardSpec
    terms: TermParams = field(default_factory=TermParams)
    rsi: Optional[RSIConfig] = None

    target_vel: float = 0.0

    # reset distribution
    initial_forward_velocity: float = 0.0
    initial_forward_velocity_std: float = 0.0
    reset_position_std: float = 0.005
    reset_velocity_std: float = 0.005
    pelvis_height_position_std: float = 0.002
    init_load: float = 0.5

    # termination
    min_com_height: float = 0.55
    min_head_height: float = 0.75
    episode_steps: int = 1000

    # control
    action_rate_limit: float = 0.05

    def __post_init__(self):
        if self.target_vel < 0.0:
            raise ValueError("target_vel must be non-negative")
        if "velocity" in self.reward.active_weights and self.target_vel <= 0.0:
            raise ValueError(
                "stage %r weights the velocity term but target_vel is %r"
                % (self.name, self.target_vel)
            )
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        if not 0.0 < self.action_rate_limit <= 2.0:
            raise ValueError("action_rate_limit must lie in (0, 2]")

    @property
    def velocity_sigma(self) -> float:
        return max(self.target_vel * self.terms.velocity_sigma_fraction, 1e-6)

    @property
    def needs_crutch_force(self) -> bool:
        return any(t in self.reward.active_weights for t in CRUTCH_FORCE_TERMS)

    @property
    def needs_crutch_pose(self) -> bool:
        return any(t in self.reward.active_weights for t in CRUTCH_POSE_TERMS)

    def with_overrides(self, **overrides: Any) -> "StageSpec":
        """Apply flat keyword overrides, as supplied by a tonic env_args block.

        Namespaces, checked in order:
          w_<term>       -> reward weight for that term
          <reward field> -> alive, shaping_scale, fall_penalty, term_floor,
                            composition
          <term field>   -> any TermParams field
          <stage field>  -> any StageSpec field except name/reward/terms

        Unknown keys raise. v3 swallowed configuration mistakes silently; a
        typo in a YAML coefficient should stop the run, not train for 10M steps
        against a default nobody chose.
        """
        if not overrides:
            return self

        reward_fields = {"alive", "shaping_scale", "fall_penalty", "term_floor", "composition"}
        term_fields = {f.name for f in dataclasses.fields(TermParams)}
        stage_fields = {f.name for f in dataclasses.fields(StageSpec)} - {
            "name",
            "reward",
            "terms",
            "rsi",
        }
        rsi_fields = {f.name for f in dataclasses.fields(RSIConfig)}

        weight_updates: Dict[str, float] = {}
        reward_updates: Dict[str, Any] = {}
        term_updates: Dict[str, Any] = {}
        stage_updates: Dict[str, Any] = {}
        rsi_updates: Dict[str, Any] = {}
        unknown = []

        for key, value in overrides.items():
            if key.startswith("w_"):
                weight_updates[key[2:]] = float(value)
            elif key in reward_fields:
                reward_updates[key] = value
            elif key in term_fields:
                term_updates[key] = float(value)
            elif key in stage_fields:
                stage_updates[key] = value
            elif key.startswith("rsi_") and key[4:] in rsi_fields:
                rsi_updates[key[4:]] = value
            else:
                unknown.append(key)

        if unknown:
            raise KeyError(
                "unknown override(s) for stage %r: %s\n"
                "valid prefixes: w_<term>, rsi_<field>; valid keys: %s"
                % (
                    self.name,
                    ", ".join(sorted(unknown)),
                    ", ".join(sorted(reward_fields | term_fields | stage_fields)),
                )
            )

        reward = self.reward
        if weight_updates:
            merged = dict(reward.weights)
            merged.update(weight_updates)
            reward_updates["weights"] = merged
        if reward_updates:
            reward = dataclasses.replace(reward, **reward_updates)

        terms = dataclasses.replace(self.terms, **term_updates) if term_updates else self.terms

        rsi = self.rsi
        if rsi_updates:
            base = rsi if rsi is not None else RSIConfig()
            rsi = dataclasses.replace(base, **rsi_updates)

        return dataclasses.replace(
            self, reward=reward, terms=terms, rsi=rsi, **stage_updates
        )

    def describe(self) -> str:
        w = self.reward.active_weights
        order = sorted(w, key=lambda k: -w[k])
        return "%s | alive=%.2f scale=%.2f floor=%.2f %s | %s" % (
            self.name,
            self.reward.alive,
            self.reward.shaping_scale,
            self.reward.term_floor,
            self.reward.composition,
            ", ".join("%s=%.2f" % (k, w[k]) for k in order),
        )


def _spec(alive: float, weights: Mapping[str, float], **kw) -> RewardSpec:
    """RewardSpec with shaping_scale set so the max step reward is 1.0."""
    return RewardSpec(alive=alive, shaping_scale=1.0 - alive, weights=dict(weights), **kw)


# ---------------------------------------------------------------------------
# Stage A: balance, initialised at random phases of the reference gait.
#
# Episodes start from a uniformly random frame of the Moco tracking solution
# rather than always from the neutral pose, and the task is to hold whatever
# pose the model was dropped into. velocity_scale=0.0 means the pose is handed
# over at rest -- raise it to fade mid-stride momentum in.
#
# posture_reference="init" is essential here, not cosmetic: the reference leans
# 19-31 degrees forward throughout (pelvis_tilt -0.55 to -0.34 rad), so posture
# measured against an upright ideal would score about 0.0004 on every frame.
# ---------------------------------------------------------------------------
STAGE_A = StageSpec(
    name="A-stand",
    reward=_spec(alive=0.20, weights={"height": 0.25, "posture": 0.45}, fall_penalty=5.0),
    terms=TermParams(pelvis_tilt_sigma=0.12, lumbar_sigma=0.12),
    rsi=RSIConfig(
        trajectory="models/reference/gaitTracking_solution_raw.sto",
        velocity_scale=0.0,
        posture_reference=INIT_FRAME,
    ),
)

# ---------------------------------------------------------------------------
# Stage B: standing with crutch load sharing. One new axis.
# ---------------------------------------------------------------------------
STAGE_B = StageSpec(
    name="B-stand-crutch",
    reward=_spec(
        alive=0.20,
        weights={"height": 0.25, "posture": 0.45, "crutch": 0.25},
        fall_penalty=5.0,
    ),
    terms=TermParams(
        pelvis_tilt_sigma=0.12,
        lumbar_sigma=0.12,
        cane_target_load_fraction=0.15,
        cane_load_sigma_fraction=0.15,
    ),
    # Same trajectory and posture reference as stage A, but sampling is now
    # restricted to the four gait keyframes rather than the whole cycle. Stage A
    # learned to hold an arbitrary pose from anywhere in the record; stage B
    # learns to hold the specific poses the gait is built from -- one crutch
    # forward, then the opposite leg forward, alternating sides.
    #
    # velocity_scale stays 0.0, so each pose arrives at rest and the task is
    # static: stand in this configuration without falling, with the crutches
    # taking their share of the load. Connecting the poses into motion is a
    # later stage's job.
    rsi=RSIConfig(
        trajectory="models/reference/gaitTracking_solution_raw.sto",
        velocity_scale=0.0,
        posture_reference=INIT_FRAME,
        phase_window_groups=KEYFRAME_GROUPS,
    ),
)

# ---------------------------------------------------------------------------
# Stage C: tiny forward locomotion with the crutch.
# ---------------------------------------------------------------------------
STAGE_C = StageSpec(
    name="C-tiny-forward",
    reward=_spec(
        alive=0.15,
        weights={
            "height": 0.20,
            "posture": 0.45,
            "crutch": 0.20,
            "velocity": 0.25,
            "backward": 0.15,
            "displacement": 0.05,
        },
        fall_penalty=4.0,
    ),
    terms=TermParams(
        pelvis_tilt_sigma=0.15,
        lumbar_sigma=0.15,
        cane_target_load_fraction=0.15,
        cane_load_sigma_fraction=0.25,
        velocity_sigma_fraction=0.7,
        displacement_cap=0.5,
        # Widened from 0.35 because this is the first stage that has to MOVE.
        # posture measures hip and knee deviation from the reference frame, and
        # a stride changes hip flexion by roughly 0.6 rad. At 0.35 the posture
        # term (weight 0.45) would score about exp(-(0.6/0.35)^2) = 0.05 for any
        # real step, directly opposing the velocity term it shares the reward
        # with. At 0.60 a full stride costs a factor of about 0.37 instead.
        # Revert to 0.35 if stage C walks but the legs stop tracking the
        # reference at all.
        hip_knee_sigma=0.60,
    ),
    target_vel=0.03,
    initial_forward_velocity=0.02,
    initial_forward_velocity_std=0.01,
    reset_position_std=0.01,
    reset_velocity_std=0.01,
    # Same keyframe sampling as stage B, but posture now targets the NEXT
    # keyframe rather than the one the episode starts at. B learned to hold the
    # four poses; C learns the transitions between them:
    #
    #     crutch_r -> leg_l -> crutch_l -> leg_r -> crutch_r ...
    #
    # That is the single new axis. The episode still runs 1000 steps, so the
    # task is reach the next pose and then hold it -- and holding is exactly
    # what stage B trained, so the warm start carries.
    rsi=RSIConfig(
        trajectory="models/reference/gaitTracking_solution_raw.sto",
        velocity_scale=0.0,
        posture_reference=NEXT_KEYFRAME,
        phase_window_groups=KEYFRAME_GROUPS,
    ),
)

# ---------------------------------------------------------------------------
# Stage D: crutch placement and trunk-over-feet correction.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Stage D: chain the sub-movements into continuous gait, at reference speed.
#
# Stage B holds the four keyframe poses, stage C moves from one to the next, and
# stage D repeats that: each time the model arrives at its target the target
# advances to the following keyframe, so a single episode walks
#
#     crutch_r -> leg_l -> crutch_l -> leg_r -> crutch_r -> ...
#
# for as long as it can keep going. Speed is a consequence of chaining quickly
# rather than a separate objective, so velocity sits at a low weight as a
# tiebreaker and pelvis_forward rewards the distance that results.
#
# crutch_forward and pelvis_lag are dropped. They were from the older
# "posture fix" design, neither has ever run in training, and the chained pose
# targets already specify where the crutches and pelvis should be.
# ---------------------------------------------------------------------------
STAGE_D = StageSpec(
    name="D-chain",
    reward=_spec(
        alive=0.15,
        weights={
            # Reaching the current target pose is the task.
            "posture": 0.45,
            # Distance covered, which is what chaining produces.
            "pelvis_forward": 0.30,
            "backward": 0.25,
            "height": 0.20,
            # A tiebreaker toward reference cadence, not a driver: the model
            # moves because it is chasing pose targets, not because of this.
            "velocity": 0.20,
            "crutch": 0.15,
            "displacement": 0.05,
        },
        fall_penalty=5.0,
    ),
    terms=TermParams(
        pelvis_tilt_sigma=0.15,
        lumbar_sigma=0.18,
        # As in stage C: this stage strides, and the standing value of 0.35
        # would make posture score about 0.05 for any real step.
        hip_knee_sigma=0.60,
        height_drop_sigma=0.18,
        cane_target_load_fraction=0.08,
        cane_load_sigma_fraction=0.25,
        velocity_sigma_fraction=0.7,
        # A full episode at reference speed covers 1.42 m, so the stage C cap of
        # 0.5 m would penalise exactly what this stage exists to produce.
        displacement_cap=2.0,
    ),
    target_vel=REFERENCE_SPEED,
    initial_forward_velocity=0.10,
    initial_forward_velocity_std=0.02,
    reset_position_std=0.01,
    reset_velocity_std=0.01,
    # v3's stage D config declared init_load twice (0.5 then 0.4); the second
    # silently won. Resolved here to 0.5 to match stages A-C.
    init_load=0.5,
    rsi=RSIConfig(
        trajectory="models/reference/gaitTracking_solution_raw.sto",
        velocity_scale=0.0,
        posture_reference=CHAIN,
        phase_window_groups=KEYFRAME_GROUPS,
    ),
)


STAGES: Dict[str, StageSpec] = {
    "A": STAGE_A,
    "B": STAGE_B,
    "C": STAGE_C,
    "D": STAGE_D,
}

STAGE_ORDER: Tuple[str, ...] = ("A", "B", "C", "D")


def get_stage(key: str) -> StageSpec:
    try:
        return STAGES[key.upper()]
    except KeyError:
        raise KeyError(
            "unknown stage %r; expected one of %s" % (key, ", ".join(STAGE_ORDER))
        ) from None
