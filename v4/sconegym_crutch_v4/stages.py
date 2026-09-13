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


NEUTRAL = "neutral"
INIT_FRAME = "init"
POSTURE_REFERENCES = (NEUTRAL, INIT_FRAME)


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

    # "init"    -> posture and height are measured against the frame the
    #              episode started from ("hold the pose you were dropped in")
    # "neutral" -> measured against the model's neutral standing pose
    #              ("recover to standing from wherever you start")
    posture_reference: str = INIT_FRAME

    # Keep the model at the origin rather than inheriting the reference's x.
    zero_travel: bool = True

    def __post_init__(self):
        if not 0.0 <= self.velocity_scale <= 1.0:
            raise ValueError("velocity_scale must lie in [0, 1]")
        if self.posture_reference not in POSTURE_REFERENCES:
            raise ValueError(
                "posture_reference must be one of %r, got %r"
                % (POSTURE_REFERENCES, self.posture_reference)
            )
        lo, hi = self.phase_range
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("phase_range must satisfy 0 <= lo < hi <= 1")


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
    # Same RSI setup as stage A, deliberately. Stage A learns to hold the
    # reference's forward-leaning gait poses (tilt -0.34 to -0.55 rad); a stage B
    # that reset upright and scored posture against an upright ideal would change
    # the initial-state distribution, the posture reference and the reward all at
    # once. Keeping RSI identical leaves crutch load as the single new axis.
    rsi=RSIConfig(
        trajectory="models/reference/gaitTracking_solution_raw.sto",
        velocity_scale=0.0,
        posture_reference=INIT_FRAME,
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
    # Same RSI as A and B, so the initial-state distribution and posture
    # reference carry across the transfer and velocity, backward and
    # displacement are the only new axes. Stage B's policy is tuned for the
    # reference's forward lean; resetting to the upright neutral pose here and
    # scoring posture against upright would have discarded most of it.
    rsi=RSIConfig(
        trajectory="models/reference/gaitTracking_solution_raw.sto",
        velocity_scale=0.0,
        posture_reference=INIT_FRAME,
    ),
)

# ---------------------------------------------------------------------------
# Stage D: crutch placement and trunk-over-feet correction.
# ---------------------------------------------------------------------------
STAGE_D = StageSpec(
    name="D-posture-fix",
    reward=_spec(
        alive=0.15,
        weights={
            "height": 0.20,
            "posture": 0.45,
            "crutch": 0.15,
            "crutch_forward": 0.20,
            "velocity": 0.25,
            "backward": 0.45,
            "displacement": 0.05,
            # pelvis_forward is deliberately absent. It is clip(travel/cap, 0, 1),
            # which measured a flat 0.0 for an entire episode because the model
            # drifts backward (-0.075 m) under zero torque. It was contributing a
            # permanent multiplicative tax with no gradient, and it duplicates
            # what velocity already rewards. Re-add with w_pelvis_forward if a
            # trained policy actually travels forward.
            "pelvis_lag": 0.20,
        },
        fall_penalty=5.0,
    ),
    terms=TermParams(
        pelvis_tilt_sigma=0.15,
        lumbar_sigma=0.18,
        hip_knee_sigma=0.35,
        height_drop_sigma=0.18,
        cane_target_load_fraction=0.08,
        # Widened from 0.15 to match stage C. With the tighter value the 8%
        # target scored at most 0.22 across a whole episode, against 0.97 in
        # stage C -- unloading the crutch is the goal, but it has to be
        # reachable from where stage C leaves the policy.
        cane_load_sigma_fraction=0.25,
        velocity_sigma_fraction=0.7,
        displacement_cap=0.5,
    ),
    target_vel=0.03,
    # Stage C resets with a small forward push and reached velocity=0.99; stage D
    # inherited C's 0.03 m/s target but reset from a standstill, so velocity
    # never exceeded 0.043. Match C so the target is reachable at step 1.
    initial_forward_velocity=0.02,
    initial_forward_velocity_std=0.01,
    # v3's stage D config declared init_load twice (0.5 then 0.4); the second
    # silently won. Resolved here to 0.5 to match stages A-C. Override in YAML
    # if 0.4 was in fact intended.
    init_load=0.5,
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
