"""The muscle-locomotion curriculum, as data.

Same shape as v4's stages.py: one `StageSpec` per stage, differing only in
coefficients, so there is one environment class rather than a family of
near-identical ones.

The task is different from v4's, and so are the terms. v4 drove a planar
9-torque skeleton with welded crutches; this drives a planar 290-muscle body
with no assistive device. The model is planar like v4's, so lateral drift and
turning are structurally impossible and need no terms. One kind of term is
new:

* **An effort term.** 290 Hill-type muscles are hugely overactuated: many
  activation patterns produce the same motion, and most of them are
  co-contraction that a real person would not use. `effort` selects among them.
  A torque model with 9 actuators did not need this.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Tuple

from .rewards import ADDITIVE, GEOMETRIC, RewardSpec

STAND = "A-stand"
WALK = "B-walk"
WALK_TRACK = "W-walk-track"

STAGE_ORDER: Tuple[str, ...] = ("A", "B", "W")


@dataclass(frozen=True)
class TermParams:
    """Shape parameters for the individual reward terms."""

    # height: pelvis height measured against the settled standing height
    height_drop_sigma: float = 0.12

    # upright: trunk tilt away from vertical, in radians
    upright_sigma: float = 0.35

    # velocity: forward speed error, m/s
    velocity_sigma: float = 0.35

    # effort: mean muscle activation that still scores well. 0.15 is a
    # deliberately loose budget -- tight effort penalties on an overactuated
    # model suppress motion before they suppress co-contraction.
    effort_target: float = 0.15
    effort_sigma: float = 0.25

    # velocity_gate: the speed at which the forward term saturates, m/s. Used
    # by stage W, where velocity is a smoothstep from 0 to this rather than a
    # Gaussian around a target. "Reward for moving faster than 0.1 m/s" as a
    # hard threshold would have no gradient at all from a standing start,
    # which is the trap v4's stage D fell into; smoothstep keeps the gradient
    # and still saturates at the asked-for speed.
    velocity_gate: float = 0.10

    # tracking: RMS joint deviation from the reference frame, in radians,
    # at which the term has fallen to 1/e.
    tracking_sigma: float = 0.35


@dataclass(frozen=True)
class StageSpec:
    key: str
    name: str
    reward: RewardSpec
    terms: TermParams = field(default_factory=TermParams)

    target_vel: float = 0.0
    episode_steps: int = 1000

    # Reference tracking. `track_reference` turns on the tracking term and the
    # reference clock; `rsi` starts each episode at a random phase of the
    # reference instead of the solved standing stance, which is what makes
    # imitation tractable and, incidentally, gives a phase-discovering gate a
    # balanced sample of phases for free.
    track_reference: bool = False
    rsi: bool = False
    rsi_velocity_scale: float = 0.0
    # Terminate when the RMS joint deviation from the reference exceeds this,
    # so the policy never banks return from a desynced state. None disables it.
    max_tracking_error: float = 0.80

    # Forward progress, paid once when the episode ends rather than per step.
    #
    # Per step, "reward for moving faster than 0.1 m/s" is collectable by
    # falling forward: toppling produces v > 0.1 m/s and the term saturates
    # immediately. Paid at the end against *distance covered*, it is not --
    # falling ends the episode after a few centimetres, and smoothstep of that
    # against a metre is essentially zero.
    #
    # `forward_bonus` is a per-step-equivalent weight, so the terminal payment
    # is `forward_bonus * episode_steps * smoothstep(travel, 0, target)`. That
    # keeps the asked-for proportions over a full episode: alive 0.1, tracking
    # 0.2 and forward 0.7 of a maximum 1.0 per step.
    forward_bonus: float = 0.0
    # Distance for full credit, metres. Defaults to velocity_gate * duration,
    # i.e. the distance covered by holding the target speed for the episode.
    forward_target_distance: float = 1.0

    # Stance width needs no parameter any more: hip adduction is pinned to
    # zero by the planar constraint, so the feet sit at the model's own hip
    # spacing and the legs cannot cross.

    # Termination. The model stands with its pelvis near 0.98 m.
    min_pelvis_height: float = 0.65
    max_trunk_tilt: float = 1.20  # radians from vertical

    # Reset randomisation, applied to the independent joints only. The
    # constrained ones (knee coupling, lumbar distribution) are left to their
    # equality constraints -- perturbing them directly would fight the solver.
    reset_position_std: float = 0.02
    reset_velocity_std: float = 0.02
    initial_forward_velocity: float = 0.0
    initial_forward_velocity_std: float = 0.0

    # Muscle activations start here. Non-zero because a fully deactivated
    # muscle model collapses before the policy's first action takes effect --
    # activation dynamics have a rise time of tens of milliseconds.
    initial_activation: float = 0.05

    # Action smoothing, as in v4. prev_action is in the observation, so the
    # MDP stays Markov under it.
    action_rate_limit: float = 0.20

    def with_overrides(self, **overrides) -> "StageSpec":
        """Apply constructor overrides, raising on any name we do not know.

        Same contract as v4: an unknown name stops the run rather than
        silently training against a default nobody chose.
        """
        if not overrides:
            return self

        weights = dict(self.reward.weights)
        reward_fields = {}
        term_fields = {}
        stage_fields = {}

        for key, value in overrides.items():
            if key.startswith("w_"):
                term = key[2:]
                if term not in weights:
                    raise KeyError(
                        "stage %s has no reward term %r (weights: %s)"
                        % (self.key, term, ", ".join(sorted(weights)))
                    )
                weights[term] = float(value)
            elif key in ("alive", "shaping_scale", "fall_penalty", "term_floor",
                         "composition"):
                reward_fields[key] = value
            elif key in TermParams.__dataclass_fields__:
                term_fields[key] = value
            elif key in StageSpec.__dataclass_fields__:
                stage_fields[key] = value
            else:
                raise KeyError(
                    "unknown override %r for stage %s" % (key, self.key)
                )

        reward = replace(self.reward, weights=weights, **reward_fields)
        terms = replace(self.terms, **term_fields) if term_fields else self.terms
        return replace(self, reward=reward, terms=terms, **stage_fields)

    def describe(self) -> str:
        w = ", ".join(
            "%s=%.2f" % (k, v) for k, v in sorted(self.reward.active_weights.items())
        )
        return "%s | alive=%.2f scale=%.2f %s | target_vel=%.2f | %s" % (
            self.name,
            self.reward.alive,
            self.reward.shaping_scale,
            self.reward.composition,
            self.target_vel,
            w,
        )


# Stage A: hold a standing posture. No velocity term at all -- not a velocity
# term with a target of zero, which would reward freezing and make the
# transition to B a discrete jump in what the reward measures.
STAGE_A = StageSpec(
    key="A",
    name=STAND,
    target_vel=0.0,
    episode_steps=1000,
    reward=RewardSpec(
        alive=0.20,
        shaping_scale=0.80,
        composition=GEOMETRIC,
        weights={
            "height": 0.40,
            "upright": 0.40,
            "effort": 0.20,
        },
    ),
)

# Stage B: walk forward. `lateral` and `heading` keep it going straight, which
# a 3-D model will not do on its own.
STAGE_B = StageSpec(
    key="B",
    name=WALK,
    target_vel=1.20,
    episode_steps=1000,
    initial_forward_velocity=0.30,
    initial_forward_velocity_std=0.10,
    # A wide velocity sigma, for the reason v4's stage D had to learn twice:
    # a Gaussian velocity term whose sigma is small against the distance from
    # standstill to target has no gradient where the policy actually starts.
    # At the shipped 0.35 a motionless model scored exp(-(1.2/0.35)^2) = 7e-6,
    # which the term floor then flattened completely. At 0.80 standstill
    # scores 0.11 and every 0.1 m/s gained is worth something.
    terms=TermParams(velocity_sigma=0.80),
    reward=RewardSpec(
        alive=0.20,
        shaping_scale=0.80,
        composition=GEOMETRIC,
        weights={
            # velocity carries the most weight for the reason v4's stage D
            # documents: with several terms each exponent shrinks, and the
            # terms that reward *not moving* (height, upright, effort) are all
            # satisfied by standing still. Walking has to be worth more.
            # Dropping `lateral` and `heading` with the move to a planar model
            # helps here too -- both were satisfied by standing still, and
            # removing them raises every remaining term's exponent.
            "velocity": 0.50,
            "height": 0.25,
            "upright": 0.25,
            "effort": 0.15,
        },
    ),
)

# Stage W: the reward the project actually asked for.
#
#   big penalty for falling      -> early termination plus a small explicit
#                                   penalty; see below
#   0.1  for surviving 10 s      -> 0.10/step alive bonus over 1000 steps
#   0.7  for moving forward      -> paid ONCE at the end, against distance
#                                   covered: 0.70 * 1000 * smoothstep(travel,
#                                   0, 1.0 m)
#   0.2  for following the gait  -> reference tracking, weight 0.20/step
#
# Composed **additively**, not by v4's geometric mean. Geometrically, a
# tracking term near zero early in training would gate the velocity term to
# zero as well and neither would learn. Additively the weights already encode
# the priority: standing still scores 0.10, moving scores 0.80, moving on the
# reference scores 1.00.
#
# Over a full episode the maxima are alive 100, tracking 200 and forward 700,
# summing to 1000 -- the same proportions as a 1.0/step reward, with the
# forward share moved to the end.
#
# Paying forward progress terminally is what stops it being farmed by falling.
# Per step, any topple produces v > 0.1 m/s and saturates the term; against
# distance at the end, a fall at step 32 has covered ~0.05 m, and
# smoothstep(0.05, 0, 1.0) is 0.007.
#
# fall_penalty stays small on purpose. With early termination, falling already
# forfeits the rest of the episode -- up to ~900 steps at ~1.0 -- and that is
# the real penalty. A large explicit one on top makes standing still dominate
# any policy that risks moving. RewardSpec.termination_report() checks this.
STAGE_W = StageSpec(
    key="W",
    name=WALK_TRACK,
    target_vel=0.142,           # the reference's own mean speed
    episode_steps=1000,         # 10 s at dt = 0.01
    track_reference=True,
    rsi=True,
    initial_forward_velocity=0.10,
    initial_forward_velocity_std=0.03,
    forward_bonus=0.70,
    forward_target_distance=1.00,   # 0.10 m/s held for the full 10 s
    reward=RewardSpec(
        alive=0.10,
        shaping_scale=0.20,
        composition=ADDITIVE,
        fall_penalty=5.0,
        weights={"tracking": 1.00},
    ),
)

STAGES: Dict[str, StageSpec] = {"A": STAGE_A, "B": STAGE_B, "W": STAGE_W}


def get_stage(key: str) -> StageSpec:
    k = str(key).upper()
    if k not in STAGES:
        raise KeyError(
            "unknown stage %r; expected one of %s" % (key, ", ".join(STAGE_ORDER))
        )
    return STAGES[k]


def describe_stages() -> str:
    return "\n".join(STAGES[k].describe() for k in STAGE_ORDER)
