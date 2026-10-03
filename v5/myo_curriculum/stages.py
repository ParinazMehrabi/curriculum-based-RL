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
from typing import Dict, Mapping, Optional, Tuple

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

    # A captured initial state, by filename under models/init/ or by
    # path. Supplies pose, joint velocities and muscle activations, and
    # takes precedence over rsi and over initial_forward_velocity.
    #
    # Needed because the model is exactly left/right symmetric and so is
    # a deterministic policy's response to a symmetric observation. From
    # a standing start both legs hold identical angles, velocities and
    # activations, the two sides of the network see mirror-identical
    # input, and nothing in the reward breaks the tie: the result hops
    # rather than steps. Reset noise breaks it only by accident, a
    # little, and differently every episode -- and stage W sets that
    # noise to zero, which made it exact.
    init_state: Optional[str] = None
    # Terminate when the RMS joint deviation from the reference exceeds this,
    # so the policy never banks return from a desynced state. None disables it.
    #
    # 1.5 rather than the 0.80 it started at. Measured on a low-activation
    # policy, 0.80 ended 13 of 20 episodes at a mean length of 32 steps; 1.5
    # never fires and every episode ends on trunk tilt at 39. Above 1.5 it
    # stops binding altogether, so this is the point where it becomes a
    # backstop against a policy that drifts off the reference while staying
    # upright, rather than the thing that ends most episodes.
    max_tracking_error: float = 1.50

    # Forward progress, paid once when the episode ends rather than per step,
    # and **open-ended**: further is always worth more, with no distance at
    # which the term stops paying.
    #
    # Per step, "reward for moving faster than 0.1 m/s" is collectable by
    # falling forward: toppling produces v > 0.1 m/s and saturates the term
    # immediately. Paid at the end against *distance covered*, it is not.
    #
    # `forward_bonus` is a per-step-equivalent weight, so the payment is
    #
    #     forward_bonus * episode_steps * travel / forward_reference_distance
    #
    # linear in distance and unbounded above. Over an episode covering
    # `forward_reference_distance` that is alive 0.1, tracking 0.2 and forward
    # 0.7 of a maximum 1.0 per step -- the asked-for proportions -- and
    # covering twice the distance pays twice as much.
    #
    # It is signed, so walking backwards costs what walking forwards earns.
    # An open-ended distance reward has to be capped in *height* instead, or
    # the cheapest way to cover ground is to leave it: see fly_penalty.
    forward_bonus: float = 0.0
    # The distance worth one full episode of `forward_bonus`, metres. A scale,
    # not a ceiling: 1.0 m is velocity_gate held for the whole episode, so
    # matching the asked-for 0.1 m/s pays the full 0.70/step-equivalent and
    # beating it pays proportionally more.
    forward_reference_distance: float = 1.0
    # Multiply the bonus by the fraction of the episode survived, so distance
    # only pays if it is held. Without it the policy dives: accelerate, bank
    # the distance, fall. See MyoLocomotionEnv.forward_bonus.
    forward_requires_survival: bool = True
    # Retreating is charged at this multiple of the rate advancing earns. 1.0
    # is symmetric; above it, ground given up costs more than the same ground
    # gained is worth, which is what makes standing still preferable to
    # toppling backwards rather than merely equal to it.
    backward_multiplier: float = 1.0

    # Flying. With an open-ended distance reward the cheapest way to cover
    # ground is to stop touching it -- launch, travel ballistically, land. A
    # penalty on centre-of-mass height makes that unprofitable, and it is the
    # only cap on the forward term.
    #
    # Measured on this model: the centre of mass sits at 1.012 m standing and
    # spans 0.935 .. 1.032 m across the reference gait cycle. 1.10 m leaves
    # ~7 cm of headroom above anything walking does, so the penalty is silent
    # during normal gait and bites only once the model leaves the ground.
    #
    # Charged per step, per metre of excess: 10 cm too high costs 1.0 a step
    # against a maximum per-step reward of 0.30, so height cannot be held
    # profitably for long.
    fly_threshold: float = 1.10      # COM height, metres
    fly_penalty: float = 10.0        # per metre of excess, per step

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

        # `weights=` replaces the whole set, which is how a caller adds a term
        # this stage does not currently weight. `w_<term>= ` still only changes
        # a weight that is already there, so a typo in one stays an error.
        weights = dict(overrides.pop("weights", self.reward.weights))
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
#   big penalty for falling      -> early termination: the forfeited remainder
#                                   of the episode, which is hundreds of
#                                   points. No explicit spike; see below.
#   0.1  for surviving 10 s      -> 0.10/step alive bonus over 1000 steps
#   0.7  for moving forward      -> paid ONCE at the end, against distance
#                                   covered, open-ended: 0.70 * 1000 * travel
#                                   / 1.0 m. 2 m pays 1400, 3 m pays 2100;
#                                   there is no distance at which it stops
#                                   paying. Height is capped instead, by
#                                   fly_penalty.
#   0.2  for following the gait  -> REMOVED. Tracking is gone, and with it
#                                   reference-state initialisation: every
#                                   episode starts from the same standing
#                                   pose. The 0.20 now weights `effort`; see
#                                   the reward spec below.
#
# Composed **additively**, not by v4's geometric mean. Geometrically, a
# tracking term near zero early in training would gate the velocity term to
# zero as well and neither would learn. Additively the weights already encode
# the priority: standing still scores 0.10, moving scores 0.80, moving on the
# reference scores 1.00.
#
# Over a full episode alive is at most 100 and tracking at most 200; forward is
# 700 for the reference distance of 1 m and has no maximum at all, so a policy
# that walks further always scores higher.
#
# Paying forward progress terminally is what stops it being farmed by falling.
# Per step, any topple produces v > 0.1 m/s and saturates the term; against
# distance at the end, a fall at step 32 has covered ~0.05 m, worth 35 before
# the survival factor scales it to 1.1.
#
# fall_penalty is **zero**, and that is deliberate. Falling is a part of
# training, not a cliff: it has to be something the policy can try, be scored
# for, and learn from. Early termination already prices it -- a fall at step 60
# forfeits ~940 steps of alive and tracking plus the whole forward payment,
# which is hundreds of points -- so an explicit penalty on top only adds a
# spike. At the 5.0 it used to be, the falling step scored -4.89 against a
# per-step maximum of 0.30: sixteen times the largest reward any step can earn,
# concentrated on one transition. That is a wall, and a value function fitting
# it spends its capacity on the wall rather than on what led to it.
#
# Travelling backwards is priced the same way, by the forward term being signed
# rather than clipped at zero: retreating costs proportionally, so there is a
# gradient back towards forwards from anywhere. Nothing terminates on it.
#
# RewardSpec.termination_report() still checks the remaining case -- that
# continuing is never worse than quitting -- and with a non-negative per-step
# reward it is satisfied outright.
STAGE_W = StageSpec(
    key="W",
    name=WALK_TRACK,
    target_vel=0.0,             # no velocity target; distance is paid at the end
    episode_steps=1000,         # 10 s at dt = 0.01
    # No reference tracking, and no reference-state initialisation with it:
    # every episode starts from the same solved standing pose.
    track_reference=False,
    rsi=False,
    max_tracking_error=None,    # nothing to be off by
    # Asymmetric by construction: right-leg stance just after contact,
    # left leg at toe-off, already travelling at 1.08 m/s. The one pose
    # below is this state, not a standing one. See init_state.py.
    init_state="InitStateH0918Gait10ActA.zml",
    # One pose, exactly. The reset noise is zero, so the only variation between
    # episodes is the policy's own action sampling.
    reset_position_std=0.0,
    reset_velocity_std=0.0,
    initial_forward_velocity=0.10,
    initial_forward_velocity_std=0.0,
    forward_bonus=0.70,
    forward_reference_distance=1.00,  # 0.10 m/s for 10 s = one full payment
    backward_multiplier=4.0,
    reward=RewardSpec(
        alive=0.10,
        shaping_scale=0.20,
        composition=ADDITIVE,
        fall_penalty=0.0,
        # `effort` holds the 0.20 that tracking used to. RewardSpec requires at
        # least one shaping term, and of the terms this env has it is the only
        # one that is not a behavioural objective in its own right: `height`,
        # `upright` and `velocity` would each pay for something the reward does
        # not ask for, while effort only chooses among the many activation
        # patterns that produce the same motion. This model has 290 muscles and
        # is hugely overactuated; without it, co-contraction is free.
        weights={"effort": 1.00},
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
