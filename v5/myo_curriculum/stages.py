"""The muscle-locomotion curriculum, as data.

Same shape as v4's stages.py: one `StageSpec` per stage, differing only in
coefficients, so there is one environment class rather than a family of
near-identical ones.

The task is different from v4's, and so are the terms. v4 drove a planar
9-torque skeleton with welded crutches; this drives a 3-D, 290-muscle full
body with no assistive device, so two kinds of term are new:

* **Out-of-plane terms.** v4's model was strictly sagittal -- every joint was a
  z-hinge -- so lateral drift and turning were structurally impossible and had
  no terms. Here they are the most common failure mode, hence `lateral` and
  `heading`.
* **An effort term.** 290 Hill-type muscles are hugely overactuated: many
  activation patterns produce the same motion, and most of them are
  co-contraction that a real person would not use. `effort` selects among them.
  A torque model with 9 actuators did not need this.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Tuple

from .rewards import GEOMETRIC, RewardSpec

STAND = "A-stand"
WALK = "B-walk"

STAGE_ORDER: Tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class TermParams:
    """Shape parameters for the individual reward terms."""

    # height: pelvis height measured against the settled standing height
    height_drop_sigma: float = 0.12

    # upright: trunk tilt away from vertical, in radians
    upright_sigma: float = 0.35

    # velocity: forward speed error, m/s
    velocity_sigma: float = 0.35

    # lateral: sideways speed, m/s. Tighter than `velocity` on purpose --
    # sideways motion is never wanted, whereas forward speed has a target.
    lateral_sigma: float = 0.25

    # heading: yaw away from +x, in radians
    heading_sigma: float = 0.50

    # effort: mean muscle activation that still scores well. 0.15 is a
    # deliberately loose budget -- tight effort penalties on an overactuated
    # model suppress motion before they suppress co-contraction.
    effort_target: float = 0.15
    effort_sigma: float = 0.25


@dataclass(frozen=True)
class StageSpec:
    key: str
    name: str
    reward: RewardSpec
    terms: TermParams = field(default_factory=TermParams)

    target_vel: float = 0.0
    episode_steps: int = 1000

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
            "velocity": 0.50,
            "height": 0.25,
            "upright": 0.25,
            "lateral": 0.20,
            "heading": 0.15,
            "effort": 0.15,
        },
    ),
)

STAGES: Dict[str, StageSpec] = {"A": STAGE_A, "B": STAGE_B}


def get_stage(key: str) -> StageSpec:
    k = str(key).upper()
    if k not in STAGES:
        raise KeyError(
            "unknown stage %r; expected one of %s" % (key, ", ".join(STAGE_ORDER))
        )
    return STAGES[k]


def describe_stages() -> str:
    return "\n".join(STAGES[k].describe() for k in STAGE_ORDER)
