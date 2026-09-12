"""Reward terms and composition for the crutch curriculum.

This module deliberately imports nothing from sconegym/sconepy so the reward
maths can be unit-tested without a simulator present.

Two invariants hold everywhere below, and they are the point of the rewrite:

1. Every shaping term is a scalar in [0, 1] where 1 means "ideal". Penalties
   are expressed as terms that fall toward 0, never as subtractions.
2. Consequently the per-step reward is non-negative. An agent can never do
   better by ending the episode early than by continuing, so the
   termination-seeking behaviour that an unbounded per-step penalty creates is
   structurally impossible rather than merely tuned away.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

import numpy as np

GEOMETRIC = "geometric"
ADDITIVE = "additive"
COMPOSITIONS = (GEOMETRIC, ADDITIVE)


def gaussian(error: float, sigma: float) -> float:
    """exp(-(error/sigma)^2), clamped into [0, 1]."""
    s = max(float(sigma), 1e-9)
    return float(np.exp(-((float(error) / s) ** 2)))


def smoothstep(x: float, lo: float, hi: float) -> float:
    """C1-continuous ramp: 0 at or below lo, 1 at or above hi.

    Replaces the hard "if force < 20: return 0" gate, which put a step
    discontinuity in the reward at a boundary the policy can oscillate across.
    """
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = float(np.clip((float(x) - lo) / (hi - lo), 0.0, 1.0))
    return float(t * t * (3.0 - 2.0 * t))


def weighted_geometric_mean(
    values: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    """Weighted geometric mean of values in [0, 1]; result in [0, 1].

    Every term gates every other: a value of 0 anywhere zeroes the result. That
    is what stops the policy farming the cheap terms (alive, height, posture)
    while ignoring the expensive one (velocity).
    """
    total_w = float(sum(weights.values()))
    if total_w <= 0.0:
        return 0.0
    acc = 0.0
    for name, w in weights.items():
        if w <= 0.0:
            continue
        v = float(values[name])
        if v <= 0.0:
            return 0.0
        acc += (float(w) / total_w) * math.log(v)
    return float(math.exp(acc))


def weighted_arithmetic_mean(
    values: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    """Weight-normalised arithmetic mean, for ablation against the geometric form."""
    total_w = float(sum(w for w in weights.values() if w > 0.0))
    if total_w <= 0.0:
        return 0.0
    acc = 0.0
    for name, w in weights.items():
        if w <= 0.0:
            continue
        acc += (float(w) / total_w) * float(values[name])
    return float(acc)


@dataclass(frozen=True)
class RewardSpec:
    """Declarative reward definition for one curriculum stage.

    step reward = alive + shaping_scale * compose(terms) - legacy_penalties

    With legacy_penalties empty (the default for every shipped stage) the step
    reward is bounded below by alive + shaping_scale * term_floor >= 0.
    """

    alive: float = 0.0
    shaping_scale: float = 1.0
    weights: Mapping[str, float] = field(default_factory=dict)
    composition: str = GEOMETRIC
    term_floor: float = 0.05
    fall_penalty: float = 5.0
    # Opt-in reproduction of the v3 subtractive penalties, for ablation only.
    # name -> coefficient; the named term is expected as a raw magnitude >= 0.
    legacy_penalties: Mapping[str, float] = field(default_factory=dict)
    legacy_bounds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.composition not in COMPOSITIONS:
            raise ValueError(
                "composition must be one of %r, got %r"
                % (COMPOSITIONS, self.composition)
            )
        if not 0.0 <= self.term_floor < 1.0:
            raise ValueError("term_floor must lie in [0, 1)")
        if self.alive < 0.0 or self.shaping_scale < 0.0:
            raise ValueError("alive and shaping_scale must be non-negative")
        for name, w in self.weights.items():
            if w < 0.0:
                raise ValueError("weight for %r must be non-negative" % name)
        if not self.active_weights:
            raise ValueError("at least one shaping weight must be positive")

    @property
    def active_weights(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.weights.items() if v > 0.0}

    @property
    def required_terms(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.active_weights) | set(self.legacy_penalties)))

    def _floored(self, value: float) -> float:
        v = float(np.clip(value, 0.0, 1.0))
        return self.term_floor + (1.0 - self.term_floor) * v

    def compose(self, terms: Mapping[str, float]) -> Tuple[float, Dict[str, float]]:
        """Return (total, breakdown) for one step.

        breakdown carries the raw term values plus the composed shaping value
        and total, and is what gets handed to the gym info dict. A geometric
        mean does not decompose into per-term contributions, so the raw values
        are reported rather than a fabricated attribution.
        """
        active = self.active_weights
        missing = [t for t in self.required_terms if t not in terms]
        if missing:
            raise KeyError("missing reward terms: %s" % ", ".join(missing))

        raw = {k: float(terms[k]) for k in self.required_terms}
        floored = {k: self._floored(raw[k]) for k in active}

        if self.composition == GEOMETRIC:
            shaping = weighted_geometric_mean(floored, active)
        else:
            shaping = weighted_arithmetic_mean(floored, active)

        penalty = 0.0
        for name, coeff in self.legacy_penalties.items():
            penalty += float(coeff) * max(0.0, float(terms[name]))

        total = self.alive + self.shaping_scale * shaping - penalty

        # Keys are the bare term names, matching the contract v3 used. deprl's
        # test_scone pre-allocates rwd_metrics and indexes it by these keys, so a
        # prefix here becomes a KeyError there rather than a missing metric.
        breakdown = dict(raw)
        breakdown["shaping"] = float(shaping)
        breakdown["alive"] = float(self.alive)
        if self.legacy_penalties:
            breakdown["legacy_penalty"] = float(penalty)
        breakdown["total"] = float(total)
        return float(total), breakdown

    def min_step_reward(self) -> float:
        """Tightest lower bound on a single non-terminal step reward.

        Returns -inf when a legacy penalty is declared without a bound, since
        the reward is then genuinely unbounded below.
        """
        worst = self.alive + self.shaping_scale * self._floored(0.0)
        for name, coeff in self.legacy_penalties.items():
            if name not in self.legacy_bounds:
                return float("-inf")
            worst -= float(coeff) * float(self.legacy_bounds[name])
        return float(worst)

    def termination_report(self, gamma: float = 0.99) -> Dict[str, object]:
        """Diagnose whether the agent is better off terminating the episode.

        Continuing forever at the worst per-step reward is worth
        min_step_reward / (1 - gamma). Falling now is worth -fall_penalty.
        If falling is the better of the two, the policy will learn to fall.
        """
        r_min = self.min_step_reward()
        terminate_value = -float(self.fall_penalty)
        if not math.isfinite(r_min):
            return {
                "min_step_reward": r_min,
                "worst_continuation": float("-inf"),
                "terminate_value": terminate_value,
                "termination_preferred": True,
                "verdict": "unbounded per-step penalty: termination is always preferred",
            }
        worst_continuation = r_min / max(1.0 - float(gamma), 1e-9)
        preferred = terminate_value > worst_continuation
        if r_min >= 0.0:
            verdict = "safe: per-step reward is non-negative"
        elif preferred:
            verdict = "UNSAFE: falling (%.2f) beats the worst continuation (%.2f)" % (
                terminate_value,
                worst_continuation,
            )
        else:
            verdict = "bounded: continuation dominates falling, but reward can go negative"
        return {
            "min_step_reward": float(r_min),
            "worst_continuation": float(worst_continuation),
            "terminate_value": terminate_value,
            "termination_preferred": bool(preferred),
            "verdict": verdict,
        }

    def warn_if_unsafe(self, gamma: float = 0.99, label: str = "") -> None:
        report = self.termination_report(gamma)
        if report["termination_preferred"]:
            warnings.warn(
                "[%s] reward spec makes early termination attractive: %s"
                % (label or "RewardSpec", report["verdict"]),
                RuntimeWarning,
                stacklevel=2,
            )
