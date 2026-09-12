"""Tests for the reward primitives and composition.

These encode the three behavioural claims the rewrite rests on:
  * smoothstep has no discontinuity (v3 had a 0 -> 0.9 jump at 20 N)
  * geometric composition cannot be farmed one term at a time
  * the per-step reward of every shipped stage is non-negative
"""
from __future__ import annotations

import math

import pytest

from _bootstrap import load

rewards, stages = load()

RewardSpec = rewards.RewardSpec
gaussian = rewards.gaussian
smoothstep = rewards.smoothstep
wgm = rewards.weighted_geometric_mean
wam = rewards.weighted_arithmetic_mean


# -- primitives -------------------------------------------------------------


def test_gaussian_peaks_at_zero_error():
    assert gaussian(0.0, 0.2) == pytest.approx(1.0)
    assert 0.0 < gaussian(0.4, 0.2) < gaussian(0.2, 0.2) < 1.0


def test_gaussian_is_symmetric_and_bounded():
    for e in (0.0, 0.05, 0.3, 5.0):
        assert gaussian(e, 0.1) == pytest.approx(gaussian(-e, 0.1))
        assert 0.0 <= gaussian(e, 0.1) <= 1.0


def test_gaussian_tolerates_zero_sigma():
    assert 0.0 <= gaussian(1.0, 0.0) <= 1.0


def test_smoothstep_endpoints():
    assert smoothstep(5.0, 10.0, 30.0) == 0.0
    assert smoothstep(10.0, 10.0, 30.0) == 0.0
    assert smoothstep(30.0, 10.0, 30.0) == 1.0
    assert smoothstep(99.0, 10.0, 30.0) == 1.0


def test_smoothstep_is_monotonic():
    xs = [i * 0.5 for i in range(0, 100)]
    vals = [smoothstep(x, 10.0, 30.0) for x in xs]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_smoothstep_has_no_cliff():
    """The regression test for v3's hard 20 N gate.

    v3 jumped from 0.0 to ~0.9 across an infinitesimal force change. Here the
    largest step over a 0.1 N increment must stay small.
    """
    step = 0.1
    worst = 0.0
    x = 0.0
    while x < 40.0:
        a = smoothstep(x, 10.0, 30.0)
        b = smoothstep(x + step, 10.0, 30.0)
        worst = max(worst, abs(b - a))
        x += step
    assert worst < 0.02, "largest jump was %.4f" % worst


def test_weighted_geometric_mean_basics():
    assert wgm({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0}) == pytest.approx(1.0)
    assert wgm({"a": 0.25, "b": 0.25}, {"a": 1.0, "b": 3.0}) == pytest.approx(0.25)
    assert wgm({"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 1.0}) == 0.0
    assert wgm({"a": 0.5, "b": 0.5}, {"a": 0.0, "b": 0.0}) == 0.0


def test_weighted_geometric_mean_matches_closed_form():
    got = wgm({"a": 0.4, "b": 0.9}, {"a": 1.0, "b": 1.0})
    assert got == pytest.approx(math.sqrt(0.4 * 0.9))


def test_weighted_geometric_mean_ignores_zero_weights():
    got = wgm({"a": 0.4, "b": 0.0}, {"a": 1.0, "b": 0.0})
    assert got == pytest.approx(0.4)


def test_weighted_means_stay_in_unit_interval():
    vals = {"a": 0.13, "b": 0.77, "c": 1.0}
    w = {"a": 0.2, "b": 0.45, "c": 0.25}
    assert 0.0 <= wgm(vals, w) <= 1.0
    assert 0.0 <= wam(vals, w) <= 1.0


def test_geometric_never_exceeds_arithmetic():
    vals = {"a": 0.2, "b": 0.8, "c": 0.5}
    w = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert wgm(vals, w) <= wam(vals, w) + 1e-12


# -- RewardSpec validation --------------------------------------------------


def test_spec_rejects_unknown_composition():
    with pytest.raises(ValueError):
        RewardSpec(weights={"a": 1.0}, composition="harmonic")


def test_spec_rejects_bad_term_floor():
    with pytest.raises(ValueError):
        RewardSpec(weights={"a": 1.0}, term_floor=1.0)
    with pytest.raises(ValueError):
        RewardSpec(weights={"a": 1.0}, term_floor=-0.1)


def test_spec_rejects_negative_weight():
    with pytest.raises(ValueError):
        RewardSpec(weights={"a": -1.0})


def test_spec_requires_a_positive_weight():
    with pytest.raises(ValueError):
        RewardSpec(weights={"a": 0.0})


def test_compose_reports_missing_terms():
    spec = RewardSpec(weights={"height": 1.0, "posture": 1.0})
    with pytest.raises(KeyError):
        spec.compose({"height": 1.0})


def test_required_terms_includes_legacy_penalties():
    spec = RewardSpec(
        weights={"height": 1.0},
        legacy_penalties={"backward": 0.45},
        legacy_bounds={"backward": 5.0},
    )
    assert spec.required_terms == ("backward", "height")


# -- the farming claim ------------------------------------------------------


def _farm_specs(floor):
    weights = {"height": 0.20, "posture": 0.45, "velocity": 0.25}
    common = dict(alive=0.15, shaping_scale=0.85, weights=weights, term_floor=floor)
    return (
        RewardSpec(composition=rewards.GEOMETRIC, **common),
        RewardSpec(composition=rewards.ADDITIVE, **common),
    )


@pytest.mark.parametrize("floor", [0.0, 0.05])
def test_geometric_composition_punishes_an_ignored_term(floor):
    """Standing still while ignoring velocity must not pay well."""
    geo, add = _farm_specs(floor)
    farmed = {"height": 1.0, "posture": 1.0, "velocity": 0.0}
    geo_total, _ = geo.compose(farmed)
    add_total, _ = add.compose(farmed)
    assert geo_total < add_total
    # The additive form still hands out most of the shaping budget.
    assert add_total > 0.70
    assert geo_total < 0.55


def test_geometric_composition_rewards_satisfying_everything():
    geo, _ = _farm_specs(0.05)
    total, breakdown = geo.compose(
        {"height": 1.0, "posture": 1.0, "velocity": 1.0}
    )
    assert total == pytest.approx(1.0)


def test_zero_floor_annihilates_and_nonzero_floor_does_not():
    geo_hard, _ = _farm_specs(0.0)
    geo_soft, _ = _farm_specs(0.05)
    farmed = {"height": 1.0, "posture": 1.0, "velocity": 0.0}
    hard, _ = geo_hard.compose(farmed)
    soft, _ = geo_soft.compose(farmed)
    assert hard == pytest.approx(geo_hard.alive)
    assert soft > hard


def test_breakdown_carries_raw_terms():
    geo, _ = _farm_specs(0.05)
    _, breakdown = geo.compose({"height": 0.5, "posture": 0.6, "velocity": 0.7})
    assert breakdown["height"] == pytest.approx(0.5)
    assert breakdown["velocity"] == pytest.approx(0.7)
    assert "total" in breakdown and "alive" in breakdown


def test_breakdown_keys_are_unprefixed():
    """deprl's test_scone indexes its metric buffers by these exact names.

    A "term_" prefix here surfaced as KeyError: 'term_height' inside
    custom_test_environment.py after a full epoch of training.
    """
    geo, _ = _farm_specs(0.05)
    _, breakdown = geo.compose({"height": 1.0, "posture": 1.0, "velocity": 1.0})
    assert not any(k.startswith("term_") for k in breakdown), sorted(breakdown)
    assert set(geo.required_terms) <= set(breakdown)


def test_breakdown_keys_match_declared_reward_keys():
    """The env exposes REWARD_KEYS; it must agree with what compose emits."""
    geo, _ = _farm_specs(0.05)
    _, breakdown = geo.compose({"height": 1.0, "posture": 1.0, "velocity": 1.0})
    declared = list(geo.required_terms) + ["alive", "total"]
    assert set(declared) == set(breakdown)


def test_shaping_is_not_on_the_wire():
    """Only keys v3 also emitted may reach deprl's metric buffers.

    'shaping' was the one novel key; it is recoverable from total and alive and
    is exposed as env.shaping_value instead.
    """
    geo, _ = _farm_specs(0.05)
    terms = {"height": 0.8, "posture": 0.7, "velocity": 0.6}
    total, breakdown = geo.compose(terms)
    assert "shaping" not in breakdown
    recovered = (total - geo.alive) / geo.shaping_scale
    assert 0.0 <= recovered <= 1.0
    assert geo.alive + geo.shaping_scale * recovered == pytest.approx(total)


def test_terms_are_clipped_into_unit_interval():
    geo, _ = _farm_specs(0.0)
    total, _ = geo.compose({"height": 5.0, "posture": 5.0, "velocity": 5.0})
    assert total == pytest.approx(1.0)
    total, _ = geo.compose({"height": -3.0, "posture": 1.0, "velocity": 1.0})
    assert total == pytest.approx(geo.alive)


# -- non-negativity and termination incentive -------------------------------


def test_shipped_specs_have_non_negative_floor():
    for key, spec in stages.STAGES.items():
        assert spec.reward.min_step_reward() >= 0.0, key


def test_shipped_specs_never_prefer_termination():
    for key, spec in stages.STAGES.items():
        report = spec.reward.termination_report(gamma=0.99)
        assert report["termination_preferred"] is False, (key, report["verdict"])
        assert "safe" in str(report["verdict"])


def test_random_term_draws_never_go_negative():
    import numpy as np

    rng = np.random.RandomState(0)
    for key, stage in stages.STAGES.items():
        spec = stage.reward
        for _ in range(200):
            terms = {t: float(rng.rand()) for t in spec.required_terms}
            total, _ = spec.compose(terms)
            assert total >= 0.0, (key, total)


def test_v3_style_stage_d_is_flagged_as_unsafe():
    """The configuration this rewrite replaces must be detected, not tolerated.

    v3 stage D subtracted up to 0.45 * 5.0 = 2.25 per step for backward drift
    against a one-off fall penalty of 5.0, making an immediate fall the best
    available outcome.
    """
    v3_like = RewardSpec(
        alive=0.15,
        shaping_scale=0.85,
        weights={"height": 0.20, "posture": 0.45},
        composition=rewards.ADDITIVE,
        term_floor=0.0,
        fall_penalty=5.0,
        legacy_penalties={"backward": 0.45},
        legacy_bounds={"backward": 5.0},
    )
    report = v3_like.termination_report(gamma=0.99)
    assert report["min_step_reward"] < 0.0
    assert report["termination_preferred"] is True
    assert "UNSAFE" in str(report["verdict"])


def test_unbounded_legacy_penalty_reports_unbounded():
    spec = RewardSpec(
        weights={"height": 1.0},
        legacy_penalties={"backward": 0.45},
    )
    assert spec.min_step_reward() == float("-inf")
    report = spec.termination_report()
    assert report["termination_preferred"] is True


def test_warn_if_unsafe_emits_for_bad_spec():
    spec = RewardSpec(weights={"height": 1.0}, legacy_penalties={"backward": 1.0})
    with pytest.warns(RuntimeWarning):
        spec.warn_if_unsafe(label="test")
