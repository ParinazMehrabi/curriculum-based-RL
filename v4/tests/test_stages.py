"""Tests for the stage definitions and the override mechanism."""
from __future__ import annotations

import ast

import pytest

from _bootstrap import env_source, load

rewards, stages = load()

STAGES = stages.STAGES
STAGE_ORDER = stages.STAGE_ORDER


def _implemented_terms():
    """Term names env.py can actually evaluate, read without importing gym."""
    tree = ast.parse(env_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_TERM_FNS":
                    return {k.value for k in node.value.keys}
    raise AssertionError("_TERM_FNS not found in env.py")


# -- stage sanity -----------------------------------------------------------


def test_all_stages_present_and_ordered():
    assert set(STAGES) == set(STAGE_ORDER)
    assert STAGE_ORDER == ("A", "B", "C", "D")


def test_stage_names_are_distinct():
    names = [s.name for s in STAGES.values()]
    assert len(set(names)) == len(names)


def test_every_weighted_term_has_an_implementation():
    """Catches a stage asking for a term nobody wrote."""
    available = _implemented_terms()
    for key, stage in STAGES.items():
        for term in stage.reward.active_weights:
            assert term in available, "stage %s wants unimplemented term %r" % (key, term)


def test_get_stage_is_case_insensitive():
    assert stages.get_stage("a") is STAGES["A"]
    assert stages.get_stage("D") is STAGES["D"]


def test_get_stage_rejects_unknown():
    with pytest.raises(KeyError):
        stages.get_stage("Z")


def test_max_step_reward_is_one_for_every_stage():
    """alive + shaping_scale == 1 keeps the reward scale comparable across stages."""
    for key, stage in STAGES.items():
        spec = stage.reward
        assert spec.alive + spec.shaping_scale == pytest.approx(1.0), key
        total, _ = spec.compose({t: 1.0 for t in spec.required_terms})
        assert total == pytest.approx(1.0), key


def test_no_stage_uses_legacy_subtractive_penalties():
    for key, stage in STAGES.items():
        assert not stage.reward.legacy_penalties, key


# -- curriculum shape -------------------------------------------------------


def test_curriculum_adds_terms_monotonically():
    """Each stage should introduce terms, never silently drop one."""
    seen = set()
    for key in STAGE_ORDER:
        terms = set(STAGES[key].reward.active_weights)
        assert seen <= terms, "stage %s dropped %s" % (key, sorted(seen - terms))
        seen = terms


def test_only_locomotion_stages_have_a_velocity_target():
    assert STAGES["A"].target_vel == 0.0
    assert STAGES["B"].target_vel == 0.0
    assert STAGES["C"].target_vel > 0.0
    assert STAGES["D"].target_vel > 0.0


def test_velocity_weight_requires_positive_target():
    for key, stage in STAGES.items():
        if "velocity" in stage.reward.active_weights:
            assert stage.target_vel > 0.0, key
            assert stage.velocity_sigma > 0.0, key


def test_stage_rejects_velocity_weight_without_target():
    with pytest.raises(ValueError):
        stages.StageSpec(
            name="broken",
            reward=rewards.RewardSpec(weights={"velocity": 1.0}),
            target_vel=0.0,
        )


def test_crutch_requirements_are_derived_from_weights():
    assert STAGES["A"].needs_crutch_force is False
    assert STAGES["A"].needs_crutch_pose is False
    assert STAGES["B"].needs_crutch_force is True
    assert STAGES["B"].needs_crutch_pose is False
    assert STAGES["D"].needs_crutch_force is True
    assert STAGES["D"].needs_crutch_pose is True


def test_stage_d_init_load_is_unambiguous():
    """v3's stage D YAML set init_load twice (0.5 then 0.4); 0.4 silently won."""
    assert STAGES["D"].init_load == 0.5
    assert len({s.init_load for s in STAGES.values()}) == 1


def test_crutch_gate_opens_below_the_target_load():
    """The smooth gate must be fully open at the stage's target crutch load.

    Otherwise the gate would be clipping the very behaviour it is meant to
    reward. Assumes a body weight near the model's 64.85 kg.
    """
    body_weight_n = 64.85 * 9.81
    for key, stage in STAGES.items():
        if not stage.needs_crutch_force:
            continue
        t = stage.terms
        target_n = t.cane_target_load_fraction * body_weight_n
        assert target_n > t.cane_gate_hi_n, (key, target_n, t.cane_gate_hi_n)


def test_describe_mentions_the_heaviest_term():
    text = STAGES["A"].describe()
    assert "posture" in text
    assert "A-stand" in text


# -- overrides --------------------------------------------------------------


def test_overrides_return_self_when_empty():
    stage = STAGES["A"]
    assert stage.with_overrides() is stage


def test_override_reward_weight():
    stage = STAGES["A"].with_overrides(w_posture=0.9)
    assert stage.reward.weights["posture"] == pytest.approx(0.9)
    assert STAGES["A"].reward.weights["posture"] == pytest.approx(0.45)


def test_override_can_introduce_a_new_term():
    stage = STAGES["A"].with_overrides(w_crutch=0.3)
    assert "crutch" in stage.reward.active_weights
    assert stage.needs_crutch_force is True


def test_override_can_disable_a_term():
    stage = STAGES["B"].with_overrides(w_crutch=0.0)
    assert "crutch" not in stage.reward.active_weights
    assert stage.needs_crutch_force is False


def test_override_reward_scalar():
    stage = STAGES["A"].with_overrides(fall_penalty=9.0, alive=0.3)
    assert stage.reward.fall_penalty == pytest.approx(9.0)
    assert stage.reward.alive == pytest.approx(0.3)


def test_override_term_param():
    stage = STAGES["A"].with_overrides(lumbar_sigma=0.5)
    assert stage.terms.lumbar_sigma == pytest.approx(0.5)
    assert STAGES["A"].terms.lumbar_sigma == pytest.approx(0.12)


def test_override_stage_field():
    stage = STAGES["A"].with_overrides(init_load=0.4, episode_steps=500)
    assert stage.init_load == pytest.approx(0.4)
    assert stage.episode_steps == 500


def test_override_rejects_unknown_key():
    """v3 swallowed configuration mistakes; a YAML typo must stop the run."""
    with pytest.raises(KeyError) as exc:
        STAGES["A"].with_overrides(postrue_reward_coeff=0.4)
    assert "postrue_reward_coeff" in str(exc.value)


def test_override_rejects_v3_coefficient_names():
    """The v3 names are gone; asking for them should say so loudly."""
    for old in ("posture_reward_coeff", "crutch_reward_coeff", "backward_penalty_coeff"):
        with pytest.raises(KeyError):
            STAGES["D"].with_overrides(**{old: 0.1})


def test_override_validation_still_applies():
    with pytest.raises(ValueError):
        STAGES["C"].with_overrides(target_vel=0.0)
    with pytest.raises(ValueError):
        STAGES["A"].with_overrides(term_floor=2.0)


def test_overridden_stage_stays_reward_safe():
    stage = STAGES["D"].with_overrides(alive=0.05, w_backward=1.0)
    assert stage.reward.min_step_reward() >= 0.0
    assert stage.reward.termination_report()["termination_preferred"] is False
