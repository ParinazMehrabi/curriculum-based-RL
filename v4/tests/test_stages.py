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


def test_neutral_pose_references_are_set():
    """Measured by scripts/calibrate.py; zero is the wrong reference.

    The calcn bodies sit ~0.10 m ahead of the pelvis body COM at rest and the
    crutches ~0.05 m ahead, so terms that compared against 0 were constants.
    """
    t = stages.TermParams()
    assert t.pelvis_foot_offset_ref == pytest.approx(0.101)
    assert t.crutch_offset_ref == pytest.approx(0.052)


def test_lag_terms_use_the_neutral_references():
    """Guards against a regression to measuring against zero."""
    src = env_source()
    tree = ast.parse(src)
    bodies = {
        node.name: ast.get_source_segment(src, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    # Under RSI these are re-measured per episode into _lag_ref / _crutch_ref;
    # the TermParams values remain the non-RSI fallback, set in reset().
    assert "self._lag_ref" in bodies["_term_pelvis_lag"]
    assert "self._crutch_ref" in bodies["_term_crutch_forward"]
    assert "pelvis_foot_offset_ref" in src
    assert "crutch_offset_ref" in src


def test_neutral_pose_scores_one_for_both_lag_terms():
    """At the measured neutral offsets neither term should penalise anything."""
    t = stages.TermParams()

    excess_lag = t.pelvis_foot_offset_ref - t.pelvis_foot_offset_ref
    assert rewards.gaussian(max(0.0, excess_lag), t.pelvis_lag_sigma) == pytest.approx(1.0)

    shortfall = (t.crutch_offset_ref - t.crutch_offset_ref) - t.crutch_forward_margin
    assert rewards.gaussian(max(0.0, shortfall), t.crutch_forward_sigma) == pytest.approx(1.0)


def test_lag_terms_discriminate_away_from_neutral():
    """Moving away from neutral must actually cost something."""
    t = stages.TermParams()

    # feet 5 cm further ahead of the pelvis than at rest
    lag = (t.pelvis_foot_offset_ref + 0.05) - t.pelvis_foot_offset_ref
    assert 0.1 < rewards.gaussian(lag, t.pelvis_lag_sigma) < 0.6

    # crutch 5 cm behind where it rests
    shortfall = (t.crutch_offset_ref - (t.crutch_offset_ref - 0.05)) - t.crutch_forward_margin
    assert 0.1 < rewards.gaussian(max(0.0, shortfall), t.crutch_forward_sigma) < 0.9


def test_only_stage_d_rewards_progress():
    """pelvis_forward belongs only where the model can actually travel."""
    assert "pelvis_forward" in STAGES["D"].reward.active_weights
    for key in ("A", "B", "C"):
        assert "pelvis_forward" not in STAGES[key].reward.active_weights, key


def test_standing_still_is_clearly_worse_than_walking_in_stage_d():
    """The reason stage D was reweighted.

    Standing still maximises posture, backward, displacement, crutch_forward and
    pelvis_lag simultaneously -- 49% of the original weight. Only velocity and
    pelvis_forward penalise it. If the gap between standing and walking is
    small, standing wins because it is easier and risks no fall penalty.
    """
    spec = STAGES["D"].reward
    d = STAGES["D"]

    def score(moving: bool):
        terms = {t: 1.0 for t in spec.required_terms}
        if not moving:
            # velocity at v=0 against a 0.03 target; no distance covered
            terms["velocity"] = rewards.gaussian(-d.target_vel, d.velocity_sigma)
            terms["pelvis_forward"] = 0.0
        total, _ = spec.compose(terms)
        return total

    standing, walking = score(False), score(True)
    assert walking > standing
    gap = (walking - standing) / walking
    assert gap > 0.35, "standing still is only %.0f%% worse than walking" % (100 * gap)


def test_stage_d_weights_moving_over_staying_put():
    """velocity plus pelvis_forward must outweigh the terms standing satisfies."""
    w = STAGES["D"].reward.active_weights
    moving = w["velocity"] + w["pelvis_forward"]
    still = w["backward"] + w["displacement"]
    assert moving > still


def test_stage_d_velocity_target_is_reachable_at_reset():
    """D asked for 0.03 m/s from a standstill; velocity never exceeded 0.043."""
    d = STAGES["D"]
    assert d.initial_forward_velocity > 0.0
    # Within one sigma of the target at reset, so step 1 is not already a loss.
    assert abs(d.initial_forward_velocity - d.target_vel) < d.velocity_sigma


def test_stage_d_crutch_sigma_matches_stage_c():
    """The tighter sigma capped D's crutch term at 0.22 against C's 0.97."""
    assert STAGES["D"].terms.cane_load_sigma_fraction == pytest.approx(
        STAGES["C"].terms.cane_load_sigma_fraction
    )


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


# -- deprl info-dict contract -----------------------------------------------


def test_info_keys_are_pinned_to_the_v3_set():
    """The eight keys v3 emitted, fixed and stage-independent.

    deprl's test_scone pre-allocates rwd_metrics and indexes it by the keys in
    info; it does not read REWARD_KEYS. Two runs died at the end of their first
    epoch because v4 emitted a stage-dependent set. v3 trained to millions of
    steps with exactly these, so the set must not drift.
    """
    src = env_source()
    tree = ast.parse(src)
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "V3_INFO_KEYS":
                    found = [e.value for e in node.value.elts]
    assert found is not None, "V3_INFO_KEYS not found in env.py"
    assert found == [
        "alive",
        "height",
        "posture",
        "crutch",
        "velocity",
        "backward",
        "displacement",
        "total",
    ]


def test_rwd_dict_is_never_none():
    """The regression test for the failure that killed two runs.

    deprl pre-allocates its metric buffers from environment.rwd_dict before the
    test episode, then indexes them by the keys found there afterwards. If reset
    nulls or empties rwd_dict, the buffer is empty and the loop raises KeyError
    on the first key. v3 never cleared it on reset; v4 did, and paid for it.
    """
    src = env_source()
    assert "self.rwd_dict: Dict[str, float] = {k: 0.0 for k in V3_INFO_KEYS}" in src
    assert "self.rwd_dict = None" not in src
    assert "self.rwd_dict: Optional[Dict[str, float]] = None" not in src


def test_reset_zeroes_values_but_keeps_keys():
    """reset must not rebind or clear rwd_dict, only zero its values."""
    src = env_source()
    tree = ast.parse(src)
    reset = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "reset"
    )
    body = ast.get_source_segment(src, reset)
    assert "for key in self.rwd_dict:" in body
    assert "self.rwd_dict = " not in body
    assert "self.rwd_dict.clear()" not in body


def test_get_rwd_dict_never_returns_empty():
    """It must not depend on a step having happened first."""
    src = env_source()
    tree = ast.parse(src)
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_rwd_dict"
    )
    body = ast.get_source_segment(src, fn)
    assert "if self.rwd_dict is None" not in body


def test_step_builds_info_from_the_pinned_dict():
    src = env_source()
    assert "info.update(self.rwd_dict)" in src


def test_v4_only_terms_stay_out_of_info():
    """crutch_forward, pelvis_lag and pelvis_forward have no v3 slot.

    They remain on env.term_values; putting them in info would reintroduce the
    KeyError this contract exists to prevent.
    """
    src = env_source()
    tree = ast.parse(src)
    pinned = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "V3_INFO_KEYS":
                    pinned = {e.value for e in node.value.elts}
    for term in ("crutch_forward", "pelvis_lag", "pelvis_forward"):
        assert term not in pinned, term


def test_every_stage_term_is_either_pinned_or_deliberately_excluded():
    """No stage may weight a term that silently vanishes from logging."""
    src = env_source()
    tree = ast.parse(src)
    pinned = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "V3_INFO_KEYS":
                    pinned = {e.value for e in node.value.elts}
    excluded = {"crutch_forward", "pelvis_lag", "pelvis_forward"}
    for key, stage in STAGES.items():
        for term in stage.reward.active_weights:
            assert term in pinned or term in excluded, (key, term)
