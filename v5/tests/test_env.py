"""Tests for the muscle-locomotion environment.

Run from v5/ with the Python 3.12 environment:

    ..\\.venv-myo\\Scripts\\python.exe -m pytest tests -q

Several of these are regression tests for bugs that were live during
development and that a training run would have hidden rather than surfaced --
they are marked as such, because they are the ones worth keeping.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("myosuite")

from myo_curriculum import rewards as rewards_shim  # noqa: E402
from myo_curriculum.env import (  # noqa: E402
    BALL_CONTACTS,
    BALL_RADIUS,
    INDEPENDENT_JOINTS,
    MYO_FOOT_GEOMS,
    MyoLocomotionEnv,
)
from myo_curriculum.stages import STAGE_ORDER, STAGES, TermParams, get_stage  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = MyoLocomotionEnv(stage="A", seed=0)
    yield e
    e.close()


@pytest.fixture(scope="module")
def walk_env():
    e = MyoLocomotionEnv(stage="B", seed=0)
    yield e
    e.close()


# -- the reward shim -------------------------------------------------------


def test_rewards_come_from_v4_not_a_copy():
    """The single-source-of-truth property, asserted rather than assumed."""
    assert rewards_shim.V4_REWARDS.is_file()
    assert rewards_shim.V4_REWARDS.parts[-3:] == (
        "v4",
        "sconegym_crutch_v4",
        "rewards.py",
    )
    assert rewards_shim.gaussian(0.0, 1.0) == pytest.approx(1.0)
    assert 0.0 < rewards_shim.gaussian(1.0, 1.0) < 1.0


def test_reward_spec_bounds_hold_for_every_stage():
    for key in STAGE_ORDER:
        spec = get_stage(key).reward
        report = spec.termination_report(gamma=0.99)
        assert float(report["min_step_reward"]) >= 0.0, key
        assert not report["termination_preferred"], key


# -- stages ----------------------------------------------------------------


def test_unknown_override_raises():
    with pytest.raises(KeyError, match="unknown override"):
        get_stage("A").with_overrides(not_a_parameter=1.0)


def test_unknown_weight_override_raises():
    with pytest.raises(KeyError, match="no reward term"):
        get_stage("A").with_overrides(w_velocity=1.0)  # stage A has no velocity term


def test_known_overrides_apply():
    spec = get_stage("B").with_overrides(w_velocity=0.9, target_vel=0.5, velocity_sigma=1.0)
    assert spec.reward.weights["velocity"] == pytest.approx(0.9)
    assert spec.target_vel == pytest.approx(0.5)
    assert spec.terms.velocity_sigma == pytest.approx(1.0)


def test_stage_a_has_no_velocity_term():
    """Standing must not be scored by a velocity term with a zero target.

    A zero-target velocity term rewards freezing, and makes the A->B
    transition a discrete change in what the reward measures.
    """
    assert "velocity" not in STAGES["A"].reward.weights
    assert "velocity" in STAGES["B"].reward.weights


def test_walking_beats_standing_still_in_stage_b():
    """The v4 stage-D lesson: most terms are maximised by standing still."""
    spec = STAGES["B"]
    still = {"velocity": rewards_shim.gaussian(-spec.target_vel, spec.terms.velocity_sigma),
             "height": 1.0, "upright": 1.0, "lateral": 1.0, "heading": 1.0, "effort": 1.0}
    walking = dict.fromkeys(still, 1.0)
    still_total, _ = spec.reward.compose(still)
    walk_total, _ = spec.reward.compose(walking)
    assert still_total < 0.75 * walk_total


# -- model wiring ----------------------------------------------------------


def test_model_is_muscle_actuated(env):
    assert env.n_muscle == 290
    assert env.model.na == env.n_muscle
    assert env.n_act == env.model.nu


def test_independent_joints_exclude_constrained_ones(env):
    """The 29 constrained joints must never be written directly.

    knee_angle_r_translation1 and friends follow knee_angle_r through equality
    constraints; posing them at reset would fight the solver.
    """
    constrained = set()
    for i in range(env.model.neq):
        if env.model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, env.model.eq_obj1id[i])
            constrained.add(name)
    assert constrained, "expected this model to use equality constraints"
    assert not (set(INDEPENDENT_JOINTS) & constrained)
    assert len(env.joint_qpos_adr) == len(INDEPENDENT_JOINTS) == 17


def test_observation_layout_matches_the_space(env):
    layout = env.obs_layout()
    assert sum(w for _, w in layout) == env.observation_space.shape[0]
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()


def test_muscle_activation_is_observed(env):
    """Without it the MDP is not Markov -- 290 hidden state variables."""
    names = [n for n, _ in env.obs_layout()]
    assert "muscle_activation" in names
    width = dict(env.obs_layout())["muscle_activation"]
    assert width == env.model.na


def test_activation_can_be_excluded_for_an_ablation():
    e = MyoLocomotionEnv(stage="A", include_activation=False, seed=0)
    try:
        assert "muscle_activation" not in [n for n, _ in e.obs_layout()]
        assert e.observation_space.shape[0] < 631
    finally:
        e.close()


# -- posture frame (regression) --------------------------------------------


def test_trunk_is_near_vertical_at_reset(env):
    """Regression: body frames here are locally y-up inside a z-up world.

    Reading a body's local +z as "up" measured an 89-degree tilt on a model
    standing straight, so every episode terminated on step 1. Posture is now
    derived from the pelvis-to-head vector instead.
    """
    env.reset(seed=0)
    assert env.trunk_tilt() < 0.4
    assert not env._is_fallen()


def test_reset_pose_is_not_terminal_for_any_stage():
    for key in STAGE_ORDER:
        e = MyoLocomotionEnv(stage=key, seed=0)
        try:
            e.reset(seed=0)
            assert not e._is_fallen(), key
        finally:
            e.close()


def test_heading_error_is_zero_at_reset(walk_env):
    walk_env.reset(seed=0)
    assert abs(walk_env.heading_error()) < 1e-9


def test_initial_push_is_along_the_facing_direction(walk_env):
    """Regression: the model's neutral pose faces ~109 degrees, not +x.

    Setting initial_forward_velocity on the root's world-x dof launched it
    mostly sideways, and the lateral term then punished the environment's own
    initial condition (it scored 0.23 at reset instead of 1.00).
    """
    walk_env.reset(seed=0)
    forward, lateral = walk_env.planar_velocity()
    assert forward > 0.15
    assert abs(lateral) < 0.05
    assert walk_env.compute_terms()["lateral"] > 0.9


# -- the standing stance and foot contact (regression) ---------------------


def test_reset_stance_is_plantigrade_on_both_feet(env):
    """Regression: the model used to start up on its toes, heels 23 mm up.

    The shipped keyframe is a mid-stride pose -- hips 0.43 rad apart, feet
    0.23 m apart along the facing direction -- and dropping the model from it
    landed it on its forefeet with both heels in the air. That is a bad start
    for a locomotion curriculum: near-singular, it biases the ankle
    plantarflexors from step one, and it leaves no heel contact for a gait
    reward to read.
    """
    for seed in range(12):
        env.reset(seed=seed)
        for foot in ("calcn_r", "calcn_l"):
            heel, toe = env._heel_z(foot), env._toe_z(foot)
            assert abs(heel) < 1e-4, "%s heel is %.5f m off the floor" % (foot, heel)
            assert abs(toe) < 1e-4, "%s toe is %.5f m off the floor" % (foot, toe)


def test_both_heels_are_on_the_floor_after_randomisation(env):
    """Reset noise must not be able to lift a foot.

    0.02 rad at the ankle moves a 0.2 m foot 4 mm, and a hip or knee
    perturbation moved a whole foot by up to 30 mm, so the reset re-seats both
    feet against the randomised pose.
    """
    heights = []
    for seed in range(20):
        env.reset(seed=seed)
        heights.append(max(env._heel_z("calcn_r"), env._heel_z("calcn_l")))
    assert max(heights) < 1e-4, "worst heel height %.6f m" % max(heights)


def test_heel_and_toe_loads_are_reported_separately(env):
    """A single per-foot total cannot distinguish heel strike from toe-off."""
    env.reset(seed=0)
    loads = env.contact_loads()
    assert loads.shape == (4,)
    assert np.all(loads >= 0.0)
    per_foot = env.foot_contact_loads()
    assert per_foot[0] == pytest.approx(loads[0] + loads[1])
    assert per_foot[1] == pytest.approx(loads[2] + loads[3])
    heels = env.heel_contact_loads()
    assert heels[0] == pytest.approx(loads[0])
    assert heels[1] == pytest.approx(loads[2])


def test_contact_load_is_in_the_observation_split_four_ways(env):
    assert dict(env.obs_layout())["contact_load_heel_toe"] == 4


def test_reset_carries_about_one_body_weight(env):
    """A balanced stance is in static equilibrium; a toppling one is not.

    Before the COM was targeted at the base of support rather than the calcn
    midpoint, this read 1.47 body weights at reset.
    """
    for seed in range(8):
        env.reset(seed=seed)
        assert 0.7 < env.contact_loads().sum() < 1.6


def test_reset_com_sits_over_the_base_of_support(env):
    for seed in range(8):
        env.reset(seed=seed)
        offset = np.asarray(env.data.subtree_com[0])[:2] - env._support_centroid()[:2]
        assert np.abs(offset).max() < 5e-3, "COM is %s m off the base" % offset


def test_reset_trunk_is_upright(env):
    for seed in range(8):
        env.reset(seed=seed)
        assert env.trunk_tilt() < 0.15


def test_stance_solve_converged(env):
    assert env.stance_residual < 1e-4
    env.reset(seed=0)
    assert env.seat_residual < 1e-4


def test_anatomical_axes_are_orthonormal(env):
    env.reset(seed=0)
    right, fwd = env.right_axis(), env.forward_axis()
    assert np.linalg.norm(right) == pytest.approx(1.0)
    assert np.linalg.norm(fwd) == pytest.approx(1.0)
    assert abs(float(right @ fwd)) < 1e-9


# -- reward gradient (regression) ------------------------------------------


def test_every_term_has_gradient_at_reset():
    """Regression: stage B's velocity target was unreachable from standstill.

    With sigma 0.35 against a 1.2 m/s target a motionless model scored 7e-6,
    which the term floor then flattened -- no gradient where the policy starts.
    """
    for key in STAGE_ORDER:
        e = MyoLocomotionEnv(stage=key, seed=0)
        try:
            e.reset(seed=0)
            terms = e.compute_terms()
            for name, value in terms.items():
                assert value >= 0.02, "%s stage %s is pinned at %.5f" % (name, key, value)
        finally:
            e.close()


def test_effort_term_discriminates(env):
    """Low activation must score better than high, or it selects nothing."""
    env.reset(seed=0)
    env.data.act[:] = 0.05
    low = env._term_effort()
    env.data.act[:] = 0.8
    high = env._term_effort()
    assert low > high
    assert 0.0 <= high <= low <= 1.0


# -- stepping --------------------------------------------------------------


def test_step_returns_the_gymnasium_five_tuple(env):
    env.reset(seed=0)
    out = env.step(np.zeros(env.n_act, np.float32))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "curriculum_stage" in info


def test_non_terminal_reward_is_never_negative(env):
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(120):
        _, reward, terminated, truncated, _ = env.step(
            rng.uniform(-1, 1, env.n_act).astype(np.float32)
        )
        if terminated:
            break
        assert reward >= 0.0
        if truncated:
            break


def test_action_rate_limit_is_enforced(env):
    env.reset(seed=0)
    limit = env.stage_spec.action_rate_limit
    before = env.prev_action.copy()
    env.step(np.ones(env.n_act, np.float32))
    assert np.all(np.abs(env.prev_action - before) <= limit + 1e-6)


def test_zero_action_is_half_activation(env):
    """The policy's zero is mid-activation, not rest -- easy to forget."""
    env.reset(seed=0)
    for _ in range(20):
        env.step(np.zeros(env.n_act, np.float32))
    assert env.data.ctrl.min() >= 0.0
    assert env.data.ctrl.max() <= 1.0
    assert env.data.ctrl.mean() == pytest.approx(0.5, abs=0.05)


def test_same_seed_gives_the_same_rollout(env):
    def run():
        env.reset(seed=7)
        rng = np.random.default_rng(3)
        total = 0.0
        for _ in range(40):
            _, r, term, trunc, _ = env.step(rng.uniform(-1, 1, env.n_act).astype(np.float32))
            total += r
            if term or trunc:
                break
        return total

    assert run() == pytest.approx(run())


def test_wrong_action_shape_raises(env):
    env.reset(seed=0)
    with pytest.raises(ValueError, match="expected action shape"):
        env.step(np.zeros(5, np.float32))


def test_pointing_it_at_a_torque_model_raises():
    """The v4 MJCF has no muscles; this env must refuse it rather than run.

    Either guard may fire first -- the joint table is resolved before the
    actuators are inspected, and v4's planar skeleton is missing thirteen of
    the seventeen joints as well as all 290 muscles. What matters is that it
    refuses at construction rather than training against the wrong body.
    """
    torque_model = V5.parent / "models" / "mjcf" / "rajagopal_crutch_2d.xml"
    if not torque_model.is_file():
        pytest.skip("v4 MJCF not generated")
    with pytest.raises(RuntimeError, match="missing expected joints|no muscle actuators"):
        MyoLocomotionEnv(stage="A", model_path=torque_model, ball_contacts=False)


def test_a_muscleless_model_is_refused_by_the_actuator_check(tmp_path):
    """The actuator guard on its own, with the joint check satisfied."""
    xml = tmp_path / "torque_only.xml"
    joints = "\n".join(
        '<body name="seg_%s" pos="0 0 %g">'
        '<joint name="%s" type="hinge" axis="0 0 1"/>'
        '<geom type="sphere" size="0.05"/>'
        "</body>" % (n, 0.1 * (i + 1), n)
        for i, n in enumerate(INDEPENDENT_JOINTS)
    )
    xml.write_text(
        '<mujoco><worldbody><body name="pelvis"><freejoint name="root"/>'
        '<geom type="sphere" size="0.1"/>'
        '<body name="torso"><geom type="sphere" size="0.1"/>'
        '<body name="head"><geom type="sphere" size="0.1"/></body></body>'
        '<body name="femur_r" pos="0.1 0 0"><geom type="sphere" size="0.05"/></body>'
        '<body name="femur_l" pos="-0.1 0 0"><geom type="sphere" size="0.05"/></body>'
        '<body name="calcn_r" pos="0.1 0 -0.5"><geom type="sphere" size="0.05"/></body>'
        '<body name="calcn_l" pos="-0.1 0 -0.5"><geom type="sphere" size="0.05"/></body>'
        + joints
        + "</body></worldbody>"
        '<actuator><motor joint="%s"/></actuator></mujoco>' % INDEPENDENT_JOINTS[0],
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="no muscle actuators"):
        MyoLocomotionEnv(stage="A", model_path=xml, ball_contacts=False)

# -- two-ball foot contact -------------------------------------------------


def test_each_foot_has_exactly_two_contact_balls(env):
    """The cane model's contact scheme: one sphere at the heel, one at the toe.

    MyoSuite wraps each foot in five capsules and an ellipsoid instead, which
    is a different contact model -- a rolling sole with no clean heel/toe
    split to read gait phase from.
    """
    collidable = set()
    for g in range(env.model.ngeom):
        if env.model.geom_contype[g] or env.model.geom_conaffinity[g]:
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name in BALL_CONTACTS or name in MYO_FOOT_GEOMS:
                collidable.add(name)
    assert collidable == set(BALL_CONTACTS)

    for name in BALL_CONTACTS:
        g = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert g >= 0, name
        assert env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        assert float(env.model.geom_size[g][0]) == pytest.approx(BALL_RADIUS)


def test_contact_balls_are_massless(env):
    """A contact primitive must not change the segment's inertia.

    Left at MuJoCo's default density the four spheres added 0.3 kg.
    """
    stock = MyoLocomotionEnv(stage="A", ball_contacts=False, seed=0)
    try:
        assert env.body_weight == pytest.approx(stock.body_weight, abs=0.05)
    finally:
        stock.close()


def test_myosuite_foot_geoms_are_kept_but_not_collidable(env):
    """They still draw the foot; they just no longer carry contact."""
    for name in MYO_FOOT_GEOMS:
        g = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert g >= 0, "%s should still exist for rendering" % name
        assert env.model.geom_contype[g] == 0
        assert env.model.geom_conaffinity[g] == 0


def test_the_feet_do_not_collide_with_each_other(env):
    """Regression: the solved stance put the feet 0.055 m apart laterally.

    MyoSuite ships 14 leg-to-leg collision pairs, which bypass
    contype/conaffinity entirely. With the legs that close the foot pair fired
    with 569 N of spurious force -- more than half body weight -- which swamped
    the real ground reaction and starved a heel of load. Nothing in the model
    stops the legs crossing, so stance width has to be constrained explicitly.
    """
    floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for seed in range(12):
        env.reset(seed=seed)
        for i in range(env.data.ncon):
            con = env.data.contact[i]
            assert floor in (con.geom1, con.geom2), (
                "non-floor contact at reset: %s <-> %s"
                % (
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1),
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2),
                )
            )


def test_stance_is_parallel_and_hip_width(env):
    for seed in range(8):
        env.reset(seed=seed)
        assert env.stance_width() == pytest.approx(
            env.stage_spec.stance_width, abs=1e-3
        )
        # Regression: left free, the solve settled on a 0.110 m split stance,
        # which loads the feet diagonally.
        assert abs(env.foot_stagger()) < 1e-3


def test_all_four_balls_touch_the_floor_at_reset(env):
    """Contact, as distinct from force.

    Whether a given ball also *carries* load is the solver's to decide: four
    coplanar point contacts against three equilibrium equations is
    indeterminate, so one of them routinely comes out at zero. Touching is what
    this env guarantees.
    """
    floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for seed in range(8):
        env.reset(seed=seed)
        touching = set()
        for i in range(env.data.ncon):
            con = env.data.contact[i]
            other = con.geom2 if con.geom1 == floor else con.geom1
            touching.add(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, other))
        assert touching == set(BALL_CONTACTS), "seed %d touched %s" % (seed, touching)


def test_ball_contacts_can_be_turned_off_for_an_ablation():
    stock = MyoLocomotionEnv(stage="A", ball_contacts=False, seed=0)
    try:
        g = mujoco.mj_name2id(stock.model, mujoco.mjtObj.mjOBJ_GEOM, "heel_r")
        assert g < 0, "stock model should not have the contact balls"
    finally:
        stock.close()
