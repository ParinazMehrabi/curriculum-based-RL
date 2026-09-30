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

import mujoco
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
    OUT_OF_PLANE_JOINTS,
    ROOT_JOINTS,
    MyoLocomotionEnv,
)
from myo_curriculum.reference import TRACKED_JOINTS  # noqa: E402
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

# -- planar structure ------------------------------------------------------


def test_model_is_left_right_symmetric():
    """Pins a correction: this model is exactly symmetric.

    An earlier version of this environment reported the right femur as 23.5 mm
    shorter than the left and built an asymmetric stance around it. That
    measurement was wrong: it set knee_angle to -0.05, which is outside the
    joint's [0, 2.0944] range, so the knee's coupling polynomials -- which
    drive translation1/2 and rotation2/3 -- were evaluated off their domain and
    returned different nonsense per side. With the joints inside their ranges
    the two legs match to floating-point precision, which is what lets the
    stance be solved symmetrically.
    """
    env = MyoLocomotionEnv(stage="A", seed=0)
    try:
        mujoco.mj_resetData(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        pos = lambda n: np.asarray(
            env.data.xipos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)]
        )
        for seg in ("femur", "tibia", "talus", "calcn", "toes", "patella"):
            right, left = pos(seg + "_r"), pos(seg + "_l")
            assert right[0] == pytest.approx(left[0], abs=1e-9), seg  # fore-aft
            assert right[2] == pytest.approx(left[2], abs=1e-9), seg  # height
    finally:
        env.close()


def test_root_is_planar(env):
    """Three joints, not a freejoint: slide x, slide z, hinge y."""
    names = [
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(env.model.njnt)
    ]
    assert not any(
        env.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        for i in range(env.model.njnt)
    ), "the free root should have been replaced"
    for name in ROOT_JOINTS:
        assert name in names
    expected = {
        "pelvis_tx": (mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0]),
        "pelvis_ty": (mujoco.mjtJoint.mjJNT_SLIDE, [0, 0, 1]),
        "pelvis_tilt": (mujoco.mjtJoint.mjJNT_HINGE, [0, 1, 0]),
    }
    for name, (jtype, axis) in expected.items():
        j = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert env.model.jnt_type[j] == jtype, name
        assert np.allclose(env.model.jnt_axis[j], axis), name


def test_twelve_independent_degrees_of_freedom(env):
    assert len(ROOT_JOINTS) == 3
    assert len(INDEPENDENT_JOINTS) == 9
    assert len(env.joint_qpos_adr) == 9


def test_out_of_plane_joints_are_pinned(env):
    """Equality constraints, one per joint, holding it at zero."""
    pinned = set()
    for i in range(env.model.neq):
        if env.model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        name = mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, env.model.eq_obj1id[i]
        )
        if name in OUT_OF_PLANE_JOINTS:
            pinned.add(name)
    assert pinned == set(OUT_OF_PLANE_JOINTS)

    env.reset(seed=0)
    for name in OUT_OF_PLANE_JOINTS:
        j = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert float(env.data.qpos[env.model.jnt_qposadr[j]]) == pytest.approx(
            0.0, abs=1e-6
        )


def test_the_model_stays_in_the_sagittal_plane(env):
    """The property the whole planar restructuring exists to provide."""
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    before = np.array([env.data.xipos[b][1] for b in range(1, env.model.nbody)])
    for _ in range(60):
        env.step(rng.uniform(-1, 1, env.n_act).astype(np.float32))
    after = np.array([env.data.xipos[b][1] for b in range(1, env.model.nbody)])
    # Bounded, not zero, and the distinction is the point. The root cannot
    # translate sideways, yaw or roll, so the *body* cannot leave the plane or
    # fall out of it. Individual segments still wander a little, because the
    # ankle, knee and mtp axes are anatomically oblique and were left that way
    # -- 2.7 cm at the worst segment under random full-range activation, far
    # less under anything resembling a policy. Projecting those axes onto the
    # sagittal plane would make it exactly planar at the cost of the anatomy.
    assert np.abs(after - before).max() < 0.05


def test_forward_is_simply_plus_x(walk_env):
    """No heading to track once the root cannot yaw."""
    walk_env.reset(seed=0)
    assert walk_env.forward_velocity() > 0.15
    # Not identically zero: the oblique ankle axis leaks a few microns per
    # second sideways. It cannot grow, because the root has no lateral dof.
    assert abs(float(walk_env.com_velocity()[1])) < 1e-3


def test_stance_is_parallel(env):
    for seed in range(8):
        env.reset(seed=seed)
        assert abs(env.foot_stagger()) < 0.02


def test_reset_is_roughly_supported_by_the_feet(env):
    """Total ground reaction at the solved stance.

    Below one body weight because the muscles are barely activated and the
    joints are already giving way; the point is that the feet carry the model
    rather than that it is in perfect equilibrium.
    """
    for seed in range(8):
        env.reset(seed=seed)
        assert 0.25 < env.contact_loads().sum() < 1.6


def test_a_model_without_a_free_root_is_refused(tmp_path):
    """The planar rebuild needs a free root to replace."""
    xml = tmp_path / "no_free_root.xml"
    xml.write_text(
        '<mujoco><worldbody><body name="pelvis">'
        '<joint name="j" type="hinge" axis="0 0 1"/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="free root joint"):
        MyoLocomotionEnv(stage="A", model_path=xml)


def test_contact_balls_are_massless(env):
    """A contact primitive must not change the segment's inertia.

    Left at MuJoCo's default density the four spheres added 0.3 kg. MyoSuite's
    stock mass is 82.038 kg.
    """
    assert env.body_weight / 9.81 == pytest.approx(82.038, abs=0.02)



# -- reference tracking (stage W) ------------------------------------------


@pytest.fixture(scope="module")
def track_env():
    e = MyoLocomotionEnv(stage="W", seed=0)
    yield e
    e.close()


def test_reference_loads_and_maps_onto_this_model(track_env):
    """Hip and knee, both sides, over one phase-averaged cycle."""
    ref = track_env.reference
    assert ref is not None
    assert TRACKED_JOINTS == (
        "hip_flexion_r", "knee_angle_r", "hip_flexion_l", "knee_angle_l",
    )
    assert ref.n_frames == 100
    assert ref.joints.shape == (100, 4)
    assert ref.duration == pytest.approx(2.7596, abs=1e-3)
    assert ref.condition == "transparent_WALKING"


def test_the_reference_holds_no_pelvis_height(track_env):
    """The record has no height channel, and `pose_at` says so with None.

    Silently returning a number here -- the previous .sto reference's absolute
    pelvis height -- would place the model vertically from data that does not
    exist.
    """
    joints, height = track_env.reference.pose_at(0.3)
    assert height is None
    assert joints.shape == (len(TRACKED_JOINTS),)


def test_the_reference_is_in_this_model_s_joint_ranges(track_env):
    """Knee flexion is positive here and negative in the source log."""
    model = track_env.model
    for i in range(100):
        joints, _ = track_env.reference.pose_at(i / 100.0)
        for name, value in zip(TRACKED_JOINTS, joints):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = model.jnt_range[jid]
            assert lo <= value <= hi, (name, value, lo, hi)
        knees = [v for n, v in zip(TRACKED_JOINTS, joints) if n.startswith("knee")]
        assert all(v > 0.0 for v in knees), "knee flexion is positive in MyoSuite"


def test_positive_pelvis_tilt_leans_forward():
    """The trunk convention, kept as a measurement even though the reference
    no longer prescribes pelvis tilt: the upright term still reads it."""
    env = MyoLocomotionEnv(stage="A", seed=0)
    try:
        def trunk_x(value):
            env.reset(seed=0)
            env.data.qpos[env.qadr["pelvis_tilt"]] = value
            mujoco.mj_forward(env.model, env.data)
            return float(env.trunk_axis()[0])

        assert trunk_x(+0.30) > trunk_x(-0.30), "positive tilt should lean forward"
    finally:
        env.close()


def test_rsi_stands_the_posed_frame_on_the_floor(track_env):
    """The reference gives no height, so reset has to find one.

    Keeping the standing pelvis height instead leaves a flexed-knee frame
    hanging: measured, the centre of mass reached 1.258 m against 1.012 m
    standing, over the 1.10 m that counts as flying. Seating the lowest contact
    point puts every frame on the ground at its own height.
    """
    heights = []
    for seed in range(12):
        track_env.reset(seed=seed)
        lowest = min(
            min(track_env._heel_z(foot), track_env._toe_z(foot))
            for foot in ("calcn_r", "calcn_l")
        )
        assert lowest == pytest.approx(0.0, abs=1e-9), "the lowest point touches"
        assert track_env.com_height < track_env.stage_spec.fly_threshold
        heights.append(track_env.pelvis_height)
    # the phase changes the height, so this is not the standing pose repeated
    assert max(heights) - min(heights) > 1e-3


def test_rsi_starts_on_the_reference_pose(track_env):
    """Only reset noise should separate the model from the reference."""
    for seed in range(8):
        track_env.reset(seed=seed)
        assert track_env.tracking_error < 0.10


def test_rsi_does_not_seat_the_feet(track_env):
    """Forcing a mid-swing frame plantigrade would corrupt it."""
    off_ground = 0
    for seed in range(12):
        track_env.reset(seed=seed)
        if max(track_env._heel_z("calcn_r"), track_env._heel_z("calcn_l")) > 1e-3:
            off_ground += 1
    assert off_ground > 0, "a gait cycle should sometimes have a foot in the air"


def test_reference_clock_advances_with_sim_time(track_env):
    track_env.reset(seed=0)
    start = track_env.ref_phase
    for _ in range(50):
        track_env.step(np.zeros(track_env.n_act, np.float32))
    advanced = (track_env.ref_phase - start) % 1.0
    assert advanced == pytest.approx(50 * track_env.dt / track_env.reference.duration)


def test_reference_phase_wraps(track_env):
    ref = track_env.reference
    assert ref.advance(0.99, ref.duration * 0.02) == pytest.approx(0.01, abs=1e-9)
    a, _ = ref.pose_at(0.0)
    b, _ = ref.pose_at(1.0)
    assert np.allclose(a, b)


def test_stage_w_maxima_are_the_ones_asked_for():
    """alive 0.1, tracking 0.2 per step; forward 0.7 paid at the end.

    Over a full episode that is 100 + 200 + 700 = 1000, the same proportions
    as a 1.0/step reward with the forward share moved to the end.
    """
    stage = STAGES["W"]
    spec = stage.reward
    total_w = sum(spec.active_weights.values())
    caps = {
        name: spec.shaping_scale * w / total_w
        for name, w in spec.active_weights.items()
    }
    assert spec.alive == pytest.approx(0.10)
    assert caps["tracking"] == pytest.approx(0.20)
    assert spec.alive + sum(caps.values()) == pytest.approx(0.30)

    terminal = stage.forward_bonus * stage.episode_steps
    assert terminal == pytest.approx(700.0)
    episode = 0.30 * stage.episode_steps + terminal
    assert episode == pytest.approx(1000.0)


def test_stage_w_is_additive_not_geometric():
    """Geometric composition would gate the alive bonus on early tracking."""
    spec = STAGES["W"].reward
    assert spec.composition == "additive"
    total, _ = spec.compose({"tracking": 0.0})
    assert total >= spec.alive, "surviving should score even with no tracking"


def test_forward_progress_is_paid_at_the_end_not_per_step(track_env):
    """Per step it is farmable by falling; terminally it is not."""
    assert "velocity" not in track_env.stage_spec.reward.weights
    track_env.reset(seed=1)
    paid = []
    for _ in range(1000):
        _, reward, term, trunc, _ = track_env.step(
            np.full(track_env.n_act, -0.6, np.float32)
        )
        paid.append(track_env.terminal_bonus)
        if term or trunc:
            break
    assert all(p == 0.0 for p in paid[:-1]), "nothing terminal before the end"
    assert paid[-1] == pytest.approx(track_env.forward_bonus())


def test_falling_collects_almost_none_of_the_forward_bonus(track_env):
    """The whole point of moving it to the end.

    Per step, any topple produces v > 0.1 m/s and saturated the term at 100%.
    Against distance it cannot: a fall ends the episode after a few
    centimetres.
    """
    track_env.reset(seed=1)
    for _ in range(1000):
        _, _, term, trunc, _ = track_env.step(
            np.full(track_env.n_act, -0.6, np.float32)
        )
        if term or trunc:
            break
    assert term, "this policy should fall"
    cap = track_env.stage_spec.forward_bonus * track_env.stage_spec.episode_steps
    assert track_env.terminal_bonus < 0.02 * cap


def test_forward_bonus_is_linear_and_open_ended(track_env):
    """Zero at a standstill, linear in distance, with no ceiling.

    The reference distance is a scale, not a cap: covering twice it pays twice
    as much, and there is no distance at which walking further stops paying.
    Survival is held at a full episode so this isolates the distance factor;
    `test_forward_bonus_is_monotone_in_survival` covers the other one.
    """
    stage = track_env.stage_spec
    full = stage.forward_bonus * stage.episode_steps
    track_env.reset(seed=0)
    base = track_env._start_tx
    track_env.steps = stage.episode_steps

    seen = {}
    for d in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0):
        track_env.data.qpos[track_env.qadr["pelvis_tx"]] = base + d
        seen[d] = track_env.forward_bonus()

    assert seen[0.0] == pytest.approx(0.0)
    scale = stage.forward_reference_distance
    for d, value in seen.items():
        assert value == pytest.approx(full * d / scale), d
    assert seen[2.0] == pytest.approx(2 * seen[1.0]), "twice as far pays twice"
    assert seen[5.0] > seen[2.0] > seen[1.0], "further is always worth more"


def test_walking_backwards_costs_what_walking_forwards_earns(track_env):
    """The forward term is signed, so retreating is not merely unrewarded."""
    stage = track_env.stage_spec
    track_env.reset(seed=0)
    base = track_env._start_tx
    track_env.steps = stage.episode_steps

    track_env.data.qpos[track_env.qadr["pelvis_tx"]] = base + 0.4
    forwards = track_env.forward_bonus()
    track_env.data.qpos[track_env.qadr["pelvis_tx"]] = base - 0.4
    backwards = track_env.forward_bonus()
    assert backwards == pytest.approx(-forwards)


def test_travel_is_the_root_not_the_com(track_env):
    """Root translation cannot be inflated by swinging the limbs forward.

    Measured on a toppling rollout the two disagree in sign: pelvis_tx goes
    -0.032 m while the whole-body COM goes +0.028 m, because the legs swing
    forward as the model falls.
    """
    track_env.reset(seed=0)
    base = track_env._start_tx
    track_env.data.qpos[track_env.qadr["pelvis_tx"]] = base + 0.37
    assert track_env.travel == pytest.approx(0.37)


def test_velocity_term_is_a_smoothstep_that_saturates(track_env):
    """A hard threshold at 0.1 m/s would have no gradient from a standstill."""
    track_env.reset(seed=0)
    seen = []
    for v in (0.0, 0.025, 0.05, 0.075, 0.10, 0.30):
        # Zero every other dof: forward_velocity reads the COM, so leftover
        # joint velocity from reset noise would shift it off the root's.
        track_env.data.qvel[:] = 0.0
        track_env.data.qvel[track_env.dadr["pelvis_tx"]] = v
        mujoco.mj_forward(track_env.model, track_env.data)
        seen.append(track_env._term_velocity())
    assert seen[0] == pytest.approx(0.0, abs=1e-6)
    assert all(b >= a for a, b in zip(seen, seen[1:])), "must be monotone"
    assert 0.0 < seen[1] < seen[2] < seen[3] < 1.0, "gradient below the gate"
    assert seen[4] == pytest.approx(1.0)
    assert seen[5] == pytest.approx(1.0), "saturates above the gate"


def test_tracking_term_is_one_on_the_reference_and_falls_off(track_env):
    track_env.reset(seed=0)
    track_env.apply_reference_pose(track_env.ref_phase)
    track_env.tracking_error = track_env.reference_error()
    on_ref = track_env._term_tracking()
    assert on_ref > 0.99

    track_env.data.qpos[track_env.qadr["hip_flexion_r"]] += 0.6
    mujoco.mj_forward(track_env.model, track_env.data)
    track_env.tracking_error = track_env.reference_error()
    assert track_env._term_tracking() < on_ref


def test_early_termination_on_tracking_error(track_env):
    """Otherwise the policy banks alive and velocity return from a desynced state."""
    track_env.reset(seed=0)
    assert not track_env._is_fallen()
    track_env.tracking_error = track_env.stage_spec.max_tracking_error + 0.01
    assert track_env._is_fallen()


def test_stage_w_reward_is_never_negative_while_alive(track_env):
    track_env.reset(seed=0)
    for _ in range(200):
        _, reward, term, trunc, _ = track_env.step(
            np.full(track_env.n_act, -0.6, np.float32)
        )
        if term:
            break
        assert reward >= 0.0
        if trunc:
            break


def test_forward_bonus_requires_surviving_not_just_travelling(track_env):
    """Regression: the policy learned to dive.

    At iteration 180 it accelerated to 1.67 m/s -- eleven times the
    reference's speed -- covered 0.57 m in 74 steps and fell, collecting 425
    of 700, which was 98.7% of its return. Distance alone rewards a lunge
    exactly as well as a walk. Multiplying by the fraction of the episode
    survived makes survival a multiplier rather than a bonus.
    """
    stage = track_env.stage_spec
    assert stage.forward_requires_survival
    full = stage.forward_bonus * stage.episode_steps
    distance_only = full * 0.572 / stage.forward_reference_distance

    track_env.reset(seed=0)
    base = track_env._start_tx
    track_env.data.qpos[track_env.qadr["pelvis_tx"]] = base + 0.572
    track_env.steps = 74                      # the dive
    dive = track_env.forward_bonus()
    track_env.steps = stage.episode_steps     # same distance, full episode
    survived = track_env.forward_bonus()

    assert dive < 0.1 * distance_only, "a dive should not collect the distance"
    assert survived == pytest.approx(distance_only)
    assert survived > 10 * dive


def test_flying_is_penalised_above_the_threshold(track_env):
    """The only cap on an open-ended distance reward is on height.

    Leaving the ground is otherwise the cheapest way to cover ground, so
    centre-of-mass height above `fly_threshold` costs per step, per metre.
    """
    stage = track_env.stage_spec
    assert stage.fly_penalty > 0.0
    track_env.reset(seed=0)

    assert track_env.com_height < stage.fly_threshold, "standing is not flying"
    assert track_env.fly_penalty() == 0.0

    # lift the whole model by raising the pelvis slide
    adr = track_env.qadr["pelvis_ty"]
    ground = track_env.data.qpos[adr]
    seen = []
    for lift in (0.0, 0.05, 0.10, 0.30):
        track_env.data.qpos[adr] = ground + lift
        mujoco.mj_forward(track_env.model, track_env.data)
        seen.append(track_env.fly_penalty())

    assert seen[0] == 0.0, "the standing pose is below the threshold"
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert seen[-1] > 0.0, "a model 30 cm in the air is flying"
    # charged per metre of excess: the penalty is linear in how high it is
    excess = track_env.com_height - stage.fly_threshold
    assert seen[-1] == pytest.approx(stage.fly_penalty * excess)


def test_flying_costs_more_than_a_step_is_worth(track_env):
    """Otherwise height can be held profitably and the cap does not bind."""
    stage = track_env.stage_spec
    per_step = stage.reward.alive + stage.reward.shaping_scale
    # 10 cm above the threshold, which is the scale of a hop
    assert stage.fly_penalty * 0.10 > per_step


def test_the_reference_gait_never_trips_the_fly_penalty(track_env):
    """No frame of the gait, standing on the floor, counts as flying.

    Seated the way reset seats it: `apply_reference_pose` sets hip and knee
    only, and the record carries no height, so the pose has to be put on the
    ground before its centre-of-mass height means anything. Measured across the
    cycle, seated: 0.974 m at the highest, against a 1.10 m threshold.
    """
    highest = 0.0
    for i in range(100):
        track_env.apply_reference_pose(i / 100.0)
        track_env.seat_lowest_contact()
        highest = max(highest, track_env.com_height)
        assert track_env.fly_penalty() == 0.0, i
    assert highest < track_env.stage_spec.fly_threshold
    assert highest == pytest.approx(0.974, abs=0.01)


def test_fly_penalty_is_subtracted_from_the_step_reward(track_env):
    """It reaches the reward itself, and is reported under its own name."""
    track_env.reset(seed=0)
    grounded = track_env._get_reward()
    assert track_env.rwd_dict["fly"] == 0.0
    assert track_env.fly_cost == 0.0

    adr = track_env.qadr["pelvis_ty"]
    track_env.data.qpos[adr] += 0.40
    mujoco.mj_forward(track_env.model, track_env.data)
    airborne = track_env._get_reward()

    assert track_env.fly_cost > 0.0
    assert track_env.rwd_dict["fly"] == pytest.approx(-track_env.fly_cost)
    # the same pose without the penalty would score the composed terms alone
    assert airborne == pytest.approx(
        track_env.stage_spec.reward.compose(track_env.term_values)[0]
        - track_env.fly_cost
    )
    assert airborne < grounded


def test_forward_bonus_is_monotone_in_survival(track_env):
    track_env.reset(seed=0)
    base = track_env._start_tx
    track_env.data.qpos[track_env.qadr["pelvis_tx"]] = base + 0.5
    seen = []
    for steps in (50, 200, 500, 1000):
        track_env.steps = steps
        seen.append(track_env.forward_bonus())
    assert all(b > a for a, b in zip(seen, seen[1:]))


def test_forward_bonus_survival_factor_saturates_at_the_episode_length(track_env):
    track_env.reset(seed=0)
    base = track_env._start_tx
    track_env.data.qpos[track_env.qadr["pelvis_tx"]] = base + 2.0
    track_env.steps = track_env.stage_spec.episode_steps
    full = track_env.forward_bonus()
    track_env.steps = track_env.stage_spec.episode_steps * 3
    assert track_env.forward_bonus() == pytest.approx(full)


def test_tracking_threshold_is_a_backstop_not_the_main_terminator(track_env):
    """Measured: at 0.80 it ended 13 of 20 episodes at a mean of 32 steps.

    At 1.5 it never fires and every episode ends on trunk tilt at 39; above
    1.5 it stops binding at all. So 1.5 is where it becomes a guard against
    drifting off the reference while upright, rather than the thing that ends
    most episodes.
    """
    assert track_env.stage_spec.max_tracking_error == pytest.approx(1.50)
