"""The captured initial state, and the asymmetry it exists to provide.

The symmetry argument is what these tests are really about. The model is
exactly left/right symmetric and so is a deterministic policy's response to a
symmetric observation, so an initial state whose two legs match cannot produce
a step -- both sides of the network see the same input and emit the same
output. Every assertion here that looks pedantic is guarding that.
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

from _myosuite_data import require_model  # noqa: E402

from myo_curriculum.env import FOOT_BODIES, MyoLocomotionEnv  # noqa: E402
from myo_curriculum.init_state import (  # noqa: E402
    JOINT_MAP,
    MUSCLE_MAP,
    load_init_state,
    resolve_init_state,
)
from myo_curriculum.stages import STAGE_W  # noqa: E402

PAIRS = ("hip_flexion", "knee_angle", "ankle_angle")


@pytest.fixture(scope="module")
def state():
    return load_init_state()


@pytest.fixture(scope="module")
def env():
    require_model()
    e = MyoLocomotionEnv(stage="W", seed=0)
    try:
        yield e
    finally:
        e.close()


# -- the file ------------------------------------------------------------

def test_the_file_parses_into_three_blocks(state):
    assert len(state.values) == 9
    assert len(state.velocities) == 9
    assert len(state.activations) == 18
    assert state.forward_velocity == pytest.approx(1.0757)
    assert state.pelvis_height == pytest.approx(0.900237)


def test_the_file_is_asymmetric_in_all_three_blocks(state):
    """Pose, velocity and activation each differ between the legs.

    One of the three would not be enough. Matching poses with differing
    velocities still start both legs in the same configuration, and matching
    velocities with differing poses have the legs converge.
    """
    for block in (state.values, state.velocities, state.activations):
        paired = [k for k in block if k.endswith("_r") and k[:-2] + "_l" in block]
        assert paired, "no paired quantities in this block"
        assert any(
            abs(block[k] - block[k[:-2] + "_l"]) > 1e-6 for k in paired
        ), "both legs identical in this block"


def test_the_asymmetry_is_large_not_incidental(state):
    """Big enough to drive a gait, not rounding.

    Measured: hip_flexion velocity differs by 4.70 rad/s, knee_angle by
    3.42 rad/s. The pose differs by 0.64 rad at the knee.
    """
    gaps = state.asymmetry()
    assert gaps["hip_flexion"] > 4.0
    assert gaps["knee_angle"] > 3.0
    assert gaps["vasti"] > 0.4


def test_knee_and_pelvis_tilt_are_the_only_negated_dofs(state):
    """The file is in OpenSim's convention; this model is in Rajagopal's."""
    negated = {dof for dof, (_, sign) in JOINT_MAP.items() if sign < 0}
    assert negated == {"pelvis_tilt", "knee_angle_r", "knee_angle_l"}

    # the file's knees are negative, and the model's range starts at zero
    assert state.values["knee_angle_r"] < 0.0
    assert state.values["knee_angle_l"] < 0.0
    positions = state.joint_positions()
    assert positions["knee_angle_r"] > 0.0
    assert positions["knee_angle_l"] > 0.0
    # negative tilt is a forward lean there, positive here
    assert state.values["pelvis_tilt"] < 0.0
    assert positions["pelvis_tilt"] > 0.0


def test_velocities_are_negated_wherever_positions_are(state):
    """A flipped coordinate flips its derivative.

    Getting only the angles right would start the model in the right pose
    moving the wrong way, which is far harder to spot than a wrong pose.
    """
    for dof, (joint, sign) in JOINT_MAP.items():
        if dof in state.velocities:
            assert state.joint_velocities()[joint] == pytest.approx(
                sign * state.velocities[dof]
            )
    # vertical velocity is up-positive in both, so it is not negated
    assert state.joint_velocities()["pelvis_ty"] == pytest.approx(
        state.velocities["pelvis_ty"]
    )


def test_every_lumped_muscle_expands(state):
    """Nine groups a side, onto the actuators that make them up."""
    expanded = state.muscle_activations()
    assert len(expanded) == 34
    for group, members in MUSCLE_MAP.items():
        for side in ("r", "l"):
            value = state.activations["%s_%s" % (group, side)]
            for stem in members:
                assert expanded["%s_%s" % (stem, side)] == pytest.approx(value)


def test_a_missing_file_says_so():
    with pytest.raises(FileNotFoundError):
        load_init_state(resolve_init_state("no_such_state.zml"))


# -- applied to the model ------------------------------------------------

def test_stage_w_uses_it(env):
    assert STAGE_W.init_state == "InitStateH0918Gait10ActA.zml"
    assert env.init_state is not None


def test_the_model_is_symmetric_so_the_state_must_not_be(env):
    """The premise, asserted rather than assumed.

    If the model were not mirror-symmetric, a symmetric start would still
    produce a gait and none of this would be needed.
    """
    model = env.model
    for pair in PAIRS:
        right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pair + "_r")
        left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pair + "_l")
        assert np.allclose(model.jnt_range[right], model.jnt_range[left])
    masses = {}
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name and name.endswith(("_r", "_l")):
            masses[name] = float(model.body_mass[body])
    mirrored = [(n, masses[n[:-2] + "_l"]) for n in masses if n.endswith("_r")
                and n[:-2] + "_l" in masses]
    assert mirrored
    for name, left_mass in mirrored:
        assert masses[name] == pytest.approx(left_mass), name


def test_reset_leaves_the_legs_asymmetric(env):
    """What the whole file is for: the two legs do not match at step 0."""
    env.reset(seed=0)
    for pair in PAIRS:
        right_q = env.data.qpos[env.qadr[pair + "_r"]]
        left_q = env.data.qpos[env.qadr[pair + "_l"]]
        right_v = env.data.qvel[env.dadr[pair + "_r"]]
        left_v = env.data.qvel[env.dadr[pair + "_l"]]
        assert abs(right_q - left_q) > 0.1, pair
        assert abs(right_v - left_v) > 0.1, pair

    activations = {n: env.data.act[i] for n, i in env.actuator_index.items()}
    assert abs(activations["vaslat_r"] - activations["vaslat_l"]) > 0.4
    assert abs(activations["iliacus_r"] - activations["iliacus_l"]) > 0.2


def test_reset_applies_pose_velocity_and_activation(env, state):
    env.reset(seed=0)
    for joint, value in state.joint_positions().items():
        if joint == "pelvis_ty":
            continue
        assert env.data.qpos[env.qadr[joint]] == pytest.approx(value, abs=1e-9)
    for joint, value in state.joint_velocities().items():
        assert env.data.qvel[env.dadr[joint]] == pytest.approx(value, abs=1e-9)
    assert env.pelvis_height == pytest.approx(state.pelvis_height, abs=1e-9)
    for name, value in state.muscle_activations().items():
        index = env.actuator_index[name]
        assert env.data.act[index] == pytest.approx(value)
        # ctrl matches act, so the first step holds the captured activation
        assert env.data.ctrl[index] == pytest.approx(value)


def test_the_unnamed_dofs_keep_the_standing_stance(env):
    """A 9-dof planar record says nothing about the trunk or the toes."""
    env.reset(seed=0)
    for joint in ("flex_extension", "mtp_angle_r", "mtp_angle_l"):
        assert env.data.qpos[env.qadr[joint]] == pytest.approx(
            env._neutral_qpos[env.qadr[joint]], abs=1e-9
        )


def test_the_captured_height_is_kept_not_seated(env, state):
    """Seating would overwrite part of the state.

    The captured height already puts the left toe on the floor, the right foot
    9-13 mm above it and the left heel 174 mm up -- left toe-off with the right
    foot about to land. Nothing is underground, so nothing is moved.
    """
    env.reset(seed=0)
    assert env.pelvis_height == pytest.approx(state.pelvis_height, abs=1e-9)
    lowest = min(min(env._heel_z(f), env._toe_z(f)) for f in FOOT_BODIES)
    assert lowest >= -1e-9, "nothing starts underground"
    assert lowest < 0.005, "the stance foot is on the ground, not hovering"
    # and the two feet are at different heights, which is the asymmetry again
    heights = [min(env._heel_z(f), env._toe_z(f)) for f in FOOT_BODIES]
    assert abs(heights[0] - heights[1]) > 1e-3


def test_it_carries_its_own_forward_velocity(env):
    """And the nominal `initial_forward_velocity` is not applied over it."""
    env.reset(seed=0)
    assert env.data.qvel[env.dadr["pelvis_tx"]] == pytest.approx(1.0757, abs=1e-9)
    assert STAGE_W.initial_forward_velocity != pytest.approx(1.0757)


def test_the_start_is_still_exactly_one_state(env):
    """Asymmetric, but not random: the same state every episode."""
    first, _ = env.reset(seed=0)
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()
    act = env.data.act.copy()
    for seed in (1, 2, 99):
        env.reset(seed=seed)
        assert np.array_equal(env.data.qpos, qpos)
        assert np.array_equal(env.data.qvel, qvel)
        assert np.array_equal(env.data.act, act)
    again, _ = env.reset(seed=12345)
    assert np.allclose(first, again)


def test_it_moves_forward_under_no_action(env):
    """The momentum is real, not just a number in qvel.

    From a standing start a zero-action rollout toppled backwards 0.28 m. From
    this state it covers about +0.42 m before falling, which is what carrying
    1.08 m/s into the first step looks like.
    """
    env.reset(seed=0)
    for _ in range(STAGE_W.episode_steps):
        _, _, terminated, truncated, _ = env.step(
            np.zeros(env.n_act, dtype=np.float32)
        )
        if terminated or truncated:
            break
    assert env.travel > 0.2, env.travel
