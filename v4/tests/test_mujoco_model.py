"""Tests for the .hfd -> MJCF translation and the MuJoCo backend.

Split in two. The converter tests are pure text processing and run anywhere,
like the rest of this suite. The backend tests need the `mujoco` wheel and skip
without it, so a machine set up only for the SCONE backend still runs green.

Nothing here needs sconegym, sconepy or a Hyfydy licence.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import _bootstrap

hfd_to_mjcf = _bootstrap.load_script("hfd_to_mjcf")

mujoco = pytest.importorskip("mujoco", reason="the mujoco backend needs the mujoco wheel")

REPO_ROOT = Path(__file__).resolve().parents[2]
HFD = hfd_to_mjcf.DEFAULT_HFD
MJCF = hfd_to_mjcf.DEFAULT_OUT
INIT_STATE = REPO_ROOT / "models" / "init_states" / "InitState_A0_walk_003_v2.zml"

# The .hfd's own dof declaration order. env.py indexes by name against this.
EXPECTED_DOFS = (
    "pelvis_tilt",
    "pelvis_tx",
    "pelvis_ty",
    "lumbar_extension",
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
    "mtp_angle_l",
    "arm_flex_r",
    "elbow_flex_r",
    "arm_flex_l",
    "elbow_flex_l",
)
EXPECTED_LOCKED = ("ankle_angle_r", "mtp_angle_r", "ankle_angle_l", "mtp_angle_l")
EXPECTED_ACTUATORS = hfd_to_mjcf.ACTUATOR_ORDER
CONTACT_GEOMS = (
    "heel_r",
    "toe_r",
    "heel_l",
    "toe_l",
    "crutch_tip_r",
    "crutch_tip_l",
)


@pytest.fixture(scope="module")
def hfd():
    return hfd_to_mjcf.HfdModel(hfd_to_mjcf.parse_hfd(HFD))


@pytest.fixture(scope="module")
def backend_module():
    return _bootstrap.load_module("mujoco_backend")


@pytest.fixture()
def model(backend_module):
    return backend_module.MujocoModel(MJCF, init_state_path=INIT_STATE)


# -- the converter ---------------------------------------------------------


def test_hfd_parses_into_the_expected_shape(hfd):
    assert len(hfd.bodies) == 17
    assert [d.name for d in hfd.dofs] == list(EXPECTED_DOFS)
    assert {g.name for g in hfd.geoms if g.type == "sphere"} == set(CONTACT_GEOMS)


def test_dof_sources_map_onto_joints(hfd):
    assert hfd.dof_by_name["hip_flexion_r"].joint_name() == "hip_r"
    assert hfd.dof_by_name["lumbar_extension"].joint_name() == "back"
    assert hfd.dof_by_name["elbow_flex_l"].joint_name() == "elbow_l"
    # The floating root has no joint block of its own.
    for name in ("pelvis_tx", "pelvis_ty", "pelvis_tilt"):
        assert hfd.dof_by_name[name].joint_name() is None


def test_only_ankle_and_mtp_are_locked_dofs(hfd):
    assert tuple(d.name for d in hfd.dofs if hfd.is_locked_dof(d)) == EXPECTED_LOCKED


def test_ranges_are_degrees_and_defaults_are_radians(hfd):
    # The .hfd mixes the two; getting it backwards would put the elbow's 0.80
    # default (46 degrees) at 0.8 degrees, or its 150-degree limit at 150 rad.
    elbow = hfd.dof_by_name["elbow_flex_r"]
    assert elbow.range == (0.0, 150.0)
    assert elbow.range_rad == pytest.approx((0.0, math.radians(150.0)))
    assert elbow.default == pytest.approx(0.80)

    pelvis_ty = hfd.dof_by_name["pelvis_ty"]
    assert pelvis_ty.translational
    assert pelvis_ty.range_rad == (-1.0, 2.0)  # metres, not converted
    assert pelvis_ty.default == pytest.approx(0.94)


def test_inertia_rebalance_preserves_the_largest_component(hfd):
    """The correction must not touch I_z, the only axis this planar model uses."""
    for body in hfd.bodies:
        before = tuple(body.inertia)
        after = body.balanced_inertia()
        a, b, c = sorted(after)
        assert a + b >= c - 1e-15, "%s still violates A+B>=C" % body.name
        assert max(after) == pytest.approx(max(before))
        if body.inertia_note is None:
            assert after == before
        else:
            # Only the two smaller components move, and only upward.
            assert all(x >= y - 1e-15 for x, y in zip(sorted(after), sorted(before)))


def test_exactly_the_two_known_bodies_need_rebalancing(hfd):
    needed = sorted(b.name for b in hfd.bodies if (b.balanced_inertia(), b.inertia_note)[1])
    assert needed == ["forearm_l", "forearm_r", "toes_l", "toes_r"]


def test_total_mass_matches_the_hfd(hfd):
    assert hfd.total_mass() == pytest.approx(64.338, abs=1e-3)


def test_checked_in_xml_is_what_the_converter_produces(hfd):
    """Guards against hand-editing the generated file, or letting it go stale."""
    regenerated, _, _ = hfd_to_mjcf.build_mjcf(hfd, HFD)
    on_disk = MJCF.read_text(encoding="utf-8")
    assert on_disk == regenerated, (
        "models/mjcf/%s is out of date with the .hfd.\n"
        "Regenerate it: python v4/scripts/hfd_to_mjcf.py" % MJCF.name
    )


# -- the MJCF as MuJoCo sees it -------------------------------------------


def test_mjcf_compiles_with_the_expected_counts():
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    assert m.nu == len(EXPECTED_ACTUATORS)
    assert m.nq == m.nv == len(EXPECTED_DOFS) - len(EXPECTED_LOCKED)
    # The .hfd declares 17 bodies, one of which is the massless `ground`; that
    # becomes MuJoCo's world rather than a body of its own.
    assert m.nbody == 17
    assert float(np.sum(m.body_mass)) == pytest.approx(64.338, abs=1e-3)


def test_world_is_y_up_not_z_up():
    """The whole translation depends on this; a Z-up flip would invert signs."""
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    assert np.allclose(m.opt.gravity, [0.0, -9.81, 0.0])
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    normal = np.asarray(m.geom_quat[floor])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # The plane's local +Z (its normal) must point along world +Y.
    rot = np.asarray(d.geom_xmat[floor]).reshape(3, 3)
    assert np.allclose(rot[:, 2], [0.0, 1.0, 0.0], atol=1e-9)
    assert normal is not None


def test_locked_joints_are_absent_from_the_mjcf():
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    names = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)
    }
    assert names.isdisjoint(EXPECTED_LOCKED)
    assert names == set(EXPECTED_DOFS) - set(EXPECTED_LOCKED)


def test_contact_spheres_collide_only_with_the_floor():
    """Feet and crutch tips must never collide with each other."""
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for name in CONTACT_GEOMS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert g >= 0, name
        # A pair collides when one's contype shares a bit with the other's
        # conaffinity, in either direction.
        assert (m.geom_contype[g] & m.geom_conaffinity[floor]) or (
            m.geom_contype[floor] & m.geom_conaffinity[g]
        )
        for other in CONTACT_GEOMS:
            if other == name:
                continue
            o = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, other)
            assert not (m.geom_contype[g] & m.geom_conaffinity[o])
            assert not (m.geom_contype[o] & m.geom_conaffinity[g])


def test_visual_geoms_cannot_collide():
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    contact = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in CONTACT_GEOMS}
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for g in range(m.ngeom):
        if g in contact or g == floor:
            continue
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


# -- the backend -----------------------------------------------------------


def test_dof_table_is_the_hfd_order_including_locked(model):
    assert tuple(d.name() for d in model.dofs()) == EXPECTED_DOFS
    assert tuple(model.locked_dofs()) == EXPECTED_LOCKED


def test_actuator_order_matches_the_env_contract(model):
    assert tuple(a.name() for a in model.actuators()) == EXPECTED_ACTUATORS


def test_model_is_torque_only(model):
    assert model.muscles() == []


def test_dof_positions_round_trip(model):
    q = np.linspace(-0.2, 0.2, len(EXPECTED_DOFS))
    for name in EXPECTED_LOCKED:
        q[EXPECTED_DOFS.index(name)] = 0.0
    model.set_dof_positions(q)
    assert np.allclose(model.dof_position_array(), q)


def test_locked_dofs_read_zero_and_reject_writes(model):
    q = model.dof_position_array()
    q[EXPECTED_DOFS.index("ankle_angle_r")] = 0.3
    with pytest.warns(RuntimeWarning, match="locked dof"):
        model.set_dof_positions(q)
    assert model.dof_position_array()[EXPECTED_DOFS.index("ankle_angle_r")] == 0.0


def test_init_state_comes_from_the_zml_not_the_hfd_defaults(model):
    """The .scone model names this file, so both backends start from one pose."""
    index = {n: i for i, n in enumerate(EXPECTED_DOFS)}
    assert model.init_q[index["pelvis_ty"]] == pytest.approx(0.94)
    assert model.init_q[index["pelvis_tilt"]] == pytest.approx(-0.03)
    assert model.init_q[index["hip_flexion_r"]] == pytest.approx(0.15)
    assert model.init_q[index["knee_angle_l"]] == pytest.approx(-0.15)
    # The .hfd defaults would be arm_flex 0.30 / elbow_flex 0.80 here.
    assert model.init_q[index["arm_flex_r"]] == pytest.approx(0.0)
    assert model.init_q[index["elbow_flex_r"]] == pytest.approx(0.30)
    assert model.init_dq[index["pelvis_tx"]] == pytest.approx(0.02)


def test_unknown_dof_in_an_init_state_raises(model, tmp_path):
    bad = tmp_path / "bad.zml"
    bad.write_text("values {\n  not_a_dof = 1.0\n}\n", encoding="utf-8")
    with pytest.raises(KeyError, match="not_a_dof"):
        model.load_init_state(bad)


def test_body_lookup(model):
    for name in ("pelvis", "torso", "calcn_r", "calcn_l", "Crutch_r", "Crutch_l"):
        body = model.find_body(name)
        assert body is not None and body.name() == name
    assert model.find_body("no_such_body") is None


def test_contact_force_is_upward_and_carries_body_weight(model):
    """Signs and magnitude together: a standing model pushes up by its weight."""
    model.adjust_state_for_load(1.0)
    feet = sum(model.find_body(b).contact_force().y for b in ("calcn_r", "calcn_l"))
    assert feet > 0.0, "ground reaction must push the feet up, not down"
    total = model.total_vertical_contact_force()
    assert total == pytest.approx(model.body_weight, rel=0.05)


def test_adjust_state_for_load_hits_reachable_targets(model):
    """0.5 is what every shipped stage uses, so it must be hit accurately."""
    for fraction in (0.5, 0.75, 1.0):
        model.reset()
        achieved = model.adjust_state_for_load(fraction)
        assert achieved == pytest.approx(fraction, abs=0.02)


def test_adjust_state_for_load_says_so_when_a_target_is_unreachable(model):
    """MuJoCo's contact has a nonzero force at onset, so low targets can't be met.

    The model must still end up touching the floor rather than hovering, and
    the caller must be told rather than silently given a different load.
    """
    model.reset()
    with pytest.warns(RuntimeWarning, match="below what MuJoCo's contact solver"):
        achieved = model.adjust_state_for_load(0.05)
    assert achieved > 0.0, "the model should be in contact, not hovering"
    assert model.total_vertical_contact_force() > 0.0
    # Warns once per model, not once per reset.
    model.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.adjust_state_for_load(0.05)


def test_adjust_state_for_load_leaves_joint_angles_alone(model):
    """It may only move the pelvis, so an RSI pose survives the settling."""
    before = model.dof_position_array()
    model.adjust_state_for_load(0.5)
    after = model.dof_position_array()
    moved = [
        name
        for name, a, b in zip(EXPECTED_DOFS, before, after)
        if abs(a - b) > 1e-9
    ]
    assert moved == ["pelvis_ty"]


def test_the_model_stays_in_the_sagittal_plane(model):
    """Every joint is a z-hinge or an x/y slide, so z must never change."""
    rng = np.random.RandomState(0)
    z_before = np.asarray([b.com_pos().z for b in model.bodies()])
    for _ in range(50):
        model.set_actuator_inputs(rng.uniform(-60, 60, len(model.actuators())))
        model.advance_simulation_to(model.time + 0.01)
    z_after = np.asarray([b.com_pos().z for b in model.bodies()])
    assert np.allclose(z_before, z_after, atol=1e-9)


def test_advance_simulation_reaches_the_requested_time(model):
    model.advance_simulation_to(0.25)
    assert model.time == pytest.approx(0.25, abs=model.m.opt.timestep)


def test_com_velocity_tracks_a_ballistic_drop(backend_module):
    """A free fall gives an analytic answer, so this pins units and sign."""
    model = backend_module.MujocoModel(MJCF, init_state_path=INIT_STATE)
    q = model.dof_position_array()
    q[EXPECTED_DOFS.index("pelvis_ty")] = 3.0  # clear of the floor
    model.set_dof_positions(q)
    model.set_dof_velocities(np.zeros(len(EXPECTED_DOFS)))
    model.init_state_from_dofs()
    model.advance_simulation_to(0.2)
    assert model.com_vel().y == pytest.approx(-9.81 * 0.2, rel=0.02)
    assert abs(model.com_vel().x) < 1e-6


def test_zero_torque_collapse_matches_the_hyfydy_measurement(model):
    """Cross-check against the numbers the v4 README records for Hyfydy.

    The README's calibration section reports that, from the neutral pose under
    zero torque, COM height runs 0.922 -> 0.858 over 36 steps and the model
    falls (COM below 0.55) at step 73. This is the one measurement of the
    original simulator available without a licence, so it is worth pinning:
    it would catch a wrong mass, a wrong gravity axis or a badly scaled
    contact model, all of which would change the collapse rate.

    Tolerances are loose on purpose. Hyfydy and MuJoCo resolve contact
    differently and the two will never agree to more than a few percent.
    """
    heights = [model.com_pos().y]
    fell_at = None
    for step in range(1, 121):
        model.set_actuator_inputs(np.zeros(len(model.actuators())))
        model.advance_simulation_to(model.time + 0.01)
        heights.append(model.com_pos().y)
        if fell_at is None and heights[-1] < 0.55:
            fell_at = step

    assert heights[0] == pytest.approx(0.922, abs=0.03)
    assert heights[36] == pytest.approx(0.858, abs=0.05)
    assert fell_at is not None, "the model should collapse without any torque"
    assert 55 <= fell_at <= 95, "fell at step %s, Hyfydy falls at 73" % fell_at


def test_write_results_emits_a_readable_sto(model, tmp_path):
    trajectory = _bootstrap.load_trajectory()
    model.set_store_data(True)
    model.advance_simulation_to(0.05)
    path = model.write_results(tmp_path, "rollout")
    assert path is not None and path.is_file()
    loaded = trajectory.load_sto(path, list(EXPECTED_DOFS))
    assert loaded.n_frames > 0


def test_write_results_without_recording_returns_none(model, tmp_path):
    model.set_store_data(False)
    model.advance_simulation_to(0.05)
    assert model.write_results(tmp_path, "rollout") is None


# -- the gym base ----------------------------------------------------------


def test_observation_layout_is_declared_and_pinned():
    pytest.importorskip("gym")
    from sconegym_crutch_v4.mujoco_gym import OBS_LAYOUT, OBS_SIZE

    assert OBS_SIZE == sum(n for _, n in OBS_LAYOUT)
    assert OBS_SIZE == 30
    assert [name for name, _ in OBS_LAYOUT] == [
        "pelvis_height",
        "pelvis_tilt_sin_cos",
        "actuated_q",
        "root_dq",
        "actuated_dq",
        "com_vel_xy",
        "contact_load",
    ]


def test_backend_selection_rejects_an_unknown_name():
    pytest.importorskip("gym")
    from sconegym_crutch_v4 import backends

    with pytest.raises(ValueError, match="must be one of"):
        backends.selected("physx")


def test_backend_selection_honours_an_explicit_choice():
    pytest.importorskip("gym")
    from sconegym_crutch_v4 import backends

    assert backends.selected("mujoco") == "mujoco"
    assert backends.selected("scone") == "scone"
