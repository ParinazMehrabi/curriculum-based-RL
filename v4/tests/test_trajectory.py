"""Tests for reference trajectory loading and RSI configuration.

Runs against the real Moco solution in models/reference/ when it is present,
and against synthetic .sto files otherwise, so the parser is covered either way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _bootstrap import PKG_DIR, load, load_trajectory

rewards, stages = load()
trajectory = load_trajectory()

REFERENCE = PKG_DIR.parents[1] / "models" / "reference" / "gaitTracking_solution_raw.sto"

MODEL_DOFS = (
    "pelvis_tilt",
    "pelvis_tx",
    "pelvis_ty",
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
    "mtp_angle_l",
    "lumbar_extension",
    "arm_flex_r",
    "elbow_flex_r",
    "arm_flex_l",
    "elbow_flex_l",
)


def _write_sto(path: Path, columns, rows) -> Path:
    lines = ["someheader=1", "endheader", "\t".join(columns)]
    lines += ["\t".join("%.10g" % v for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- parser -----------------------------------------------------------------


def test_moco_column_convention(tmp_path):
    cols = ["time", "/jointset/back/lumbar_extension/value", "/jointset/back/lumbar_extension/speed"]
    rows = [[0.0, 0.1, 1.0], [0.1, 0.2, 2.0], [0.2, 0.3, 3.0]]
    p = _write_sto(tmp_path / "a.sto", cols, rows)
    traj = trajectory.load_sto(p, ["lumbar_extension"])
    assert traj.convention.startswith("moco")
    assert traj.n_frames == 3
    assert traj.q[:, 0] == pytest.approx([0.1, 0.2, 0.3])
    assert traj.dq[:, 0] == pytest.approx([1.0, 2.0, 3.0])


def test_bare_column_convention(tmp_path):
    p = _write_sto(
        tmp_path / "b.sto",
        ["time", "pelvis_tilt", "pelvis_tilt_u"],
        [[0.0, 0.5, -1.0], [0.1, 0.6, -2.0]],
    )
    traj = trajectory.load_sto(p, ["pelvis_tilt"])
    assert traj.convention.startswith("bare")
    assert traj.dq[:, 0] == pytest.approx([-1.0, -2.0])


def test_velocities_are_differenced_when_absent(tmp_path):
    p = _write_sto(tmp_path / "c.sto", ["time", "pelvis_ty"], [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    traj = trajectory.load_sto(p, ["pelvis_ty"])
    assert "differenced" in traj.convention
    assert traj.dq[:, 0] == pytest.approx([1.0, 1.0, 1.0])


def test_missing_dof_raises_with_available_names(tmp_path):
    p = _write_sto(
        tmp_path / "d.sto",
        ["time", "/jointset/back/lumbar_extension/value"],
        [[0.0, 0.1], [0.1, 0.2]],
    )
    with pytest.raises(ValueError) as exc:
        trajectory.load_sto(p, ["lumbar_extension", "hip_flexion_r"])
    assert "hip_flexion_r" in str(exc.value)
    assert "lumbar_extension" in str(exc.value)


def test_missing_dof_can_be_zero_filled(tmp_path):
    p = _write_sto(
        tmp_path / "e.sto",
        ["time", "/jointset/back/lumbar_extension/value"],
        [[0.0, 0.1], [0.1, 0.2]],
    )
    traj = trajectory.load_sto(p, ["lumbar_extension", "hip_flexion_r"], require_all=False)
    assert traj.missing == ("hip_flexion_r",)
    assert np.all(traj.q[:, 1] == 0.0)


def test_file_without_endheader_raises(tmp_path):
    p = tmp_path / "f.sto"
    p.write_text("time\tpelvis_ty\n0\t1\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        trajectory.load_sto(p, ["pelvis_ty"])
    assert "endheader" in str(exc.value)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        trajectory.load_sto("does/not/exist.sto", ["pelvis_ty"])


def test_single_frame_is_rejected(tmp_path):
    p = _write_sto(tmp_path / "g.sto", ["time", "pelvis_ty"], [[0.0, 1.0]])
    with pytest.raises(ValueError):
        trajectory.load_sto(p, ["pelvis_ty"])


# -- sampling ---------------------------------------------------------------


def _synthetic(n=100):
    q = np.linspace(0, 1, n).reshape(-1, 1)
    return trajectory.Trajectory(
        dof_names=("pelvis_ty",),
        time=np.linspace(0, 1, n),
        q=q,
        dq=q * 2.0,
        source="synthetic",
        convention="test",
    )


def test_sampling_covers_the_whole_reference():
    traj = _synthetic()
    rng = np.random.RandomState(0)
    seen = {traj.sample_frame(rng)[0] for _ in range(2000)}
    assert min(seen) == 0
    assert max(seen) == traj.n_frames - 1
    assert len(seen) > traj.n_frames * 0.8


def test_phase_range_restricts_sampling():
    traj = _synthetic()
    rng = np.random.RandomState(0)
    seen = {traj.sample_frame(rng, (0.5, 0.6))[0] for _ in range(500)}
    assert min(seen) >= 49
    assert max(seen) <= 60


def test_bad_phase_range_raises():
    traj = _synthetic()
    for bad in [(0.5, 0.5), (-0.1, 0.5), (0.5, 1.5), (0.8, 0.2)]:
        with pytest.raises(ValueError):
            traj.frame_range(bad)


def test_frame_returns_a_copy():
    traj = _synthetic()
    q, _ = traj.frame(3)
    q[0] = 999.0
    assert traj.q[3, 0] != 999.0


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        trajectory.Trajectory(
            dof_names=("a", "b"),
            time=np.zeros(4),
            q=np.zeros((4, 1)),
            dq=np.zeros((4, 1)),
        )


# -- RSI configuration ------------------------------------------------------


def test_stage_a_uses_rsi_with_init_reference():
    rsi = stages.STAGES["A"].rsi
    assert rsi is not None
    assert rsi.posture_reference == stages.INIT_FRAME
    assert rsi.velocity_scale == 0.0


def test_stage_b_matches_stage_a_rsi():
    """A and B must share an initial-state distribution.

    Stage A learns to hold the reference's forward-leaning poses. If B reset
    upright and scored posture against upright, the A->B transfer would change
    the init distribution, the posture reference and the reward simultaneously.
    """
    a, b = stages.STAGES["A"].rsi, stages.STAGES["B"].rsi
    assert b is not None
    assert b.trajectory == a.trajectory
    assert b.velocity_scale == a.velocity_scale
    assert b.posture_reference == a.posture_reference
    assert b.phase_range == a.phase_range


def test_locomotion_stages_do_not_use_rsi_yet():
    for key in ("C", "D"):
        assert stages.STAGES[key].rsi is None, key


def test_rsi_validates_its_fields():
    with pytest.raises(ValueError):
        stages.RSIConfig(velocity_scale=1.5)
    with pytest.raises(ValueError):
        stages.RSIConfig(velocity_scale=-0.1)
    with pytest.raises(ValueError):
        stages.RSIConfig(posture_reference="whatever")
    with pytest.raises(ValueError):
        stages.RSIConfig(phase_range=(0.9, 0.1))


def test_rsi_fields_are_overridable():
    stage = stages.STAGES["A"].with_overrides(rsi_velocity_scale=0.5)
    assert stage.rsi.velocity_scale == pytest.approx(0.5)
    assert stages.STAGES["A"].rsi.velocity_scale == 0.0


def test_rsi_override_can_switch_the_posture_reference():
    stage = stages.STAGES["A"].with_overrides(rsi_posture_reference="neutral")
    assert stage.rsi.posture_reference == stages.NEUTRAL


def test_rsi_override_creates_a_config_for_a_stage_without_one():
    stage = stages.STAGES["B"].with_overrides(rsi_velocity_scale=0.25)
    assert stage.rsi is not None
    assert stage.rsi.velocity_scale == pytest.approx(0.25)


def test_unknown_rsi_override_raises():
    with pytest.raises(KeyError):
        stages.STAGES["A"].with_overrides(rsi_nonsense=1.0)


def test_bad_rsi_override_still_validates():
    with pytest.raises(ValueError):
        stages.STAGES["A"].with_overrides(rsi_velocity_scale=2.0)


# -- the real reference -----------------------------------------------------

pytestmark_reference = pytest.mark.skipif(
    not REFERENCE.is_file(), reason="reference trajectory not present"
)


@pytestmark_reference
def test_real_reference_covers_every_model_dof():
    traj = trajectory.load_sto(REFERENCE, MODEL_DOFS)
    assert traj.missing == ()
    assert traj.convention == "moco"
    assert traj.n_frames > 100


@pytestmark_reference
def test_real_reference_is_compatible_with_locked_ankles():
    """The model locks ankle and mtp; the reference must keep them near zero.

    If a future reference has real ankle motion, zeroing those dofs at reset
    would silently distort every sampled frame.
    """
    traj = trajectory.load_sto(REFERENCE, MODEL_DOFS)
    for dof in ("ankle_angle_r", "ankle_angle_l", "mtp_angle_r", "mtp_angle_l"):
        col = traj.q[:, MODEL_DOFS.index(dof)]
        assert np.abs(col).max() < 0.05, "%s reaches %.4f rad" % (dof, np.abs(col).max())


@pytestmark_reference
def test_real_reference_leans_forward_throughout():
    """Why posture_reference must be "init" for stage A.

    The reference trunk never approaches upright, so posture measured against
    an upright ideal would score near zero on every frame.
    """
    traj = trajectory.load_sto(REFERENCE, MODEL_DOFS)
    tilt = traj.q[:, MODEL_DOFS.index("pelvis_tilt")]
    assert tilt.max() < -0.3
    sigma = stages.STAGES["A"].terms.pelvis_tilt_sigma
    assert rewards.gaussian(float(tilt.max()), sigma) < 0.01
