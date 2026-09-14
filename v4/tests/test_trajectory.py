"""Tests for reference trajectory loading and RSI configuration.

Runs against the real Moco solution in models/reference/ when it is present,
and against synthetic .sto files otherwise, so the parser is covered either way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _bootstrap import PKG_DIR, env_source, load, load_trajectory

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


def test_stage_c_matches_the_shared_rsi():
    a, c = stages.STAGES["A"].rsi, stages.STAGES["C"].rsi
    assert c is not None
    assert c.trajectory == a.trajectory
    assert c.velocity_scale == a.velocity_scale
    assert c.posture_reference == a.posture_reference


def test_every_stage_shares_the_same_rsi():
    """All four stages must reset from the same distribution.

    Each transfer is a warm start; changing the initial-state distribution or
    the posture reference at a boundary discards what the previous stage learned.
    """
    base = stages.STAGES["A"].rsi
    for key in stages.STAGE_ORDER:
        rsi = stages.STAGES[key].rsi
        assert rsi is not None, key
        assert rsi.trajectory == base.trajectory, key
        assert rsi.velocity_scale == base.velocity_scale, key
        assert rsi.posture_reference == base.posture_reference, key
        assert rsi.phase_range == base.phase_range, key


def test_moving_stages_loosen_hip_knee():
    """C and D have to stride; A and B only have to stand."""
    for key in ("C", "D"):
        assert stages.STAGES[key].terms.hip_knee_sigma > stages.STAGES["A"].terms.hip_knee_sigma, key
    for key in ("A", "B"):
        assert stages.STAGES[key].terms.hip_knee_sigma == 0.35, key


def test_lag_terms_use_the_measured_episode_reference():
    """Under RSI the offsets are re-measured per episode, not taken from
    TermParams, because they swing through the stride."""
    src = env_source()
    assert "_measure_geometry_refs" in src
    assert "self._crutch_ref - offset" in src
    assert "- self._lag_ref" in src


def test_stage_c_loosens_hip_knee_for_locomotion():
    """posture must not fight velocity in the first stage that has to move.

    A stride moves hip flexion by roughly 0.6 rad. At the standing sigma of
    0.35 that scores about 0.05, so the 0.45-weighted posture term would punish
    every step the velocity term rewards.
    """
    assert stages.STAGES["C"].terms.hip_knee_sigma > stages.STAGES["B"].terms.hip_knee_sigma
    stride = 0.6
    tight = rewards.gaussian(stride, stages.STAGES["B"].terms.hip_knee_sigma)
    loose = rewards.gaussian(stride, stages.STAGES["C"].terms.hip_knee_sigma)
    assert tight < 0.1
    assert loose > 0.3


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


# -- keyframe curriculum ----------------------------------------------------


def test_keyframes_cover_the_four_sub_movements():
    """Reciprocal crutch gait: crutch forward, then the OPPOSITE leg forward."""
    assert set(stages.GAIT_KEYFRAMES) == {"crutch_r", "leg_l", "crutch_l", "leg_r"}


def test_keyframe_windows_are_disjoint_and_ordered():
    windows = stages.KEYFRAME_WINDOWS
    assert windows == tuple(sorted(windows))
    for (a_lo, a_hi), (b_lo, b_hi) in zip(windows, windows[1:]):
        assert a_hi <= b_lo, "windows %s and %s overlap" % ((a_lo, a_hi), (b_lo, b_hi))


def test_keyframe_windows_are_valid_fractions():
    for lo, hi in stages.KEYFRAME_WINDOWS:
        assert 0.0 <= lo < hi <= 1.0


def test_keyframes_follow_the_contralateral_order():
    """Within a cycle: crutch_r, leg_l, crutch_l, leg_r.

    The right crutch advances with the LEFT leg; getting this backwards would
    train an ipsilateral pattern, which is not how crutch gait works.
    """
    first = {name: windows[0][0] for name, windows in stages.GAIT_KEYFRAMES.items()}
    order = sorted(first, key=lambda n: first[n])
    assert order == ["crutch_r", "leg_l", "crutch_l", "leg_r"]


def test_stage_b_samples_only_the_keyframes():
    rsi = stages.STAGES["B"].rsi
    assert rsi.phase_window_groups == stages.KEYFRAME_GROUPS
    assert rsi.velocity_scale == 0.0, "the keyframe task is static"


def test_keyframe_groups_cover_every_window():
    flat = tuple(sorted(w for g in stages.KEYFRAME_GROUPS for w in g))
    assert flat == stages.KEYFRAME_WINDOWS
    assert len(stages.KEYFRAME_GROUPS) == 4


def test_sub_movements_are_sampled_equally():
    """leg_r has one window where the others have two.

    Sampling windows uniformly would give it half the episodes; grouping by
    sub-movement first gives each of the four an equal share.
    """
    traj = _synthetic(n=301)
    rng = np.random.RandomState(0)
    counts = {name: 0 for name in stages.GAIT_KEYFRAMES}
    draws = 4000
    for _ in range(draws):
        idx, _, _ = traj.sample_frame(rng, phase_window_groups=stages.KEYFRAME_GROUPS)
        frac = idx / (traj.n_frames - 1)
        for name, windows in stages.GAIT_KEYFRAMES.items():
            if any(lo - 0.01 <= frac <= hi + 0.01 for lo, hi in windows):
                counts[name] += 1
                break
    assert sum(counts.values()) == draws
    for name, count in counts.items():
        share = count / draws
        assert 0.20 < share < 0.30, (name, share, counts)


def test_flat_window_sampling_is_unbalanced():
    """Documents why groups exist: the flat form under-samples leg_r."""
    traj = _synthetic(n=301)
    rng = np.random.RandomState(0)
    leg_r = 0
    draws = 4000
    for _ in range(draws):
        idx, _, _ = traj.sample_frame(rng, phase_windows=stages.KEYFRAME_WINDOWS)
        frac = idx / (traj.n_frames - 1)
        if any(lo - 0.01 <= frac <= hi + 0.01 for lo, hi in stages.GAIT_KEYFRAMES["leg_r"]):
            leg_r += 1
    assert 0.11 < leg_r / draws < 0.18, leg_r / draws


def test_groups_take_precedence_over_windows():
    traj = _synthetic(n=301)
    rng = np.random.RandomState(3)
    idx, _, _ = traj.sample_frame(
        rng,
        phase_windows=((0.9, 1.0),),
        phase_window_groups=(((0.0, 0.05),),),
    )
    assert idx <= 16


def test_rsi_rejects_bad_phase_window_groups():
    with pytest.raises(ValueError):
        stages.RSIConfig(phase_window_groups=())
    with pytest.raises(ValueError):
        stages.RSIConfig(phase_window_groups=((),))
    with pytest.raises(ValueError):
        stages.RSIConfig(phase_window_groups=(((0.5, 0.4),),))


def test_stage_a_still_samples_the_whole_cycle():
    """A is general balance; only B narrows to the gait events."""
    assert stages.STAGES["A"].rsi.phase_windows is None


def test_window_sampling_hits_every_window():
    traj = _synthetic(n=301)
    rng = np.random.RandomState(0)
    windows = stages.KEYFRAME_WINDOWS
    seen_windows = set()
    for _ in range(2000):
        idx, _, _ = traj.sample_frame(rng, phase_windows=windows)
        frac = idx / (traj.n_frames - 1)
        hit = [w for w in windows if w[0] - 0.01 <= frac <= w[1] + 0.01]
        assert hit, "frame %d (frac %.4f) is outside every window" % (idx, frac)
        seen_windows.add(hit[0])
    assert seen_windows == set(windows)


def test_window_sampling_excludes_the_gaps():
    """Nothing between the keyframes should ever be sampled."""
    traj = _synthetic(n=301)
    rng = np.random.RandomState(1)
    windows = stages.KEYFRAME_WINDOWS
    for _ in range(1000):
        idx, _, _ = traj.sample_frame(rng, phase_windows=windows)
        frac = idx / (traj.n_frames - 1)
        assert any(w[0] - 0.01 <= frac <= w[1] + 0.01 for w in windows)


def test_phase_windows_takes_precedence_over_phase_range():
    traj = _synthetic(n=301)
    rng = np.random.RandomState(2)
    # A phase_range that excludes the first window, overridden by windows.
    idx, _, _ = traj.sample_frame(rng, phase_range=(0.9, 1.0), phase_windows=((0.0, 0.05),))
    assert idx <= 16


def test_rsi_rejects_bad_phase_windows():
    with pytest.raises(ValueError):
        stages.RSIConfig(phase_windows=())
    with pytest.raises(ValueError):
        stages.RSIConfig(phase_windows=((0.5, 0.4),))
    with pytest.raises(ValueError):
        stages.RSIConfig(phase_windows=((-0.1, 0.4),))


def test_phase_windows_are_overridable():
    stage = stages.STAGES["B"].with_overrides(rsi_phase_windows=((0.0, 0.1),))
    assert stage.rsi.phase_windows == ((0.0, 0.1),)
    assert stages.STAGES["B"].rsi.phase_windows is None


def test_phase_window_groups_are_overridable():
    stage = stages.STAGES["B"].with_overrides(rsi_phase_window_groups=(((0.0, 0.1),),))
    assert stage.rsi.phase_window_groups == (((0.0, 0.1),),)
    assert stages.STAGES["B"].rsi.phase_window_groups == stages.KEYFRAME_GROUPS
