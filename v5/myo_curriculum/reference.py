"""The reference gait, mapped onto this model.

Loads `models/reference/gaitTracking_solution_raw.sto` -- an OpenSim Moco
tracking solution, 301 frames over 6.4 s at a mean forward speed of
0.142 m/s -- through v4's `.sto` loader, and maps its dofs onto the planar
MyoSuite model.

Two things about that mapping are not identities.

**pelvis_tilt is sign-flipped.** The `.hfd` uses OpenSim's convention where
negative pelvis_tilt is a forward lean; this model's `pelvis_tilt` hinge is
positive-forward. Measured, not assumed: setting `pelvis_tilt = +0.30` here
puts the trunk axis at `x = +0.233`, and `-0.30` at `-0.357`. Everything else
agrees -- `lumbar_extension` -> `flex_extension`, hip and knee flexion all
share sign, which is why this reference fits the MyoSuite joint ranges when it
did not fit the `.hfd`'s (see the v4 README, section 11).

**pelvis_ty is a height, not a joint value.** In the `.hfd` it is the absolute
pelvis height. Here `pelvis_ty` is a slide offset from the root body's base
position, so the reference height is applied by solving for the offset that
produces it. The slide is a pure translation along z, so one `mj_forward` is
enough to get it exactly.

The record is not perfectly cyclic -- it is tracked from real data, so cycles
differ. The best wrap point is frame 195 (4.16 s), where the pose differs from
frame 0 by 0.30 rad summed over nine dofs. Tracking dips briefly at the seam;
`loop_end` moves it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = REPO_ROOT / "models" / "reference" / "gaitTracking_solution_raw.sto"
V4_TRAJECTORY = REPO_ROOT / "v4" / "sconegym_crutch_v4" / "trajectory.py"

# reference dof -> (model joint, sign). See the module docstring for why
# pelvis_tilt is negated and why pelvis_ty is handled apart from these.
JOINT_MAP: Dict[str, Tuple[str, float]] = {
    "pelvis_tilt": ("pelvis_tilt", -1.0),
    "lumbar_extension": ("flex_extension", +1.0),
    "hip_flexion_r": ("hip_flexion_r", +1.0),
    "knee_angle_r": ("knee_angle_r", +1.0),
    "ankle_angle_r": ("ankle_angle_r", +1.0),
    "hip_flexion_l": ("hip_flexion_l", +1.0),
    "knee_angle_l": ("knee_angle_l", +1.0),
    "ankle_angle_l": ("ankle_angle_l", +1.0),
}
HEIGHT_DOF = "pelvis_ty"

# The joints the tracking term scores, in a fixed order.
TRACKED_JOINTS: Tuple[str, ...] = tuple(v[0] for v in JOINT_MAP.values())

_MODULE_NAME = "_v5_trajectory_from_v4"


def _load_v4_loader():
    """v4's .sto reader, imported by path for the reasons rewards.py explains."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    if not V4_TRAJECTORY.is_file():
        raise FileNotFoundError(
            "v5 reuses v4's .sto loader, but it is not at %s" % V4_TRAJECTORY
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, V4_TRAJECTORY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class Reference:
    """The reference gait, resampled by phase in [0, 1)."""

    def __init__(self, path=None, loop_end: int = 195):
        path = Path(path) if path else DEFAULT_REFERENCE
        if not path.is_file():
            raise FileNotFoundError("reference trajectory not found: %s" % path)
        loader = _load_v4_loader()

        dof_names = list(JOINT_MAP) + [HEIGHT_DOF]
        traj = loader.load_sto(path, dof_names, require_all=True)
        q = np.asarray(traj.q, dtype=np.float64)
        time = np.asarray(traj.time, dtype=np.float64)

        if not 1 < loop_end <= traj.n_frames:
            raise ValueError(
                "loop_end must lie in (1, %d], got %d" % (traj.n_frames, loop_end)
            )
        self.path = path
        self.source = getattr(traj, "source", path.name)
        self.n_frames = int(loop_end)

        index = {n: i for i, n in enumerate(dof_names)}
        # Sign-corrected joint targets, in TRACKED_JOINTS order.
        self.joints = np.stack(
            [
                sign * q[:loop_end, index[ref]]
                for ref, (_, sign) in JOINT_MAP.items()
            ],
            axis=1,
        )
        self.height = q[:loop_end, index[HEIGHT_DOF]].copy()
        self.time = time[:loop_end] - time[0]
        self.duration = float(self.time[-1] - self.time[0]) + float(
            self.time[1] - self.time[0]
        )
        self.seam = float(np.linalg.norm(self.joints[-1] - self.joints[0]))

    # -- sampling ---------------------------------------------------------

    def pose_at(self, phase: float) -> Tuple[np.ndarray, float]:
        """Joint targets and pelvis height at a phase in [0, 1), interpolated."""
        p = float(phase) % 1.0
        x = p * self.n_frames
        i = int(np.floor(x)) % self.n_frames
        j = (i + 1) % self.n_frames
        f = x - np.floor(x)
        joints = (1.0 - f) * self.joints[i] + f * self.joints[j]
        height = (1.0 - f) * self.height[i] + f * self.height[j]
        return joints, float(height)

    def advance(self, phase: float, dt: float) -> float:
        """Move the phase on by `dt` seconds of wall time, wrapping."""
        return (float(phase) + dt / self.duration) % 1.0

    def describe(self) -> str:
        return (
            "%s | %d frames, %.2f s loop, seam %.3f rad | %d tracked joints"
            % (self.path.name, self.n_frames, self.duration, self.seam,
               len(TRACKED_JOINTS))
        )
