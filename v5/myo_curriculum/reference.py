"""The reference gait: hip and knee only, from the ExoGait benchmark record.

Loads `models/reference/exogait_hip_knee.csv`, which
`scripts/extract_gait_reference.py` builds from `data/AB68_ExoGait-Benchmark.csv`
-- 30215 rows of level walking at 2.76 s a cycle, phase-averaged into 100
frames. That script documents how each of the log's four unnamed dofs was
identified and why the knee sign is flipped; this module only maps the result
onto the model.

**Four tracked dofs, not nine.** Hip and knee, both sides. The record has no
pelvis, lumbar or ankle channel, so the tracking term scores hip and knee and
nothing else, and the remaining joints are the policy's to choose. Two
consequences worth stating rather than discovering:

* **The pelvis height is not prescribed.** The earlier `.sto` reference carried
  an absolute pelvis height, so posing the model at a phase also placed it
  vertically. Here `pose_at` returns no height and `apply_reference_pose`
  leaves the pelvis where the solved standing stance put it, which keeps the
  feet seated on the floor at reset.
* **The tracking error is an RMS over four dofs, not nine**, so the same
  numeric value is a larger per-joint deviation than it used to be.
  `StageSpec.max_tracking_error` is the same 1.50, which on four dofs is a
  backstop rather than a limit that binds.

**The angles need no unit conversion and one sign flip.** The log is in
radians. Hip flexion is positive in both conventions; the log's knee is
negative throughout (-1.94 .. 0 rad) against MyoSuite's positive-is-flexion
[0, 2.0944], so the extractor negates it and the values in the CSV are already
in this model's convention.

The cycle wraps at frame 99 with a 0.115 rad discontinuity summed over the four
dofs -- cleaner than the 0.30 rad seam of the `.sto` record it replaces,
because phase-averaging a hundred cycles removes the stride-to-stride variation
that made a single tracked cycle non-cyclic.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = REPO_ROOT / "models" / "reference" / "exogait_hip_knee.csv"

# The joints the tracking term scores, in the order the reference file holds
# them. Hip and knee only; see the module docstring.
TRACKED_JOINTS: Tuple[str, ...] = (
    "hip_flexion_r",
    "knee_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
)

# Kept for the record: what each tracked joint is called in the source log, and
# the sign the extractor applies. Nothing reads this at runtime.
SOURCE_COLUMNS: Dict[str, Tuple[str, float]] = {
    "hip_flexion_r": ("JointPositions_3", +1.0),
    "knee_angle_r": ("JointPositions_4", -1.0),
    "hip_flexion_l": ("JointPositions_1", +1.0),
    "knee_angle_l": ("JointPositions_2", -1.0),
}

DEFAULT_DURATION = 2.7596      # seconds per gait cycle, measured


class Reference:
    """The reference gait, resampled by phase in [0, 1)."""

    def __init__(self, path=None, duration: Optional[float] = None):
        path = Path(path) if path else DEFAULT_REFERENCE
        if not path.is_file():
            raise FileNotFoundError(
                "reference gait not found: %s -- run "
                "scripts/extract_gait_reference.py" % path
            )

        header, rows = self._read(path)
        self.path = path
        self.source = header.get("source", path.name)
        self.condition = header.get("condition", "")
        self.joints = rows
        self.n_frames = len(rows)
        if self.n_frames < 2:
            raise ValueError("%s holds %d frames" % (path, self.n_frames))

        # The cycle duration is a property of the record, so it travels in the
        # file's header; the argument is for tests and for deliberately
        # retiming the gait.
        self.duration = float(
            duration if duration is not None
            else header.get("duration_s", DEFAULT_DURATION)
        )
        if self.duration <= 0.0:
            raise ValueError("duration must be positive, got %r" % self.duration)

        self.seam = float(np.linalg.norm(self.joints[-1] - self.joints[0]))

    @staticmethod
    def _read(path: Path) -> Tuple[Dict[str, object], np.ndarray]:
        """The `# key=value` header and the frames, in TRACKED_JOINTS order."""
        header: Dict[str, object] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            lines = []
            for line in fh:
                if line.startswith("#"):
                    for field in line[1:].split():
                        key, _, value = field.partition("=")
                        try:
                            header[key] = float(value)
                        except ValueError:
                            header[key] = value
                    continue
                lines.append(line)
            reader = csv.DictReader(lines)
            missing = [j for j in TRACKED_JOINTS if j not in (reader.fieldnames or [])]
            if missing:
                raise KeyError("%s lacks columns %s" % (path, missing))
            rows = [[float(rec[j]) for j in TRACKED_JOINTS] for rec in reader]
        return header, np.asarray(rows, dtype=np.float64)

    # -- sampling ---------------------------------------------------------

    def pose_at(self, phase: float) -> Tuple[np.ndarray, None]:
        """Joint targets at a phase in [0, 1), linearly interpolated.

        The second element is the pelvis height, and it is always None: this
        record has no height channel. Callers keep the two-value shape so the
        distinction stays explicit at every call site.
        """
        p = float(phase) % 1.0
        x = p * self.n_frames
        i = int(np.floor(x)) % self.n_frames
        j = (i + 1) % self.n_frames
        f = x - np.floor(x)
        return (1.0 - f) * self.joints[i] + f * self.joints[j], None

    def advance(self, phase: float, dt: float) -> float:
        """Move the phase on by `dt` seconds of wall time, wrapping."""
        return (float(phase) + dt / self.duration) % 1.0

    def describe(self) -> str:
        return (
            "%s | %d frames, %.2f s cycle, seam %.3f rad | %d tracked joints: %s"
            % (self.path.name, self.n_frames, self.duration, self.seam,
               len(TRACKED_JOINTS), ", ".join(TRACKED_JOINTS))
        )
