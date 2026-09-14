"""Reference trajectory loading for reference-state initialisation (RSI).

No sconegym/sconepy imports, so this is unit-testable without a simulator.

The reference in this project is an OpenSim Moco tracking solution, whose
columns are named `/jointset/<joint>/<coord>/value` and `.../speed`. Other
conventions (bare coordinate names, `<coord>_u` velocities) are also accepted,
and the loader reports which one it matched rather than guessing silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

END_HEADER = "endheader"


@dataclass(frozen=True)
class Trajectory:
    """Per-frame generalised positions and velocities, in a model's dof order."""

    dof_names: Tuple[str, ...]
    time: np.ndarray  # (n_frames,)
    q: np.ndarray  # (n_frames, n_dofs)
    dq: np.ndarray  # (n_frames, n_dofs)
    source: str = ""
    convention: str = ""
    missing: Tuple[str, ...] = ()

    def __post_init__(self):
        n, d = len(self.time), len(self.dof_names)
        if self.q.shape != (n, d) or self.dq.shape != (n, d):
            raise ValueError(
                "shape mismatch: time=%d dofs=%d q=%r dq=%r"
                % (n, d, self.q.shape, self.dq.shape)
            )
        if n < 2:
            raise ValueError("a trajectory needs at least 2 frames, got %d" % n)

    @property
    def n_frames(self) -> int:
        return len(self.time)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    def frame(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        i = int(index) % self.n_frames
        return self.q[i].copy(), self.dq[i].copy()

    def frame_range(self, phase_range: Tuple[float, float] = (0.0, 1.0)) -> Tuple[int, int]:
        lo, hi = float(phase_range[0]), float(phase_range[1])
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("phase_range must satisfy 0 <= lo < hi <= 1, got %r" % (phase_range,))
        start = int(np.floor(lo * (self.n_frames - 1)))
        stop = int(np.ceil(hi * (self.n_frames - 1)))
        return start, max(start + 1, stop)

    def sample_frame(
        self,
        rng,
        phase_range: Tuple[float, float] = (0.0, 1.0),
        phase_windows: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Pick a uniformly random frame to reset to.

        This is the RSI mechanism from DeepMimic: resetting to a random phase of
        the reference rather than always to its first frame, so late-phase states
        are visited from the start of training instead of only once earlier
        phases are mastered.

        With `phase_windows`, sampling is restricted to a set of disjoint
        windows -- a window is chosen uniformly, then a frame within it. That is
        how a keyframe curriculum is expressed: give it the windows around the
        gait events and the policy only ever starts at one of those poses.
        """
        if phase_windows:
            window = phase_windows[int(rng.randint(len(phase_windows)))]
            start, stop = self.frame_range(window)
        else:
            start, stop = self.frame_range(phase_range)
        index = int(rng.randint(start, stop + 1)) if stop > start else start
        q, dq = self.frame(index)
        return index, q, dq

    def describe(self) -> str:
        lines = [
            "%s  (%s)" % (Path(self.source).name if self.source else "trajectory", self.convention),
            "  frames %d, %.3f s, dt %.4f s"
            % (self.n_frames, self.duration, self.duration / max(self.n_frames - 1, 1)),
        ]
        if self.missing:
            lines.append("  filled with zeros (not in source): %s" % ", ".join(self.missing))
        return "\n".join(lines)


def _read_table(path: Path) -> Tuple[List[str], np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        head = next(i for i, line in enumerate(text) if line.strip().lower() == END_HEADER)
    except StopIteration:
        raise ValueError("%s has no '%s' line; is it an .sto file?" % (path, END_HEADER)) from None

    columns = text[head + 1].split("\t")
    rows = []
    for line in text[head + 2 :]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != len(columns):
            continue
        rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError("%s has a header but no data rows" % path)
    return columns, np.asarray(rows, dtype=np.float64)


def _locate(columns: Sequence[str], dof: str) -> Tuple[Optional[int], Optional[int], str]:
    """Find the (position, velocity) column indices for one dof.

    Tries, in order: the Moco state paths, bare coordinate names, and the
    `<dof>_u` / `<dof>.u` velocity spellings used by some SCONE exports.
    """
    lowered = [c.strip() for c in columns]

    def find(pred):
        for i, c in enumerate(lowered):
            if pred(c):
                return i
        return None

    pos = find(lambda c: c.endswith("/%s/value" % dof))
    vel = find(lambda c: c.endswith("/%s/speed" % dof))
    if pos is not None:
        return pos, vel, "moco"

    pos = find(lambda c: c == dof)
    if pos is not None:
        vel = find(lambda c: c in ("%s_u" % dof, "%s.u" % dof, "%s_speed" % dof))
        return pos, vel, "bare"

    return None, None, "none"


def load_sto(
    path,
    dof_names: Sequence[str],
    require_all: bool = True,
    finite_difference_velocities: bool = True,
) -> Trajectory:
    """Load `path` and return its frames in `dof_names` order.

    Any dof absent from the file is zero-filled and named in
    `Trajectory.missing` -- unless require_all, in which case it raises.
    Velocities absent from the file are finite-differenced from the positions
    when finite_difference_velocities is set.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("reference trajectory not found: %s" % path)

    columns, data = _read_table(path)
    time_idx = next((i for i, c in enumerate(columns) if c.strip().lower() == "time"), None)
    if time_idx is None:
        raise ValueError("%s has no 'time' column" % path)
    time = data[:, time_idx]

    n = data.shape[0]
    q = np.zeros((n, len(dof_names)), dtype=np.float64)
    dq = np.zeros((n, len(dof_names)), dtype=np.float64)
    missing: List[str] = []
    differenced: List[str] = []
    conventions = set()

    for col, dof in enumerate(dof_names):
        pos_idx, vel_idx, how = _locate(columns, dof)
        if pos_idx is None:
            missing.append(dof)
            continue
        conventions.add(how)
        q[:, col] = data[:, pos_idx]
        if vel_idx is not None:
            dq[:, col] = data[:, vel_idx]
        elif finite_difference_velocities and n > 1:
            dq[:, col] = np.gradient(q[:, col], time, edge_order=1)
            differenced.append(dof)

    if missing and require_all:
        available = sorted({c.split("/")[-2] for c in columns if c.count("/") >= 2})
        raise ValueError(
            "%s does not contain these dofs: %s\n"
            "coordinates found in the file: %s"
            % (path, ", ".join(missing), ", ".join(available) or "(none)")
        )

    convention = "+".join(sorted(conventions)) or "empty"
    if differenced:
        convention += " (velocities differenced for: %s)" % ", ".join(differenced)

    return Trajectory(
        dof_names=tuple(dof_names),
        time=time,
        q=q,
        dq=dq,
        source=str(path),
        convention=convention,
        missing=tuple(missing),
    )


def summarise(traj: Trajectory) -> str:
    lines = [traj.describe(), "", "%-18s %-22s %-22s" % ("dof", "value min..max", "speed min..max")]
    for i, dof in enumerate(traj.dof_names):
        lines.append(
            "%-18s %+8.4f ..%+8.4f   %+8.4f ..%+8.4f"
            % (dof, traj.q[:, i].min(), traj.q[:, i].max(), traj.dq[:, i].min(), traj.dq[:, i].max())
        )
    return "\n".join(lines)
