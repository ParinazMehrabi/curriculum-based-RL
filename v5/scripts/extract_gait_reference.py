"""Turn the ExoGait benchmark log into a phase-averaged hip and knee cycle.

    python scripts/extract_gait_reference.py

Reads `data/AB68_ExoGait-Benchmark.csv` (220 MB, 468857 rows at 333 Hz) and
writes `models/reference/exogait_hip_knee.csv`: one row per percent of the gait
cycle, four columns, in this model's own convention. The big file stays out of
the runtime path -- nothing but this script reads it.

**Which column is which joint was measured, not assumed.** The log names its
four dofs `JointPositions_1..4` and says nothing about order. Two independent
signals settle it:

* `gait_cycle_left` is the cycle counter for whichever leg it belongs to, so
  the hip that swings across *that* counter is the left one. Across
  `gait_cycle_left` the swing is -0.370 rad for `_1` and -0.096 for `_3`;
  across `gait_cycle_right` it is -0.035 for `_1` and -0.291 for `_3`.
  So `_1` is the left hip and `_3` the right.
* Each knee correlates most strongly with its own hip: corr(_1, _2) = -0.549
  against corr(_3, _2) = +0.209, and corr(_3, _4) = -0.658 against
  corr(_1, _4) = +0.349. So `_2` is the left knee and `_4` the right.

**The knee sign is flipped.** The log's knee is negative throughout
(-1.94 .. 0 rad); MyoSuite's `knee_angle` is positive-is-flexion over
[0, 2.0944]. The hips already agree, positive being flexion in both.

**Only the `transparent_WALKING` condition is used** -- 30215 rows, 91 s. The
log also holds ramps, stairs and the exo's own assistance modes; "transparent"
means the exoskeleton is backdriving rather than driving, so it is the closest
thing in the record to unassisted level walking. Stair and ramp rows reach
0.97 rad of hip flexion against 0.73 for level walking and would distort the
average.

**The cycle takes 2.76 s.** This is a slow walk -- 43 steps a minute -- which
is what the record is: an exoskeleton benchmark, not a healthy overground
trial. The duration is written into the output header, because the phase clock
needs it and it is a property of the data rather than of the model.

**Averaging is by phase, not by time.** The walking rows come in 292
fragments, the longest only 3 s, so no single stretch is a usable trajectory.
Binning every row by its `gait_cycle_right` percentage instead gives ~297
samples per bin and one clean cycle. Reading the left leg against the *right*
leg's counter keeps the measured left-right offset rather than assuming the
legs are exactly half a cycle apart.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parents[1]
REPO = V5.parent
SOURCE = V5 / "data" / "AB68_ExoGait-Benchmark.csv"
DEST = REPO / "models" / "reference" / "exogait_hip_knee.csv"

CONDITION = "transparent_WALKING"
# log column -> (model joint, sign), measured; see the module docstring
COLUMNS = (
    ("JointPositions_3", "hip_flexion_r", +1.0),
    ("JointPositions_4", "knee_angle_r", -1.0),
    ("JointPositions_1", "hip_flexion_l", +1.0),
    ("JointPositions_2", "knee_angle_l", -1.0),
)
CYCLE_COLUMN = "gait_cycle_right"
BINS = 100


def read_walking(path: Path, condition: str = CONDITION):
    """Joint angles, cycle percentage and timestamps, for walking rows only."""
    angles, phase, time = [], [], []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c, _, _ in COLUMNS if c not in reader.fieldnames]
        if missing or CYCLE_COLUMN not in reader.fieldnames:
            raise KeyError("%s lacks %s" % (path, missing or [CYCLE_COLUMN]))
        for rec in reader:
            if rec["condition"] != condition:
                continue
            angles.append([float(rec[c]) * sign for c, _, sign in COLUMNS])
            phase.append(float(rec[CYCLE_COLUMN]))
            time.append(float(rec["time"]))
    return np.array(angles), np.array(phase), np.array(time)


def cycle_duration(phase: np.ndarray, time: np.ndarray,
                   max_gap: float = 0.02) -> float:
    """Seconds per gait cycle: median time between counter wraps.

    Measured **inside** a continuous fragment. The walking rows are 16
    fragments with gaps of up to 10 s between them, and the sample period is
    irregular (0.4 .. 3 ms, median 2.8 ms), which rules out the two shortcuts:
    timing wraps without checking continuity charges a fragment gap to a gait
    cycle, and dividing a total phase advance by a total elapsed time trusts
    that irregular period and reads 2.08 s.

    Nineteen wrap-to-wrap intervals survive the continuity test, at
    2.762 +- 0.314 s. Cross-checked against the record itself rather than its
    counter: the autocorrelation of the right hip angle over the longest
    fragment (17.2 s) peaks at 2.585 s, and the cross-correlation of the two
    hips peaks at 1.42 s, half a cycle apart as contralateral legs should be.
    """
    gaps = np.diff(time)
    fragments = np.split(np.arange(len(time)), np.flatnonzero(gaps > max_gap) + 1)

    durations = []
    for frag in fragments:
        if len(frag) < 2:
            continue
        wraps = np.flatnonzero(np.diff(phase[frag]) < -50)
        durations += [time[frag][b] - time[frag][a]
                      for a, b in zip(wraps, wraps[1:])]
    if not durations:
        raise ValueError("no two cycle wraps fall inside one continuous fragment")
    return float(np.median(durations))


def phase_average(angles: np.ndarray, phase: np.ndarray, bins: int = BINS):
    """Mean pose per phase bin, and how many samples each bin got."""
    idx = np.clip(phase.astype(int) * bins // 100, 0, bins - 1)
    total = np.zeros((bins, angles.shape[1]))
    count = np.zeros(bins, dtype=int)
    np.add.at(total, idx, angles)
    np.add.at(count, idx, 1)
    if not count.all():
        raise ValueError("phase bins %s got no samples" % np.flatnonzero(count == 0))
    return total / count[:, None], count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--condition", default=CONDITION)
    ap.add_argument("--bins", type=int, default=BINS)
    args = ap.parse_args(argv)

    if not args.source.is_file():
        raise SystemExit("no such file: %s" % args.source)

    angles, phase, time = read_walking(args.source, args.condition)
    print("%s: %d rows of %s" % (args.source.name, len(angles), args.condition))
    duration = cycle_duration(phase, time)
    print("gait cycle: %.3f s (%.1f steps/min)" % (duration, 2 * 60 / duration))
    cycle, count = phase_average(angles, phase, args.bins)
    print("%d phase bins, %d .. %d samples each" % (args.bins, count.min(), count.max()))

    names = [name for _, name, _ in COLUMNS]
    for i, name in enumerate(names):
        c = cycle[:, i]
        print("  %-14s %+6.3f .. %+6.3f rad  (%+.0f .. %+.0f deg)"
              % (name, c.min(), c.max(), np.degrees(c.min()), np.degrees(c.max())))
    seam = float(np.abs(cycle[0] - cycle[-1]).sum())
    print("seam discontinuity (last bin -> first): %.4f rad over %d dofs"
          % (seam, len(names)))

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with args.dest.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# source=%s condition=%s rows=%d duration_s=%.4f\n"
                 % (args.source.name, args.condition, len(angles), duration))
        w = csv.writer(fh)
        w.writerow(["phase"] + names)
        for i, row in enumerate(cycle):
            w.writerow(["%.4f" % (i / args.bins)] + ["%.6f" % v for v in row])
    print("wrote %s" % args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
