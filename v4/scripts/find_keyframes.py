"""Find the four keyframes of the reciprocal crutch-gait cycle in the reference.

Crutch-assisted reciprocal gait advances in four sub-movements:

    1. one crutch forward
    2. the opposite leg forward
    3. the other crutch forward
    4. that side's opposite leg forward

Each is detected as a peak in the relevant joint angle: arm_flex_r/l for the
crutches (they are welded to the forearms, so arm flexion is crutch position)
and hip_flexion_r/l for the legs.

Runs on numpy alone -- no simulator needed.

    python scripts/find_keyframes.py
    python scripts/find_keyframes.py --half-width 0.20 --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_V4 = Path(__file__).resolve().parents[1]
if str(REPO_V4) not in sys.path:
    sys.path.insert(0, str(REPO_V4))

import numpy as np

DEFAULT_REFERENCE = REPO_V4.parent / "models" / "reference" / "gaitTracking_solution_raw.sto"

# The four sub-movements, in the order they occur, as (label, dof, side).
# Note the contralateral pairing: the right crutch advances with the left leg.
EVENTS = (
    ("crutch_r", "arm_flex_r"),
    ("leg_l", "hip_flexion_l"),
    ("crutch_l", "arm_flex_l"),
    ("leg_r", "hip_flexion_r"),
)

TRACKED_DOFS = (
    "pelvis_tilt",
    "pelvis_tx",
    "pelvis_ty",
    "hip_flexion_r",
    "hip_flexion_l",
    "knee_angle_r",
    "knee_angle_l",
    "lumbar_extension",
    "arm_flex_r",
    "arm_flex_l",
    "elbow_flex_r",
    "elbow_flex_l",
)


def _load_trajectory_module():
    """Import trajectory.py without running the package __init__.

    The __init__ imports gym in order to register environments; this script
    needs only numpy, so it loads the submodule under a synthetic package name.
    """
    import importlib
    import types

    name = "_scv4_keyframes"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(REPO_V4 / "sconegym_crutch_v4")]
        sys.modules[name] = pkg
    return importlib.import_module(name + ".trajectory")


def smooth(x: np.ndarray, window: int = 9) -> np.ndarray:
    if window < 3:
        return x
    kernel = np.ones(window) / float(window)
    padded = np.concatenate([np.full(window, x[0]), x, np.full(window, x[-1])])
    return np.convolve(padded, kernel, mode="same")[window:-window]


def find_peaks(x: np.ndarray, min_separation: int, prominence: float) -> list:
    """Local maxima at least min_separation apart and prominence above the mean."""
    threshold = x.mean() + prominence * x.std()
    candidates = [
        i for i in range(1, len(x) - 1)
        if x[i] >= x[i - 1] and x[i] >= x[i + 1] and x[i] >= threshold
    ]
    kept = []
    for i in candidates:
        if kept and i - kept[-1] < min_separation:
            if x[i] > x[kept[-1]]:
                kept[-1] = i
        else:
            kept.append(i)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument(
        "--half-width",
        type=float,
        default=0.15,
        help="half-width of each keyframe window, in seconds",
    )
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument(
        "--min-separation",
        type=int,
        default=None,
        help="minimum frames between peaks of the same event. Defaults to a "
        "third of the record, which admits about two occurrences each. Too "
        "small and the secondary bumps inside a stride are mistaken for events.",
    )
    ap.add_argument("--prominence", type=float, default=0.3)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    load_sto = _load_trajectory_module().load_sto

    traj = load_sto(args.reference, TRACKED_DOFS, require_all=False)
    t = traj.time - traj.time[0]
    dt = float(np.diff(t).mean())
    n = traj.n_frames

    print("=" * 78)
    print("reference:", Path(args.reference).name)
    print("frames %d, duration %.2f s, dt %.4f s" % (n, t[-1], dt))
    tx = traj.q[:, TRACKED_DOFS.index("pelvis_tx")]
    print("travel %.3f m -> mean speed %.4f m/s" % (tx[-1] - tx[0], (tx[-1] - tx[0]) / t[-1]))
    print("=" * 78)
    print()

    # Detect each event's peaks, then interleave them in time order.
    detected = {}
    for label, dof in EVENTS:
        series = smooth(traj.q[:, TRACKED_DOFS.index(dof)], args.smooth)
        # Each event recurs once per cycle. The reference holds about two
        # cycles, so peaks of the same event sit roughly n/2 apart; requiring
        # n/3 admits both while rejecting the secondary bumps within a stride,
        # which a smaller separation picks up as false events.
        min_sep = args.min_separation or max(1, n // 3)
        peaks = find_peaks(series, min_separation=min_sep, prominence=args.prominence)
        detected[label] = peaks
        print("%-10s (%-14s) frames %s" % (label, dof, peaks))
        print("%-10s %-16s times  %s" % ("", "", ["%.2f" % t[i] for i in peaks]))
    print()

    ordered = sorted(
        ((i, label) for label, peaks in detected.items() for i in peaks),
        key=lambda pair: pair[0],
    )
    print("cycle order:")
    for i, label in ordered:
        print("  %6.2f s  frame %3d  %s" % (t[i], i, label))

    # Cycle period from successive occurrences of the same event.
    periods = []
    for label, peaks in detected.items():
        periods += [t[b] - t[a] for a, b in zip(peaks, peaks[1:])]
    if periods:
        print()
        print("cycle period: %.2f s (%.0f frames), from %d intervals"
              % (np.mean(periods), np.mean(periods) / dt, len(periods)))

    half = args.half_width
    print()
    print("keyframe windows, as fractions of the record (half-width %.2f s):" % half)
    windows = {}
    for label, peaks in detected.items():
        spans = []
        for i in peaks:
            lo = max(0.0, (t[i] - half) / t[-1])
            hi = min(1.0, (t[i] + half) / t[-1])
            spans.append((round(float(lo), 4), round(float(hi), 4)))
        windows[label] = spans
        print("  %-10s %s" % (label, spans))

    print()
    print("as a flat tuple for RSIConfig.phase_windows:")
    flat = sorted(w for spans in windows.values() for w in spans)
    print("    phase_windows=(")
    for lo, hi in flat:
        print("        (%.4f, %.4f)," % (lo, hi))
    print("    ),")

    overlaps = [
        (a, b) for a, b in zip(flat, flat[1:]) if b[0] < a[1]
    ]
    if overlaps:
        print()
        print("WARNING: windows overlap; reduce --half-width")
        for a, b in overlaps:
            print("  %s and %s" % (a, b))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 4))
        for label, dof in EVENTS:
            ax.plot(t, smooth(traj.q[:, TRACKED_DOFS.index(dof)], args.smooth), lw=1.2, label=dof)
        for label, peaks in detected.items():
            for i in peaks:
                ax.axvline(t[i], color="#888780", lw=0.6, ls="--")
                ax.text(t[i], ax.get_ylim()[1], label, fontsize=7, rotation=90,
                        va="top", ha="right")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("joint angle (rad)")
        ax.set_title("Reciprocal crutch-gait keyframes", fontsize=10)
        ax.legend(fontsize=8, ncol=4, frameon=False)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        out = REPO_V4 / "notebooks" / "figures" / "gait_keyframes.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print()
        print("plot written to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
