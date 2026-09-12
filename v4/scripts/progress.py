"""Read a training run's reward curve from the log tonic writes.

deprl/tonic writes a CSV of per-epoch metrics into the run directory. With
epoch_steps=10000 in the stage configs, that is one row per 10k environment
steps. This prints those rows, and can plot or follow them.

Column names differ a little between tonic versions, so the reader matches
candidates rather than assuming one spelling, and says which it used.

    python scripts/progress.py                       # latest run, table
    python scripts/progress.py --plot                # + reward curve png
    python scripts/progress.py --follow              # keep printing new rows
    python scripts/progress.py --run <path-to-run-dir>
    python scripts/progress.py --list                # show runs it can see
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_V4 = Path(__file__).resolve().parents[1]
if str(REPO_V4) not in sys.path:
    sys.path.insert(0, str(REPO_V4))

# Where deprl puts runs on this project's training machine.
DEFAULT_ROOTS = (
    Path.home() / "Documents" / "SCONE" / "results",
    REPO_V4.parent / "results",
    Path("C:/Users/FUM Care/Documents/SCONE/results"),
)

STEP_COLUMNS = ("train/steps", "total_steps", "train/episodes", "steps", "epoch")
REWARD_COLUMNS = (
    "train/episode_score/mean",
    "train/episode_score",
    "test/episode_score/mean",
    "test/episode_score",
    "train/return",
    "episode_score",
)
# The eight reward components in rwd_dict. deprl collects these into
# rwd_metrics during the test episode and writes them as columns, but the prefix
# varies by version, so they are matched as substrings.
TERM_NAMES = (
    "alive",
    "height",
    "posture",
    "crutch",
    "velocity",
    "backward",
    "displacement",
)

LENGTH_COLUMNS = (
    "train/episode_length/mean",
    "train/episode_length",
    "test/episode_length/mean",
    "episode_length",
)


def find_logs(roots: Sequence[Path]) -> List[Path]:
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.csv"):
            if path.stat().st_size > 0:
                found.append(path)
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)


def read_csv(path: Path) -> tuple:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys()) if rows else []
    return fields, rows


def pick(fields: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in fields:
            return c
    lowered = {f.lower(): f for f in fields}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def find_term_columns(fields: Sequence[str]) -> Dict[str, str]:
    """Map term name -> column, for whichever reward components got logged."""
    found = {}
    for term in TERM_NAMES:
        matches = [
            f
            for f in fields
            if term in f.lower() and "length" not in f.lower() and "score" not in f.lower()
        ]
        if not matches:
            continue
        # Prefer a test/ column over train/, and the shortest name otherwise.
        matches.sort(key=lambda f: (0 if f.lower().startswith("test") else 1, len(f)))
        found[term] = matches[0]
    return found


def _num(row: Dict[str, str], key: Optional[str]) -> Optional[float]:
    if key is None:
        return None
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def report(path: Path, tail: int, plot: bool, show: bool) -> int:
    fields, rows = read_csv(path)
    if not rows:
        print("log is empty yet:", path)
        return 1

    step_col = pick(fields, STEP_COLUMNS)
    rwd_col = pick(fields, REWARD_COLUMNS)
    len_col = pick(fields, LENGTH_COLUMNS)

    print("=" * 78)
    print("run :", path.parent.name)
    print("log :", path)
    print("rows:", len(rows))
    print("columns used: step=%r reward=%r length=%r" % (step_col, rwd_col, len_col))
    if rwd_col is None:
        print()
        print("No reward column recognised. Available columns:")
        for f in fields:
            print("   ", f)
        print()
        print("Tell me which one is the reward and I will add it to REWARD_COLUMNS.")
        return 2
    print("=" * 78)
    print()

    shown = rows[-tail:] if tail > 0 else rows
    print("%14s %14s %14s" % (step_col, rwd_col, len_col or "-"))
    print("-" * 46)
    for row in shown:
        s, r, l = _num(row, step_col), _num(row, rwd_col), _num(row, len_col)
        print(
            "%14s %14s %14s"
            % (
                "%d" % s if s is not None else "-",
                "%.4f" % r if r is not None else "-",
                "%.1f" % l if l is not None else "-",
            )
        )

    terms = find_term_columns(fields)
    if terms:
        print()
        print("reward components (last %d epochs)" % len(shown))
        names = [n for n in TERM_NAMES if n in terms]
        print("%14s %s" % (step_col, " ".join("%9s" % n[:9] for n in names)))
        print("-" * (15 + 10 * len(names)))
        for row in shown:
            s = _num(row, step_col)
            cells = []
            for n in names:
                v = _num(row, terms[n])
                cells.append("%9s" % ("%.4f" % v if v is not None else "-"))
            print("%14s %s" % ("%d" % s if s is not None else "-", " ".join(cells)))
        print()
        print("means over the whole run:")
        for n in names:
            vs = [v for v in (_num(r, terms[n]) for r in rows) if v is not None]
            if vs:
                print("  %-14s %.4f   (last %.4f)  from %s" % (n, sum(vs) / len(vs), vs[-1], terms[n]))

    vals = [v for v in (_num(r, rwd_col) for r in rows) if v is not None]
    if vals:
        print()
        print("reward: first %.4f  last %.4f  best %.4f  (%d epochs)"
              % (vals[0], vals[-1], max(vals), len(vals)))
        if len(vals) >= 20:
            early = sum(vals[:10]) / 10.0
            late = sum(vals[-10:]) / 10.0
            trend = "improving" if late > early else "flat or declining"
            print("mean of first 10 epochs %.4f -> last 10 %.4f  (%s)" % (early, late, trend))

    if plot:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [_num(r, step_col) for r in rows]
        if any(s is None for s in steps):
            steps = list(range(len(rows)))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, [_num(r, rwd_col) for r in rows], lw=1.2, color="#378ADD")
        ax.set_xlabel(step_col or "epoch")
        ax.set_ylabel(rwd_col)
        ax.set_title("Reward: %s" % path.parent.name, fontsize=10)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        out = REPO_V4 / "notebooks" / "figures" / "training_reward.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print()
        print("plot written to", out)
        if show:
            plt.show()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run directory or csv path")
    ap.add_argument("--tail", type=int, default=25, help="rows to print; 0 for all")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--show", action="store_true", help="display the plot window")
    ap.add_argument("--follow", action="store_true", help="re-read until interrupted")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--list", action="store_true", help="list logs and exit")
    ap.add_argument("--columns", action="store_true", help="dump every column name and exit")
    ap.add_argument("--root", default=None, help="extra directory to search")
    args = ap.parse_args()

    roots = list(DEFAULT_ROOTS)
    if args.root:
        roots.insert(0, Path(args.root))
    env_root = os.environ.get("SCONE_RESULTS")
    if env_root:
        roots.insert(0, Path(env_root))

    if args.run:
        target = Path(args.run)
        if target.is_dir():
            candidates = sorted(
                target.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if not candidates:
                print("no csv found under", target)
                return 1
            log = candidates[0]
        else:
            log = target
    else:
        logs = find_logs(roots)
        if args.list:
            if not logs:
                print("no logs found. Searched:")
                for r in roots:
                    print("   ", r, "(exists)" if r.is_dir() else "(missing)")
                return 1
            print("logs found, newest first:")
            for p in logs[:20]:
                print("   %s  %s" % (time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)), p))
            return 0
        if not logs:
            print("no training logs found. Searched:")
            for r in roots:
                print("   ", r, "(exists)" if r.is_dir() else "(missing)")
            print()
            print("Pass --run <dir>, or set SCONE_RESULTS to the results directory.")
            return 1
        log = logs[0]

    if args.columns:
        fields, rows = read_csv(log)
        print("log:", log)
        print("rows:", len(rows))
        print()
        last = rows[-1] if rows else {}
        for f in fields:
            print("  %-44s %s" % (f, last.get(f, "")))
        print()
        matched = find_term_columns(fields)
        print("reward components recognised:", matched or "(none)")
        return 0

    if not args.follow:
        return report(log, args.tail, args.plot, args.show)

    print("following %s every %.0fs; Ctrl+C to stop" % (log, args.interval))
    seen = -1
    try:
        while True:
            _, rows = read_csv(log)
            if len(rows) != seen:
                seen = len(rows)
                print()
                report(log, args.tail, False, False)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        print("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
