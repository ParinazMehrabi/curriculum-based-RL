"""Evaluate a trained checkpoint and report the reward terms.

Why this exists: deprl's per-component logging is not usable for this project.
It pre-allocates rwd_metrics from sconegym's canonical reward names -- constr,
gaussian_vel, grf, number_muscles, self_contact, smooth -- all of which are 0.0
for a torque-actuated model with no muscles. The reward terms this curriculum
actually uses (height, posture, crutch, velocity, backward, displacement) never
reach the CSV. episode_score and episode_length in the log are correct, but the
breakdown is not.

So: load a checkpoint, run episodes, read env.term_values directly.

    python scripts/eval_checkpoint.py --stage C              # newest stage C run
    python scripts/eval_checkpoint.py --stage C --checkpoint 100000
    python scripts/eval_checkpoint.py <run-dir> --stage B --episodes 20 --plot

<run-dir> is the directory holding config.yaml, e.g.
  .../crutch_v4_stage_A_stand/260912.135956.Rajagopal2015_crutch_2D_..._lumbar

A path to a checkpoint inside that run works too; the run directory is found by
walking up to the nearest config.yaml. deprl.load needs the run directory
because it reads config.yaml to rebuild the agent -- handing it a checkpoint
path fails with "'NoneType' object is not subscriptable" from load_utils, which
is also why v3's evaluate_A0_checkpoint.py could never have worked.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_V4 = Path(__file__).resolve().parents[1]
if str(REPO_V4) not in sys.path:
    sys.path.insert(0, str(REPO_V4))

import numpy as np

import gym
import sconegym  # noqa: F401

import sconegym_crutch_v4 as scv4


# Where deprl writes runs on this project's training machine.
RESULT_ROOTS = (
    Path.home() / "Documents" / "SCONE" / "results",
    Path("C:/Users/FUM Care/Documents/SCONE/results"),
    REPO_V4.parent / "results",
)


def find_latest_run(stage: str) -> Path:
    """Newest run directory for a stage, so the timestamp need not be typed."""
    import os

    roots = list(RESULT_ROOTS)
    env_root = os.environ.get("SCONE_RESULTS")
    if env_root:
        roots.insert(0, Path(env_root))

    needle = "stage_%s" % stage.upper()
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        for config in root.rglob("config.yaml"):
            if needle in str(config.parent):
                candidates.append(config.parent)
    if not candidates:
        raise FileNotFoundError(
            "no run directory found for stage %s. Searched:%s%s"
            % (
                stage.upper(),
                chr(10),
                chr(10).join(
                    "  %s %s" % (r, "(exists)" if r.is_dir() else "(missing)")
                    for r in roots
                ),
            )
        )
    return max(set(candidates), key=lambda d: d.stat().st_mtime)


def resolve_run_dir(path: Path) -> Path:
    """Find the directory holding config.yaml, from any path inside the run.

    deprl.load reads config.yaml to rebuild the agent, so it needs the run
    directory rather than a checkpoint file. Passing the checkpoint path gives
    'NoneType' object is not subscriptable from load_utils.
    """
    path = path.resolve()
    for candidate in [path] + list(path.parents):
        if (candidate / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "no config.yaml found at %s or in any parent directory. Pass the run "
        "directory, e.g. .../crutch_v4_stage_A_stand/<timestamp>.<model>/" % path
    )


def step_number(path: Path) -> int:
    """The integer in step_NNNNNN, for numeric sorting.

    Lexicographic sorting puts step_900000 after step_10000000, which made the
    listing report the wrong latest checkpoint.
    """
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def list_checkpoints(run_dir: Path):
    folder = run_dir / "checkpoints"
    if not folder.is_dir():
        return []
    return sorted(folder.glob("step_*"), key=step_number)


def resolve_checkpoint(run_dir: Path, requested):
    """Validate a checkpoint request against what is actually on disk.

    deprl's load_checkpoint prints "Checkpoint X not found" and returns None
    for an unknown id, then crashes on the None downstream. Checking here gives
    a usable message instead.
    """
    available = list_checkpoints(run_dir)
    if not available:
        raise FileNotFoundError("no checkpoints under %s" % (run_dir / "checkpoints"))
    if requested is None or str(requested).lower() == "last":
        return None  # let the run config pick, which is 'last'
    if str(requested).lower() == "first":
        return str(step_number(available[0]))

    want = str(requested).strip()
    digits = "".join(ch for ch in want if ch.isdigit())
    steps = [step_number(c) for c in available]
    if digits and int(digits) in steps:
        return str(int(digits))

    raise SystemExit(
        "checkpoint %r does not exist in this run.%s"
        "available: %s%s"
        "pass one of those numbers, or 'first' / 'last', or omit --checkpoint."
        % (
            requested,
            chr(10),
            ", ".join(str(s) for s in steps),
            chr(10),
        )
    )


def load_policy(deprl, run_dir: Path, env, checkpoint):
    """deprl.load with the run directory, selecting a checkpoint if supported.

    The trailing separator is load-bearing: deprl's load_checkpoint builds the
    checkpoint folder by string concatenation rather than os.path.join, so
    without it the path becomes "<run>checkpoints" and the listdir fails.
    """
    import inspect

    resolved = resolve_checkpoint(run_dir, checkpoint)

    kwargs = {"environment": env}
    if resolved is not None:
        params = inspect.signature(deprl.load).parameters
        if "checkpoint" in params:
            kwargs["checkpoint"] = resolved
        else:
            print(
                "note: this deprl.load has no 'checkpoint' parameter "
                "(accepts %s), so the run config's choice is used instead"
                % ", ".join(params)
            )

    policy = deprl.load(str(run_dir) + os.sep, **kwargs)
    if policy is None:
        raise SystemExit(
            "deprl.load returned None for %s. The checkpoint exists but could "
            "not be loaded; check the lines deprl printed above." % run_dir
        )
    return policy


def run_episodes(policy, env, episodes, seed, limit, needs_force):
    """Return (lengths, scores, term means per episode, crutch forces)."""
    u = env.unwrapped
    lengths, scores, terms, forces = [], [], defaultdict(list), []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        ep_terms = defaultdict(list)
        score, steps = 0.0, 0
        for _ in range(limit):
            obs, reward, done, _ = env.step(policy(obs))
            score += float(reward)
            steps += 1
            for name, value in u.term_values.items():
                ep_terms[name].append(float(value))
            if needs_force:
                forces.append(u.crutch_contact_force())
            if done:
                break
        lengths.append(steps)
        scores.append(score)
        for name, vals in ep_terms.items():
            terms[name].append(float(np.mean(vals)))
    return lengths, scores, terms, forces


def sweep_phase(deprl, run_dir: Path, args) -> int:
    """Evaluate each window of the reference cycle separately.

    A posture term that is low everywhere is a policy problem. One that is high
    in some windows and near zero in others means those frames are not
    statically holdable at velocity_scale 0, and the task is asking for
    something impossible rather than the policy failing.
    """
    n = int(args.sweep_phase)
    edges = np.linspace(0.0, 1.0, n + 1)
    print("=" * 78)
    print("phase sweep: %d windows, %d episodes each, stage %s"
          % (n, args.episodes, args.stage))
    print("run dir:", run_dir)
    print("=" * 78)
    print("%-14s %8s %9s %9s %9s %9s"
          % ("window", "length", "score", "posture", "crutch", "height"))
    print("-" * 64)

    rows = []
    for i in range(n):
        lo, hi = float(edges[i]), float(edges[i + 1])
        env = gym.make(
            scv4.env_id_for(args.stage),
            strict_crutch=not args.no_strict_crutch,
            rsi_phase_range=(lo, hi),
        )
        spec = env.unwrapped.stage_spec
        limit = int(args.max_steps or spec.episode_steps)
        policy = load_policy(deprl, run_dir, env, args.checkpoint)
        lengths, scores, terms, _ = run_episodes(
            policy, env, args.episodes, args.seed, limit, spec.needs_crutch_force
        )
        env.close()
        row = dict(
            lo=lo,
            hi=hi,
            length=float(np.mean(lengths)),
            score=float(np.mean(scores)),
            **{k: float(np.mean(v)) for k, v in terms.items()},
        )
        rows.append(row)
        print(
            "%.2f-%.2f      %8.1f %9.2f %9.4f %9.4f %9.4f"
            % (lo, hi, row["length"], row["score"],
               row.get("posture", float("nan")),
               row.get("crutch", float("nan")),
               row.get("height", float("nan")))
        )

    postures = [r.get("posture") for r in rows if r.get("posture") is not None]
    if postures:
        best = max(range(len(rows)), key=lambda i: rows[i].get("posture", -1))
        worst = min(range(len(rows)), key=lambda i: rows[i].get("posture", 2))
        print()
        print("posture: best window %.2f-%.2f at %.4f, worst %.2f-%.2f at %.4f"
              % (rows[best]["lo"], rows[best]["hi"], rows[best]["posture"],
                 rows[worst]["lo"], rows[worst]["hi"], rows[worst]["posture"]))
        spread = max(postures) - min(postures)
        print("spread across windows: %.4f" % spread)
        if spread > 0.25:
            print()
            print("Large spread: some windows are far more holdable than others.")
            print("That points at the task, not the policy. Restrict sampling with")
            print("  rsi_phase_range=(%.2f, %.2f)" % (rows[best]["lo"], rows[best]["hi"]))
            print("or switch rsi_posture_reference to 'neutral' so the target is")
            print("always reachable.")
        else:
            print()
            print("Small spread: posture is uniformly low, which is a policy or")
            print("sigma problem rather than unholdable frames. Consider widening")
            print("pelvis_tilt_sigma and lumbar_sigma from 0.12.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "run",
        nargs="?",
        default=None,
        help="the run directory (the one holding config.yaml), or any path "
        "inside it such as a checkpoints/step_XXXXXXX file. Omit it to use the "
        "newest run found for --stage.",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="which checkpoint to load, if deprl.load supports selecting one "
        "(e.g. 'last', or a step number). Default: whatever the run's "
        "config.yaml specifies, which is 'last'.",
    )
    ap.add_argument("--stage", default="A")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-strict-crutch", action="store_true")
    ap.add_argument(
        "--phase-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="restrict RSI sampling to this fraction of the reference cycle",
    )
    ap.add_argument(
        "--sweep-phase",
        type=int,
        default=0,
        metavar="N",
        help="split the cycle into N windows and report each separately. Tests "
        "whether a low posture term means the policy is weak or the frames are "
        "not statically holdable.",
    )
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--store",
        action="store_true",
        help="write SCONE .sto result files for each episode, so the motion can "
        "be played back in SCONE Studio",
    )
    args = ap.parse_args()

    import deprl  # noqa: F401  (imported here so --help works without it)

    if args.run is None:
        run_dir = find_latest_run(args.stage)
        print("using newest stage %s run: %s" % (args.stage.upper(), run_dir.name))
    else:
        run_dir = resolve_run_dir(Path(args.run))

    if args.sweep_phase > 0:
        return sweep_phase(deprl, run_dir, args)

    make_kwargs = {"strict_crutch": not args.no_strict_crutch}
    if args.phase_range is not None:
        make_kwargs["rsi_phase_range"] = tuple(args.phase_range)
    env = gym.make(scv4.env_id_for(args.stage), **make_kwargs)
    u = env.unwrapped
    spec = u.stage_spec
    limit = int(args.max_steps or spec.episode_steps)

    print("=" * 78)
    print("run dir    :", run_dir)
    available = list_checkpoints(run_dir)
    if available:
        print(
            "checkpoints: %d found, %s .. %s (loading the last unless --checkpoint)"
            % (len(available), available[0].name, available[-1].name)
        )
    else:
        print("checkpoints: none found under", run_dir / "checkpoints")
    print("stage      :", spec.describe())
    print("episodes   :", args.episodes)
    if spec.rsi is not None:
        print("RSI        : velocity_scale %.2f, posture ref %s"
              % (spec.rsi.velocity_scale, spec.rsi.posture_reference))
    if args.store:
        print("storing   : SCONE result files -> %s" % getattr(u, "output_dir", "?"))
    print("=" * 78)

    policy = load_policy(deprl, run_dir, env, args.checkpoint)

    per_episode = []
    term_totals = defaultdict(list)
    crutch_force = []

    for ep in range(args.episodes):
        if args.store:
            env.store_next_episode()
        obs = env.reset(seed=args.seed + ep)
        frame = u.rsi_frame
        ep_terms = defaultdict(list)
        score = 0.0
        steps = 0
        for _ in range(limit):
            action = policy(obs)
            obs, reward, done, _info = env.step(action)
            score += float(reward)
            steps += 1
            for name, value in u.term_values.items():
                ep_terms[name].append(float(value))
            if spec.needs_crutch_force:
                crutch_force.append(u.crutch_contact_force())
            if done:
                break
        fell = bool(u._is_fall())
        means = {k: float(np.mean(v)) for k, v in ep_terms.items()}
        for k, v in means.items():
            term_totals[k].append(v)
        per_episode.append(
            dict(ep=ep, frame=frame, steps=steps, score=score, fell=fell, terms=means)
        )
        print(
            "ep=%02d frame=%-5s steps=%4d score=%8.2f per_step=%.4f %s"
            % (
                ep,
                frame if frame is not None else "-",
                steps,
                score,
                score / max(steps, 1),
                "FELL" if fell else "",
            )
        )

    print()
    print("-" * 78)
    steps_all = np.array([e["steps"] for e in per_episode], dtype=float)
    scores = np.array([e["score"] for e in per_episode], dtype=float)
    falls = sum(1 for e in per_episode if e["fell"])
    print("episode length : mean %.1f  min %d  max %d  (cap %d)"
          % (steps_all.mean(), steps_all.min(), steps_all.max(), limit))
    print("episode score  : mean %.2f  std %.2f  min %.2f  max %.2f"
          % (scores.mean(), scores.std(), scores.min(), scores.max()))
    print("per-step reward: mean %.4f" % (scores.sum() / max(steps_all.sum(), 1)))
    print("falls          : %d of %d" % (falls, args.episodes))
    print()
    print("reward terms (mean over all steps of all episodes):")
    weights = spec.reward.active_weights
    for name in sorted(term_totals, key=lambda n: -weights.get(n, 0.0)):
        vals = np.array(term_totals[name], dtype=float)
        print(
            "  %-16s %.4f   (per-episode min %.4f max %.4f)   weight %.2f"
            % (name, vals.mean(), vals.min(), vals.max(), weights.get(name, 0.0))
        )

    if args.store:
        out = Path(getattr(u, "output_dir", "."))
        print()
        print("SCONE result files in %s:" % out)
        if out.is_dir():
            written = sorted(
                out.glob("*.sto"), key=lambda f: f.stat().st_mtime, reverse=True
            )[: args.episodes]
            for f in written:
                print("  %s  (%.1f KB)" % (f.name, f.stat().st_size / 1024.0))
            if written:
                print()
                print("Open the newest in SCONE Studio (File > Open, or double-click)")
                print("and press play. The model comes from the .sto header.")
        else:
            print("  directory not found; check env.output_dir")

    if crutch_force:
        arr = np.asarray(crutch_force, dtype=float)
        bw = u._body_weight_n
        print()
        print("crutch force   : mean %.1f N (%.1f%% BW)  max %.1f N  zero on %.0f%% of steps"
              % (arr.mean(), 100 * arr.mean() / bw, arr.max(), 100 * (arr <= 0).mean()))
        print("crutch target  : %.1f N (%.1f%% BW)"
              % (spec.terms.cane_target_load_fraction * bw,
                 100 * spec.terms.cane_target_load_fraction))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = sorted(term_totals, key=lambda n: -weights.get(n, 0.0))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(names, [np.mean(term_totals[n]) for n in names], color="#378ADD")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mean term value")
        ax.set_title("Reward terms: stage %s, %d episodes" % (args.stage, args.episodes), fontsize=10)
        ax.grid(alpha=0.15, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        out = REPO_V4 / "notebooks" / "figures" / ("eval_terms_stage_%s.png" % args.stage)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print()
        print("plot written to", out)

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
