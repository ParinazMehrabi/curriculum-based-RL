"""Run one episode of each curriculum stage and plot the rewards.

The logic lives here rather than in the notebook so there is one
implementation; notebooks/run_stages.ipynb just calls these functions.

CPU budget: thread-count environment variables are set at import time, before
numpy is imported, so they take effect on the BLAS backend. Import this module
before numpy if you can. `set_cpu_budget` can tighten it further at runtime.

    python scripts/run_stages.py                 # all stages, zero-action policy
    python scripts/run_stages.py --policy random --seed 1
    python scripts/run_stages.py --stages A B --no-strict-crutch
"""
from __future__ import annotations

import os

# --- CPU budget, before numpy pulls in its BLAS -----------------------------
_TOTAL_CPUS = os.cpu_count() or 2
_HALF_CPUS = max(1, _TOTAL_CPUS // 2)
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, str(_HALF_CPUS))

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_V4 = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_V4 / "notebooks" / "figures"
RESULT_DIR = REPO_V4 / "notebooks" / "results"

# Make `import sconegym_crutch_v4` work however this file was invoked: as a
# script, as a module, or from the notebook one directory down.
if str(REPO_V4) not in sys.path:
    sys.path.insert(0, str(REPO_V4))

STAGE_COLOURS = {"A": "#378ADD", "B": "#1D9E75", "C": "#BA7517", "D": "#D4537E"}


# --- cpu -------------------------------------------------------------------


def set_cpu_budget(fraction: float = 0.5, verbose: bool = True) -> Dict[str, object]:
    """Restrict this process to `fraction` of the logical CPUs.

    Sets BLAS/OpenMP thread counts, torch's thread count when torch is present,
    and process affinity when psutil is available. Affinity is the only one of
    those that is a hard cap; the rest are cooperative.
    """
    total = os.cpu_count() or 2
    n = max(1, int(total * float(fraction)))

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(n)

    affinity = None
    affinity_note = "not applied (install psutil for a hard cap)"
    try:
        import psutil

        proc = psutil.Process()
        proc.cpu_affinity(list(range(n)))
        affinity = proc.cpu_affinity()
        affinity_note = "applied via psutil"
    except ImportError:
        pass
    except Exception as exc:
        affinity_note = "psutil present but affinity failed: %r" % (exc,)

    torch_threads = None
    try:
        import torch

        torch.set_num_threads(n)
        torch_threads = torch.get_num_threads()
    except ImportError:
        pass
    except Exception:
        pass

    info = {
        "logical_cpus": total,
        "budget": n,
        "thread_env": n,
        "affinity": affinity,
        "affinity_note": affinity_note,
        "torch_threads": torch_threads,
    }
    if verbose:
        print("CPU budget: %d of %d logical cores (%.0f%%)" % (n, total, 100 * n / total))
        print("  thread env vars : %d" % n)
        print("  affinity        : %s" % affinity_note)
        if affinity is not None:
            print("  affinity mask   : %s" % affinity)
        if torch_threads is not None:
            print("  torch threads   : %d" % torch_threads)
    return info


# --- rollout ---------------------------------------------------------------


def make_policy(kind: str, seed: int, n_act: int = 9, amplitude: float = 0.25):
    """Return a callable obs -> action.

    There is no trained v4 checkpoint (the observation layout changed), so these
    rollouts are diagnostics of the environment and the reward, not evaluations
    of a policy. Use kind="checkpoint:<path>" once a v4 run exists.
    """
    if kind == "zeros":
        zero = np.zeros(n_act, dtype=np.float32)
        return lambda obs: zero
    if kind == "random":
        rng = np.random.RandomState(seed)
        return lambda obs: rng.uniform(-amplitude, amplitude, n_act).astype(np.float32)
    if kind.startswith("checkpoint:"):
        path = kind.split(":", 1)[1]

        def _load(env):
            import deprl

            return deprl.load(path, environment=env)

        return _load
    raise ValueError("unknown policy %r (zeros, random, checkpoint:<path>)" % kind)


def rollout(
    stage: str,
    policy: str = "zeros",
    seed: int = 0,
    max_steps: Optional[int] = None,
    strict_crutch: bool = True,
    amplitude: float = 0.25,
) -> Dict[str, object]:
    """Run one episode of one stage. Never raises; failures are reported."""
    out: Dict[str, object] = {"stage": stage, "policy": policy, "seed": seed}

    try:
        import gym
        import sconegym  # noqa: F401
        import sconegym_crutch_v4 as scv4
    except ImportError as exc:
        out.update(
            ok=False,
            error="missing dependency: %s (needs gym + sconegym + a licensed sconepy)"
            % exc,
            traceback=traceback.format_exc(),
        )
        return out

    try:
        env = gym.make(scv4.env_id_for(stage), strict_crutch=strict_crutch)
    except Exception as exc:
        out.update(
            ok=False,
            error="construction failed: %s" % exc,
            traceback=traceback.format_exc(),
        )
        return out

    u = env.unwrapped
    spec = u.stage_spec
    limit = int(max_steps or spec.episode_steps)

    pol = make_policy(policy, seed)
    if policy.startswith("checkpoint:"):
        pol = pol(env)

    rewards: List[float] = []
    terms: Dict[str, List[float]] = {}
    crutch_force: List[float] = []

    try:
        obs = env.reset(seed=seed)
        done = False
        step = 0
        while not done and step < limit:
            action = pol(obs)
            obs, reward, done, _info = env.step(action)
            rewards.append(float(reward))
            for name, value in u.term_values.items():
                terms.setdefault(name, []).append(float(value))
            if spec.needs_crutch_force:
                crutch_force.append(float(u.crutch_contact_force()))
            step += 1
        fell = bool(u._is_fall())
    except Exception as exc:
        out.update(
            ok=False,
            error="rollout failed at step %d: %s" % (len(rewards), exc),
            traceback=traceback.format_exc(),
            rewards=rewards,
            terms=terms,
        )
        env.close()
        return out

    env.close()
    arr = np.asarray(rewards, dtype=float)
    out.update(
        ok=True,
        name=spec.name,
        steps=len(rewards),
        episode_steps=spec.episode_steps,
        fell=fell,
        total_reward=float(arr.sum()),
        mean_reward=float(arr.mean()) if arr.size else 0.0,
        min_reward=float(arr.min()) if arr.size else 0.0,
        max_reward=float(arr.max()) if arr.size else 0.0,
        negative_steps=int((arr < 0).sum()),
        rewards=rewards,
        cumulative=np.cumsum(arr).tolist(),
        terms=terms,
        crutch_force=crutch_force,
        crutch_force_max=float(max(crutch_force)) if crutch_force else None,
        weights=dict(spec.reward.active_weights),
    )
    return out


def run_all(
    stages: Sequence[str] = ("A", "B", "C", "D"),
    policy: str = "zeros",
    seed: int = 0,
    max_steps: Optional[int] = None,
    strict_crutch: bool = True,
) -> Dict[str, Dict[str, object]]:
    results = {}
    for stage in stages:
        print("running stage %s ..." % stage, end=" ", flush=True)
        res = rollout(
            stage,
            policy=policy,
            seed=seed,
            max_steps=max_steps,
            strict_crutch=strict_crutch,
        )
        results[stage] = res
        if res.get("ok"):
            print(
                "%d steps, total %.2f, mean %.4f%s"
                % (
                    res["steps"],
                    res["total_reward"],
                    res["mean_reward"],
                    ", FELL" if res["fell"] else "",
                )
            )
        else:
            print("FAILED -- %s" % res.get("error"))
    return results


def summarise(results: Dict[str, Dict[str, object]]) -> str:
    lines = [
        "%-6s %-18s %7s %10s %9s %9s %9s %8s"
        % ("stage", "name", "steps", "total", "mean", "min", "max", "fell"),
        "-" * 88,
    ]
    for key, res in results.items():
        if not res.get("ok"):
            lines.append("%-6s %s" % (key, "FAILED: %s" % res.get("error")))
            continue
        lines.append(
            "%-6s %-18s %7d %10.2f %9.4f %9.4f %9.4f %8s"
            % (
                key,
                res["name"],
                res["steps"],
                res["total_reward"],
                res["mean_reward"],
                res["min_reward"],
                res["max_reward"],
                "yes" if res["fell"] else "no",
            )
        )
    bad = [k for k, r in results.items() if r.get("ok") and r["negative_steps"]]
    if bad:
        lines.append("")
        lines.append("WARNING: negative step rewards in stages %s -- should be impossible" % bad)
    return "\n".join(lines)


# --- plots -----------------------------------------------------------------


def _ok(results):
    return {k: v for k, v in results.items() if v.get("ok") and v.get("rewards")}


def plot_rewards(results, save_dir: Optional[Path] = None, show: bool = True):
    """Per-step reward and cumulative reward, all stages overlaid."""
    import matplotlib.pyplot as plt

    good = _ok(results)
    if not good:
        print("nothing to plot: no stage produced a rollout")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for key, res in good.items():
        colour = STAGE_COLOURS.get(key, None)
        label = "%s (%s)" % (key, res["name"])
        axes[0].plot(res["rewards"], lw=0.9, color=colour, label=label)
        axes[1].plot(res["cumulative"], lw=1.4, color=colour, label=label)

    axes[0].set_ylabel("reward per step")
    axes[0].set_title("Per-step reward, one episode per stage")
    axes[0].axhline(0.0, color="#888780", lw=0.6, ls="--")
    axes[0].set_ylim(bottom=min(-0.05, min(r["min_reward"] for r in good.values()) - 0.05))
    axes[0].legend(loc="lower left", fontsize=9, frameon=False)
    axes[0].grid(alpha=0.15)

    axes[1].set_ylabel("cumulative reward")
    axes[1].set_xlabel("step")
    axes[1].set_title("Cumulative reward")
    axes[1].grid(alpha=0.15)

    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "stage_rewards.png", dpi=150)
    if show:
        plt.show()
    return fig


def plot_terms(results, save_dir: Optional[Path] = None, show: bool = True):
    """One panel per stage showing each reward term's trace."""
    import matplotlib.pyplot as plt

    good = _ok(results)
    if not good:
        return None

    n = len(good)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True, squeeze=False)
    for ax, (key, res) in zip(axes[:, 0], good.items()):
        for name, series in sorted(res["terms"].items()):
            ax.plot(series, lw=0.9, label=name)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("term value")
        ax.set_title("Stage %s (%s) -- reward terms" % (key, res["name"]), fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=3, frameon=False)
        ax.grid(alpha=0.15)
    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "stage_terms.png", dpi=150)
    if show:
        plt.show()
    return fig


def plot_summary(results, save_dir: Optional[Path] = None, show: bool = True):
    """Episode length and mean reward per stage."""
    import matplotlib.pyplot as plt

    good = _ok(results)
    if not good:
        return None

    keys = list(good)
    colours = [STAGE_COLOURS.get(k, "#888780") for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))

    axes[0].bar(keys, [good[k]["steps"] for k in keys], color=colours)
    axes[0].set_title("Episode length (steps)")
    axes[0].set_ylabel("steps")
    for i, k in enumerate(keys):
        if good[k]["fell"]:
            axes[0].text(i, good[k]["steps"], "fell", ha="center", va="bottom", fontsize=8)

    axes[1].bar(keys, [good[k]["mean_reward"] for k in keys], color=colours)
    axes[1].set_title("Mean reward per step (max 1.0)")
    axes[1].set_ylabel("mean reward")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1.0, color="#888780", lw=0.6, ls="--")

    for ax in axes:
        ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "stage_summary.png", dpi=150)
    if show:
        plt.show()
    return fig


def plot_gating(
    stages: Sequence[str] = ("A", "B", "C", "D"),
    save_dir: Optional[Path] = None,
    show: bool = True,
):
    """How each stage's reward responds as one term degrades.

    Needs no simulator: it drives RewardSpec.compose directly. All terms are
    held at 1.0 while one is swept from 1.0 down to 0.0, which shows how much
    reward a policy keeps by abandoning a single objective. A flat line is a
    term the policy can ignore; a steep one is a term that gates.
    """
    import matplotlib.pyplot as plt

    # Import the stage table without triggering the package __init__ (no gym).
    import importlib
    import types

    pkg_name = "_scv4_gating"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(REPO_V4 / "sconegym_crutch_v4")]
        sys.modules[pkg_name] = pkg
    stages_mod = importlib.import_module(pkg_name + ".stages")

    keys = [k for k in stages if k in stages_mod.STAGES]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6), sharey=True, squeeze=False)
    sweep = np.linspace(1.0, 0.0, 41)

    for ax, key in zip(axes[0], keys):
        stage = stages_mod.STAGES[key]
        spec = stage.reward
        names = sorted(spec.active_weights, key=lambda n: -spec.active_weights[n])
        for name in names:
            ys = []
            for v in sweep:
                terms = {t: 1.0 for t in spec.required_terms}
                terms[name] = float(v)
                total, _ = spec.compose(terms)
                ys.append(total)
            ax.plot(sweep, ys, lw=1.3, label=name)
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("term value")
        ax.set_title("Stage %s (%d terms)" % (key, len(names)), fontsize=10)
        ax.legend(fontsize=6.5, loc="lower left", frameon=False)
        ax.grid(alpha=0.15)
    axes[0][0].set_ylabel("step reward")
    fig.suptitle(
        "Reward kept when a single term degrades (others held ideal)", fontsize=11
    )
    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "stage_gating.png", dpi=150)
    if show:
        plt.show()
    return fig


def save_results(results, path: Optional[Path] = None) -> Path:
    path = Path(path or (RESULT_DIR / "rollouts.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {}
    for key, res in results.items():
        slim[key] = {k: v for k, v in res.items() if k != "traceback"}
    path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return path


# --- cli -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--policy", default="zeros", help="zeros | random | checkpoint:<path>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--cpu-fraction", type=float, default=0.5)
    ap.add_argument("--no-strict-crutch", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    set_cpu_budget(args.cpu_fraction)
    print()

    results = run_all(
        stages=args.stages,
        policy=args.policy,
        seed=args.seed,
        max_steps=args.max_steps,
        strict_crutch=not args.no_strict_crutch,
    )
    print()
    print(summarise(results))
    print()

    if not _ok(results):
        print("No stage produced a rollout; skipping plots.")
        return 1

    plot_rewards(results, FIGURE_DIR, show=not args.no_show)
    plot_terms(results, FIGURE_DIR, show=not args.no_show)
    plot_summary(results, FIGURE_DIR, show=not args.no_show)
    print("figures written to", FIGURE_DIR)
    print("results written to", save_results(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
