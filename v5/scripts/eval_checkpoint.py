"""Run a trained checkpoint and print the reward detail of the best episode.

    python scripts/eval_checkpoint.py
    python scripts/eval_checkpoint.py --episodes 40 --every 10
    python scripts/eval_checkpoint.py --ckpt runs/W-p4-260930.090758/ckpt_latest.pt

Runs `--episodes` episodes and prints the per-step reward breakdown of the one
with the highest return. Nothing else.

Safe to run against a training job: the checkpoint is copied before loading,
so this process never holds the trainer's file open.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from myo_curriculum.env import MyoLocomotionEnv  # noqa: E402
from myo_curriculum.policy import PhaseGatedActorCritic, gait_state_dim  # noqa: E402


def newest_checkpoint(root: Path) -> Path:
    runs = sorted(root.glob("*/ckpt_latest.pt"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(
            "no checkpoints under %s -- pass --ckpt explicitly" % root
        )
    return runs[-1]


def load_checkpoint(path: Path, attempts: int = 5):
    """Load a copy, so the trainer's own file is never held open.

    Loading in place is not safe against a running job on Windows: torch.load
    keeps a mapped section on the file and the trainer's next save then fails
    with ERROR_USER_MAPPED_FILE (1224).
    """
    last = None
    for i in range(attempts):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
                tmp = Path(fh.name)
            shutil.copy2(path, tmp)
            return torch.load(tmp, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - torch and shutil both raise
            last = exc
            time.sleep(0.4 * (i + 1))
        finally:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    raise RuntimeError("could not read %s (%r)" % (path, last))


def contributions(env, term_values):
    """Each term's contribution to the step reward, and its maximum."""
    spec = env.stage_spec.reward
    weights = spec.active_weights
    total_w = sum(weights.values()) or 1.0
    out = [("alive", spec.alive, spec.alive)]
    for name in sorted(weights):
        cap = spec.shaping_scale * weights[name] / total_w
        out.append((name, cap * term_values.get(name, 0.0), cap))
    return out


def pct(value, cap):
    return 100.0 * value / cap if cap > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--runs", type=Path, default=V5 / "runs")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--every", type=int, default=1, help="print every Nth step")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--stage", default=None, help="override the checkpoint's stage")
    ap.add_argument("--sample", action="store_true",
                    help="sample actions instead of using the distribution mean")
    args = ap.parse_args(argv)

    ckpt_path = args.ckpt or newest_checkpoint(args.runs)
    ckpt = load_checkpoint(ckpt_path)
    saved = ckpt.get("args", {})
    stage = args.stage or saved.get("stage", "W")
    phases = int(saved.get("phases", 4))
    tau = float(saved.get("tau_end", 0.2))

    env = MyoLocomotionEnv(stage=stage, seed=args.seed)
    net = PhaseGatedActorCritic(
        env.observation_space.shape[0], env.n_act,
        gait_state_dim(env.obs_layout()), n_phases=phases,
    )
    net.load_state_dict(ckpt["net"])
    net.eval()

    max_step = env.stage_spec.reward.alive + env.stage_spec.reward.shaping_scale
    max_terminal = env.stage_spec.forward_bonus * env.stage_spec.episode_steps
    max_episode = max_step * env.stage_spec.episode_steps + max_terminal

    best = None
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = net.initial_state(1)
        reset_flag = torch.ones(1)
        record, total = [], 0.0
        while True:
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, state = net.act(
                    o, state, reset_flag, tau=tau, deterministic=not args.sample
                )
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
            total += reward
            record.append({
                "step": env.steps, "reward": reward,
                "terms": dict(env.term_values), "terminal": env.terminal_bonus,
            })
            reset_flag = torch.zeros(1)
            if terminated or truncated:
                break
        if best is None or total > best["return"]:
            best = {"seed": args.seed + ep, "return": total,
                    "length": env.steps, "travel": env.travel,
                    "terminal": env.terminal_bonus, "record": record}

    caps = contributions(env, {})
    print("best of %d episodes (seed %d): return %.2f of %.0f, %d steps"
          % (args.episodes, best["seed"], best["return"], max_episode, best["length"]))
    print("max per step: " + "   ".join("%s %.2f" % (n, c) for n, _, c in caps)
          + "   =  %.2f" % max_step)
    print("-" * 96)
    for row in best["record"]:
        if row["step"] % args.every and row is not best["record"][-1]:
            continue
        parts = "  ".join(
            "%s %.3f (%5.1f%%)" % (name, value, pct(value, cap))
            for name, value, cap in contributions(env, row["terms"])
        )
        shown = row["reward"] - row["terminal"]
        print("step %4d: reward %+.4f (%5.1f%% of max): %s"
              % (row["step"], shown, pct(shown, max_step), parts))
    print("-" * 96)
    if max_terminal > 0:
        print("forward progress at the end: travel %+.4f m of %.2f m -> %.2f (%.1f%% of %.0f)"
              % (best["travel"], env.stage_spec.forward_target_distance,
                 best["terminal"], pct(best["terminal"], max_terminal), max_terminal))
    print("return %.2f = %.2f per-step + %.2f terminal  (%.1f%% of %.0f)"
          % (best["return"], best["return"] - best["terminal"], best["terminal"],
             pct(best["return"], max_episode), max_episode))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
