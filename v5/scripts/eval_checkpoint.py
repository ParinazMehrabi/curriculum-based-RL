"""Run a trained checkpoint and print the best episode, step by step.

    python scripts/eval_checkpoint.py
    python scripts/eval_checkpoint.py --episodes 40 --every 10
    python scripts/eval_checkpoint.py --ckpt runs/W-p4-260930.090758/ckpt_latest.pt

Reads only the checkpoint file, so it is safe to run against a training job
that is still going.

Every episode is recorded; only the **best by return** is printed in full.
That is deliberate -- with an early policy the spread across episodes is the
interesting part, and the mean episode is mostly the failure mode rather than
the behaviour. The summary table shows all of them.
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
from myo_curriculum.policy import (  # noqa: E402
    PhaseGatedActorCritic,
    gait_state_dim,
    phase_alignment,
)


def newest_checkpoint(root: Path) -> Path:
    runs = sorted(
        (p for p in root.glob("*/ckpt_latest.pt")),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(
            "no checkpoints under %s -- pass --ckpt explicitly" % root
        )
    return runs[-1]


def load_checkpoint(path: Path, attempts: int = 5):
    """Load a copy, so the trainer's own file is never held open.

    Loading the checkpoint in place is not safe against a running job on
    Windows: `torch.load` keeps a mapped section on the file, and the
    trainer's next `torch.save` then fails with `ERROR_USER_MAPPED_FILE`
    (1224), which used to take the whole run down with it. Copying first means
    this process only ever touches the original for the duration of a read.
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
    raise RuntimeError(
        "could not read %s after %d attempts (%r).\nIf training is running, "
        "try again in a moment -- the file is rewritten periodically."
        % (path, attempts, last)
    )


def term_contributions(env, term_values):
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


def why_ended(env, terminated):
    if not terminated:
        return "episode length reached"
    reasons = []
    if env.pelvis_height < env.stage_spec.min_pelvis_height:
        reasons.append("pelvis below %.2f m" % env.stage_spec.min_pelvis_height)
    if env.trunk_tilt() > env.stage_spec.max_trunk_tilt:
        reasons.append("trunk tilt above %.2f rad" % env.stage_spec.max_trunk_tilt)
    if (env.reference is not None
            and env.stage_spec.max_tracking_error is not None
            and env.tracking_error > env.stage_spec.max_tracking_error):
        reasons.append(
            "tracking error above %.2f rad" % env.stage_spec.max_tracking_error
        )
    return "; ".join(reasons) or "unknown"


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

    print("=" * 108)
    print("checkpoint : %s  (iteration %s)" % (ckpt_path, ckpt.get("iteration", "?")))
    print(env.describe())
    print(env.stage_spec.describe())
    print("policy     : %d phases, tau %.2f, %s actions"
          % (phases, tau, "sampled" if args.sample else "deterministic"))
    print("max        : %.2f per step + %.0f terminal = %.0f per episode"
          % (max_step, max_terminal, max_episode))
    print("=" * 108)

    episodes = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = net.initial_state(1)
        reset_flag = torch.ones(1)
        record, total = [], 0.0
        while True:
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, value, belief, state = net.act(
                    o, state, reset_flag, tau=tau, deterministic=not args.sample
                )
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
            total += reward
            record.append({
                "step": env.steps,
                "reward": reward,
                "terms": dict(env.term_values),
                "travel": env.travel,
                "velocity": env.forward_velocity(),
                "phase": env.ref_phase,
                "error": env.tracking_error,
                "belief": belief.squeeze(0).numpy().copy(),
                "value": float(value.item()),
                "terminal": env.terminal_bonus,
            })
            reset_flag = torch.zeros(1)
            if terminated or truncated:
                break
        episodes.append({
            "index": ep, "seed": args.seed + ep, "return": total,
            "length": env.steps, "travel": env.travel,
            "terminal": env.terminal_bonus, "terminated": terminated,
            "why": why_ended(env, terminated), "record": record,
        })

    print()
    print("all %d episodes, worst to best" % len(episodes))
    print("-" * 108)
    print("  %5s %6s %9s %9s %10s   %s"
          % ("seed", "steps", "return", "travel m", "terminal", "ended by"))
    for e in sorted(episodes, key=lambda d: d["return"]):
        print("  %5d %6d %9.2f %+9.3f %10.2f   %s"
              % (e["seed"], e["length"], e["return"], e["travel"],
                 e["terminal"], e["why"]))
    rets = np.array([e["return"] for e in episodes])
    lens = np.array([e["length"] for e in episodes])
    print("-" * 108)
    print("  return %.2f +- %.2f   length %.1f +- %.1f   travel %+.3f m"
          % (rets.mean(), rets.std(), lens.mean(), lens.std(),
             float(np.mean([e["travel"] for e in episodes]))))

    best = max(episodes, key=lambda d: d["return"])
    print()
    print("=" * 108)
    print("BEST EPISODE  seed %d  |  return %.2f (%.1f%% of %.0f)  |  %d steps (%.2f s)"
          % (best["seed"], best["return"], pct(best["return"], max_episode),
             max_episode, best["length"], best["length"] * env.dt))
    print("=" * 108)

    caps = term_contributions(env, {})
    print("per step: " + "   ".join("%s max %.2f" % (n, c) for n, _, c in caps))
    print("-" * 108)
    for row in best["record"]:
        if row["step"] % args.every and row is not best["record"][-1]:
            continue
        parts = "  ".join(
            "%s %.3f (%5.1f%%)" % (name, value, pct(value, cap))
            for name, value, cap in term_contributions(env, row["terms"])
        )
        shown = row["reward"] - row["terminal"]
        belief = " ".join("%.2f" % b for b in row["belief"])
        print(
            "step %4d: reward %+.4f (%5.1f%%): %s | travel %+.3f m  v %+.3f m/s | "
            "phase %.3f err %.3f | b [%s] -> %d"
            % (row["step"], shown, pct(shown, max_step), parts, row["travel"],
               row["velocity"], row["phase"], row["error"], belief,
               int(np.argmax(row["belief"])))
        )

    print("-" * 108)
    print("ended by: %s" % best["why"])
    if max_terminal > 0:
        print("forward progress paid at the end: travel %+.4f m of %.2f m -> "
              "%.2f (%.1f%% of %.0f)"
              % (best["travel"], env.stage_spec.forward_target_distance,
                 best["terminal"], pct(best["terminal"], max_terminal), max_terminal))

    # term means over the best episode
    names = sorted({k for row in best["record"] for k in row["terms"]})
    print()
    print("term means over the best episode:")
    for name in names:
        values = np.array([row["terms"].get(name, 0.0) for row in best["record"]])
        flag = ""
        if values.mean() > 0.99:
            flag = "   <- saturated, no gradient"
        elif values.mean() < 0.02:
            flag = "   <- pinned, no gradient"
        print("   %-10s mean %.4f   min %.4f   max %.4f%s"
              % (name, values.mean(), values.min(), values.max(), flag))

    if phases > 1:
        beliefs = torch.as_tensor(np.stack([r["belief"] for r in best["record"]]))
        phase_t = torch.as_tensor(np.array([r["phase"] for r in best["record"]]))
        align = phase_alignment(beliefs, phase_t, n_bins=phases)
        print()
        print("phase vs reference cycle, best episode (rows = expert, cols = cycle bin)")
        print("a near-permutation means the gate found the gait; flat means it did not")
        for k in range(phases):
            print("   %d  %s" % (k, "  ".join("%.2f" % v for v in align[k])))
        usage = beliefs.mean(0).numpy()
        print("   usage %s | belief entropy %.3f (uniform %.3f)"
              % (" ".join("%.2f" % u for u in usage),
                 float(-(beliefs * beliefs.clamp_min(1e-8).log()).sum(-1).mean()),
                 float(np.log(phases))))

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
