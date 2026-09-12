"""Evaluate a trained checkpoint and report the reward terms.

Why this exists: deprl's per-component logging is not usable for this project.
It pre-allocates rwd_metrics from sconegym's canonical reward names -- constr,
gaussian_vel, grf, number_muscles, self_contact, smooth -- all of which are 0.0
for a torque-actuated model with no muscles. The reward terms this curriculum
actually uses (height, posture, crutch, velocity, backward, displacement) never
reach the CSV. episode_score and episode_length in the log are correct, but the
breakdown is not.

So: load a checkpoint, run episodes, read env.term_values directly.

    python scripts/eval_checkpoint.py <checkpoint> --stage B
    python scripts/eval_checkpoint.py <checkpoint> --stage B --episodes 20 --plot

<checkpoint> is a deprl checkpoint path such as
  .../crutch_v4_stage_A_stand/<run>/checkpoints/step_5600000
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to step_XXXXXXX (no extension)")
    ap.add_argument("--stage", default="A")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-strict-crutch", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--store", action="store_true", help="write SCONE result files")
    args = ap.parse_args()

    import deprl

    env = gym.make(
        scv4.env_id_for(args.stage), strict_crutch=not args.no_strict_crutch
    )
    u = env.unwrapped
    spec = u.stage_spec
    limit = int(args.max_steps or spec.episode_steps)

    print("=" * 78)
    print("checkpoint :", args.checkpoint)
    print("stage      :", spec.describe())
    print("episodes   :", args.episodes)
    if spec.rsi is not None:
        print("RSI        : velocity_scale %.2f, posture ref %s"
              % (spec.rsi.velocity_scale, spec.rsi.posture_reference))
    print("=" * 78)

    policy = deprl.load(args.checkpoint, environment=env)

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
