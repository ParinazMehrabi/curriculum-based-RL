"""Run an episode and print the reward, term by term, every step.

    python scripts/run_reward.py
    python scripts/run_reward.py --stage W --policy low --steps 300
    python scripts/run_reward.py --every 10          # print every 10th step

Each line is the step, the total reward with its share of the maximum, and
then every term's **contribution** to that total with its own share of the
maximum it could contribute. The shares are what make the line readable: a
term at 100% is saturated and is no longer a source of gradient.

For stage W the maxima are the ones the reward was specified with:

    alive     0.10   (0.10 per step, 1.0 over the full 10 s)
    velocity  0.70   (smoothstep to 0.1 m/s)
    tracking  0.20   (reference gait)
    ----------------
    total     1.00
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from myo_curriculum.env import MyoLocomotionEnv  # noqa: E402
from myo_curriculum.stages import STAGE_ORDER  # noqa: E402


def contributions(env):
    """Each term's contribution to the step reward, and its maximum.

    Mirrors RewardSpec.compose: total = alive + shaping_scale * weighted mean
    of the terms, so term i contributes shaping_scale * w_i / sum(w) * value_i.
    """
    spec = env.stage_spec.reward
    weights = spec.active_weights
    total_w = sum(weights.values()) or 1.0
    out = [("alive", spec.alive, spec.alive)]
    for name in sorted(weights):
        cap = spec.shaping_scale * weights[name] / total_w
        out.append((name, cap * env.term_values.get(name, 0.0), cap))
    return out


def pct(value, cap):
    return 100.0 * value / cap if cap > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="W", choices=list(STAGE_ORDER))
    ap.add_argument("--policy", default="low", choices=("zero", "low", "random"))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="print every Nth step")
    args = ap.parse_args(argv)

    env = MyoLocomotionEnv(stage=args.stage, seed=args.seed)
    print("=" * 100)
    print(env.describe())
    print(env.stage_spec.describe())
    caps = contributions(env)
    print("max per step:  " + "   ".join("%s %.2f" % (n, c) for n, _, c in caps)
          + "   |   total %.2f" % sum(c for _, _, c in caps))
    print("=" * 100)

    obs, _ = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    max_step = env.stage_spec.reward.alive + env.stage_spec.reward.shaping_scale
    total = 0.0
    step = 0

    for step in range(1, args.steps + 1):
        if args.policy == "zero":
            action = np.zeros(env.n_act, np.float32)
        elif args.policy == "low":
            action = np.full(env.n_act, -0.6, np.float32)
        else:
            action = rng.uniform(-1, 1, env.n_act).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total += reward

        if step % args.every == 0 or terminated or truncated:
            parts = "  ".join(
                "%s %.3f (%5.1f%%)" % (name, value, pct(value, cap))
                for name, value, cap in contributions(env)
            )
            extra = ""
            if env.reference is not None:
                extra = "  | phase %.3f  err %.3f rad" % (env.ref_phase, env.tracking_error)
            print(
                "step %4d: reward %+.4f (%5.1f%% of max): %s  | v %+.3f m/s%s"
                % (step, reward, pct(reward, max_step), parts,
                   env.forward_velocity(), extra)
            )

        if terminated or truncated:
            why = []
            if env.pelvis_height < env.stage_spec.min_pelvis_height:
                why.append("pelvis below %.2f m" % env.stage_spec.min_pelvis_height)
            if env.trunk_tilt() > env.stage_spec.max_trunk_tilt:
                why.append("trunk tilt above %.2f rad" % env.stage_spec.max_trunk_tilt)
            if (env.reference is not None
                    and env.stage_spec.max_tracking_error is not None
                    and env.tracking_error > env.stage_spec.max_tracking_error):
                why.append("tracking error above %.2f rad" % env.stage_spec.max_tracking_error)
            print("-" * 100)
            print("%s at step %d: %s"
                  % ("TERMINATED" if terminated else "truncated (episode length reached)",
                     step, "; ".join(why) or "-"))
            break

    print("-" * 100)
    print("return %.2f over %d steps | mean %.4f/step (%.1f%% of max) | %.2f s simulated"
          % (total, step, total / max(step, 1),
             pct(total / max(step, 1), max_step), step * env.dt))
    print("a perfect episode would score %.1f" % (max_step * env.stage_spec.episode_steps))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
