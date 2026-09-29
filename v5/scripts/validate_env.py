"""Smoke test for the muscle-locomotion environment.

    python v5/scripts/validate_env.py
    python v5/scripts/validate_env.py --stage B --steps 400

Checks the things that are easy to get silently wrong on a musculoskeletal
model and expensive to discover after a training run:

* that the anatomical frame is right (an earlier version read a body's local
  +z as "up" on a model whose frames are locally y-up, measured an 89-degree
  trunk tilt on a model standing straight, and terminated every episode on
  step 1),
* that muscle activation state is in the observation, without which the MDP
  is not Markov,
* that the reward cannot go negative on a non-terminal step, and
* that every reward term has usable gradient from where the policy starts --
  a velocity target unreachable from standstill is a silent dead end.

Exit code 0 when everything passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from myo_curriculum.env import INDEPENDENT_JOINTS, MyoLocomotionEnv  # noqa: E402
from myo_curriculum.stages import STAGE_ORDER, get_stage  # noqa: E402


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if not ok:
            self.failures += 1
        print("  %-5s %-44s %s" % ("ok" if ok else "FAIL", label, detail))
        return ok

    def note(self, label: str, detail: str = "") -> None:
        print("  %-5s %-44s %s" % ("", label, detail))


def section(title: str) -> None:
    print()
    print(title)
    print("-" * 78)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default=None, choices=list(STAGE_ORDER))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    stages = [args.stage] if args.stage else list(STAGE_ORDER)
    r = Report()

    for key in stages:
        env = MyoLocomotionEnv(stage=key, seed=args.seed)
        print("=" * 78)
        print(env.describe())
        print("=" * 78)
        print("  %s" % env.stage_spec.describe())

        section("1. model and spaces")
        r.check(env.n_muscle > 0, "model is muscle-actuated", "%d muscles" % env.n_muscle)
        r.check(
            env.model.na == env.n_muscle,
            "one activation state per muscle",
            "na=%d" % env.model.na,
        )
        r.check(
            len(env.joint_qpos_adr) == len(INDEPENDENT_JOINTS),
            "independent joints resolved",
            "%d of %d model joints" % (len(env.joint_qpos_adr), env.model.njnt),
        )
        layout = env.obs_layout()
        r.check(
            sum(w for _, w in layout) == env.observation_space.shape[0],
            "observation layout sums to the space",
            "%d dims" % env.observation_space.shape[0],
        )
        r.check(
            any(n == "muscle_activation" for n, _ in layout),
            "muscle activation is observed (Markov)",
            "%d dims" % env.model.na,
        )
        r.note("layout", ", ".join("%s=%d" % (n, w) for n, w in layout))

        section("2. posture frame")
        env.reset(seed=args.seed)
        tilt = env.trunk_tilt()
        r.check(
            tilt < 0.4,
            "trunk is near vertical at reset",
            "%.3f rad (%.1f deg)" % (tilt, np.degrees(tilt)),
        )
        r.check(
            not env._is_fallen(),
            "reset pose is not already terminal",
            "pelvis %.3f m" % env.pelvis_height,
        )
        r.check(
            abs(env.heading_error()) < 1e-6,
            "heading error is zero at reset",
            "ref %.1f deg" % np.degrees(env._heading_ref),
        )
        fwd, lat = env.planar_velocity()
        r.note("planar velocity at reset", "forward %+.3f  lateral %+.3f m/s" % (fwd, lat))

        section("3. reward safety")
        report = env.stage_spec.reward.termination_report(gamma=0.99)
        r.check(
            float(report["min_step_reward"]) >= 0.0,
            "per-step reward cannot go negative",
            "min %.4f" % float(report["min_step_reward"]),
        )
        r.check(
            not report["termination_preferred"],
            "falling is never the better option",
            str(report["verdict"]),
        )

        section("4. reward-term gradient from the start state")
        print("  A term pinned at its floor gives the policy nothing to climb.")
        env.reset(seed=args.seed)
        start_terms = env.compute_terms()
        for name, value in sorted(start_terms.items()):
            flag = "  <-- no gradient" if value < 0.02 else ""
            print("    %-10s %.4f%s" % (name, value, flag))
        r.check(
            all(v >= 0.02 for v in start_terms.values()),
            "every term is off its floor at reset",
            "min %.4f" % min(start_terms.values()),
        )

        section("5. rollouts")
        rng = np.random.default_rng(args.seed)
        for policy in ("zero", "random", "low"):
            env.reset(seed=args.seed + 1)
            total = 0.0
            worst = np.inf
            t0 = time.perf_counter()
            steps = 0
            for _ in range(args.steps):
                if policy == "zero":
                    a = np.zeros(env.n_act, np.float32)
                elif policy == "low":
                    a = np.full(env.n_act, -0.6, np.float32)
                else:
                    a = rng.uniform(-1, 1, env.n_act).astype(np.float32)
                _, rew, term, trunc, _ = env.step(a)
                steps += 1
                total += rew
                if not term:
                    worst = min(worst, rew)
                if term or trunc:
                    break
            dt = time.perf_counter() - t0
            print(
                "    %-7s %3d steps  return %8.2f  min step reward %+.4f  "
                "%4.0f steps/s  %s"
                % (policy, steps, total, worst, steps / max(dt, 1e-9),
                   "terminated" if term else "ran out")
            )
            r.check(worst >= -1e-9, "  %s: no negative non-terminal reward" % policy)

        env.close()

    print()
    print("=" * 78)
    print("FAIL -- %d check(s) failed" % r.failures if r.failures else "PASS")
    print("=" * 78)
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
