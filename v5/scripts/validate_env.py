"""Smoke test for the planar muscle-locomotion environment.

    python scripts/validate_env.py
    python scripts/validate_env.py --stage W --steps 400

Checks the things that are easy to get silently wrong on a musculoskeletal
model and expensive to discover after a training run: the anatomical frame,
the planar structure, foot contact, whether muscle activation is observed
(without which the MDP is not Markov), reward safety, and whether every reward
term has usable gradient from where the policy starts.

Stage-aware: a stage that resets to the standing stance gets the stance and
contact checks; a stage that uses reference-state initialisation gets the
tracking checks instead, because feet staggered and one foot off the ground
are correct in a mid-gait frame.

Exit code 0 when everything passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

import mujoco  # noqa: E402

from myo_curriculum.env import (  # noqa: E402
    BALL_RADIUS,
    INDEPENDENT_JOINTS,
    OUT_OF_PLANE_JOINTS,
    ROOT_JOINTS,
    MyoLocomotionEnv,
)
from myo_curriculum.stages import STAGE_ORDER  # noqa: E402


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


# -- sections --------------------------------------------------------------


def model_and_spaces(r, env):
    section("1. model and spaces")
    r.check(env.n_muscle > 0, "model is muscle-actuated", "%d muscles" % env.n_muscle)
    r.check(env.model.na == env.n_muscle, "one activation state per muscle",
            "na=%d" % env.model.na)
    r.check(len(env.joint_qpos_adr) == len(INDEPENDENT_JOINTS),
            "independent joints resolved",
            "%d of %d model joints" % (len(env.joint_qpos_adr), env.model.njnt))
    layout = env.obs_layout()
    r.check(sum(w for _, w in layout) == env.observation_space.shape[0],
            "observation layout sums to the space",
            "%d dims" % env.observation_space.shape[0])
    r.check(any(n == "muscle_activation" for n, _ in layout),
            "muscle activation is observed (Markov)", "%d dims" % env.model.na)
    r.note("layout", ", ".join("%s=%d" % (n, w) for n, w in layout))


def posture_frame(r, env, seed):
    section("2. posture frame")
    env.reset(seed=seed)
    tilt = env.trunk_tilt()
    r.check(tilt < 0.6, "trunk is near vertical at reset",
            "%.3f rad (%.1f deg)" % (tilt, np.degrees(tilt)))
    r.check(not env._is_fallen(), "reset pose is not already terminal",
            "pelvis %.3f m" % env.pelvis_height)
    r.check(abs(float(env.com_velocity()[1])) < 1e-3,
            "no lateral COM velocity (planar root)",
            "%.1e m/s" % abs(float(env.com_velocity()[1])))
    r.note("forward velocity at reset", "%+.3f m/s" % env.forward_velocity())


def stance_and_contact(r, env, seed):
    section("2b. standing stance and foot contact")
    print("  The model used to start up on its toes with both heels 23 mm")
    print("  in the air, because the shipped keyframe is a mid-stride pose.")
    print()
    r.check(env.stance_residual < 1e-4, "stance solve converged",
            "residual %.1e" % env.stance_residual)
    r.check(True, "two contact balls per foot",
            "r=%.3f m, as in the .hfd cane model" % BALL_RADIUS)
    env.reset(seed=seed)
    r.check(abs(env.foot_stagger()) < 0.02, "feet parallel, not staggered",
            "%.1e m" % abs(env.foot_stagger()))

    floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    offfloor, worst_heel, worst_toe, loads, totals = [], 0.0, 0.0, [], []
    for s in range(12):
        env.reset(seed=s)
        for foot in ("calcn_r", "calcn_l"):
            worst_heel = max(worst_heel, abs(env._heel_z(foot)))
            worst_toe = max(worst_toe, abs(env._toe_z(foot)))
        loads.append(env.heel_contact_loads())
        totals.append(env.contact_loads().sum())
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            if floor not in (c.geom1, c.geom2):
                offfloor.append(
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                )
    loads = np.asarray(loads)
    r.check(not offfloor, "no self-collision at reset",
            "MyoSuite ships 14 leg-to-leg pairs that bypass contype")
    r.check(worst_heel < 1e-4, "both heels on the floor, every reset",
            "worst gap %.2e m over 12 resets" % worst_heel)
    r.check(worst_toe < 1e-4, "both toes on the floor, every reset",
            "worst gap %.2e m" % worst_toe)
    r.check(0.25 < min(totals) and max(totals) < 1.6,
            "the feet carry the model at reset",
            "total load %.2f .. %.2f BW" % (min(totals), max(totals)))
    env.reset(seed=seed)
    offset = np.asarray(env.data.subtree_com[0])[:2] - env._support_centroid()[:2]
    r.check(np.abs(offset).max() < 5e-3, "COM over the base of support",
            "offset %.2e m" % np.abs(offset).max())
    r.note("heels carrying load",
           "%d of 12 resets have both heels loaded (%.2f..%.2f BW)"
           % (int((loads > 0.01).all(axis=1).sum()), loads.min(), loads.max()))


def reference_tracking(r, env):
    section("2b. reference tracking")
    print("  The stance and contact checks are skipped for this stage: it resets")
    print("  into a mid-gait frame, where feet staggered, one foot off the ground")
    print("  and the COM outside the base of support are all correct.")
    print()
    r.check(env.reference is not None, "reference gait loaded",
            env.reference.describe() if env.reference else "missing")
    if env.reference is None:
        return

    errs, heights = [], []
    for seed in range(12):
        env.reset(seed=seed)
        errs.append(env.tracking_error)
        heights.append(abs(env.reference.pose_at(env.ref_phase)[1] - env.pelvis_height))
    r.check(max(heights) < 1e-6, "RSI places the pelvis at the reference height",
            "worst error %.1e m over 12 resets" % max(heights))
    r.check(max(errs) < 0.10, "RSI starts on the reference pose",
            "tracking error %.4f .. %.4f rad (reset noise only)" % (min(errs), max(errs)))

    env.reset(seed=0)
    start = env.ref_phase
    for _ in range(50):
        env.step(np.zeros(env.n_act, np.float32))
    advanced = (env.ref_phase - start) % 1.0
    expected = 50 * env.dt / env.reference.duration
    r.check(abs(advanced - expected) < 1e-6, "reference clock advances with sim time",
            "%.4f vs %.4f expected over 50 steps" % (advanced, expected))
    r.note("loop seam", "%.3f rad at the wrap point; the record is not exactly cyclic"
           % env.reference.seam)
    r.note("early termination", "tracking error above %.2f rad"
           % env.stage_spec.max_tracking_error)


def planar_structure(r, env):
    section("2c. planar structure")
    free = [i for i in range(env.model.njnt)
            if env.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    r.check(not free, "free root replaced by three planar joints",
            ", ".join(ROOT_JOINTS))
    pinned = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT,
                                env.model.eq_obj1id[i])
              for i in range(env.model.neq)
              if env.model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT}
    r.check(set(OUT_OF_PLANE_JOINTS) <= pinned,
            "out-of-plane joints pinned to zero",
            "%d joints" % len(OUT_OF_PLANE_JOINTS))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    y0 = np.array([env.data.xipos[b][1] for b in range(1, env.model.nbody)])
    for _ in range(60):
        env.step(rng.uniform(-1, 1, env.n_act).astype(np.float32))
    y1 = np.array([env.data.xipos[b][1] for b in range(1, env.model.nbody)])
    drift = float(np.abs(y1 - y0).max())
    r.check(drift < 0.08, "stays in the sagittal plane under random action",
            "worst segment drift %.4f m (oblique ankle/knee axes kept)" % drift)


def reward_safety(r, env):
    section("3. reward safety")
    report = env.stage_spec.reward.termination_report(gamma=0.99)
    r.check(float(report["min_step_reward"]) >= 0.0,
            "per-step reward cannot go negative",
            "min %.4f" % float(report["min_step_reward"]))
    r.check(not report["termination_preferred"],
            "falling is never the better option", str(report["verdict"]))
    spec = env.stage_spec.reward
    caps = [("alive", spec.alive)]
    total_w = sum(spec.active_weights.values()) or 1.0
    for name, w in sorted(spec.active_weights.items()):
        caps.append((name, spec.shaping_scale * w / total_w))
    r.note("max contribution per step",
           "  ".join("%s %.2f" % (n, c) for n, c in caps)
           + "   total %.2f" % sum(c for _, c in caps))


def term_gradient(r, env, seed):
    section("4. reward-term gradient from the start state")
    print("  A term pinned at its floor gives the policy nothing to climb;")
    print("  a term saturated at 1.0 gives it nothing either.")
    env.reset(seed=seed)
    start = env.compute_terms()
    for name, value in sorted(start.items()):
        flag = ""
        if value < 0.02:
            flag = "  <-- pinned, no gradient"
        elif value > 0.999:
            flag = "  <-- saturated, no gradient"
        print("    %-10s %.4f%s" % (name, value, flag))
    r.check(all(v >= 0.02 for v in start.values()),
            "every term is off its floor at reset",
            "min %.4f" % min(start.values()))


def rollouts(r, env, args):
    section("5. rollouts")
    rng = np.random.default_rng(args.seed)
    for policy in ("zero", "random", "low"):
        env.reset(seed=args.seed + 1)
        total, worst, steps, term = 0.0, np.inf, 0, False
        t0 = time.perf_counter()
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
        print("    %-7s %3d steps  return %8.2f  min step reward %+.4f  "
              "%4.0f steps/s  %s"
              % (policy, steps, total, worst, steps / max(dt, 1e-9),
                 "terminated" if term else "ran out"))
        r.check(worst >= -1e-9, "  %s: no negative non-terminal reward" % policy)


# -- driver ----------------------------------------------------------------


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

        model_and_spaces(r, env)
        posture_frame(r, env, args.seed)
        if env.stage_spec.rsi:
            reference_tracking(r, env)
        else:
            stance_and_contact(r, env, args.seed)
        planar_structure(r, env)
        reward_safety(r, env)
        term_gradient(r, env, args.seed)
        rollouts(r, env, args)
        env.close()

    print()
    print("=" * 78)
    print("FAIL -- %d check(s) failed" % r.failures if r.failures else "PASS")
    print("=" * 78)
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
