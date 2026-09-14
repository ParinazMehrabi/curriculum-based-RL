"""Smoke test for a v4 stage environment. Requires sconegym + Hyfydy.

Unlike the v3 validation script this targets an environment that actually
exists, and it checks the things that were silently wrong before:

  * prev_action is present in the observation (the Markov fix)
  * the crutch contact force API returns a real number under load
  * every reward term lands in [0, 1]
  * no step reward is negative

Usage:
    python scripts/validate_env.py --stage A
    python scripts/validate_env.py --stage D --steps 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import sconegym_crutch_v4` work however this file was invoked.
REPO_V4 = Path(__file__).resolve().parents[1]
if str(REPO_V4) not in sys.path:
    sys.path.insert(0, str(REPO_V4))

import numpy as np

import gym
import sconegym  # noqa: F401  (registers the base envs)

import sconegym_crutch_v4 as scv4


def keyframe_label(lo: float, hi: float) -> str:
    """Name the gait sub-movement a window corresponds to."""
    for name, windows in scv4.GAIT_KEYFRAMES.items():
        for wlo, whi in windows:
            if abs(wlo - lo) < 1e-6 and abs(whi - hi) < 1e-6:
                return name
    return "?"


def frame_keyframe(frac: float) -> str:
    """Name the sub-movement a sampled frame fell in, with a small tolerance."""
    for name, windows in scv4.GAIT_KEYFRAMES.items():
        for lo, hi in windows:
            if lo - 0.005 <= frac <= hi + 0.005:
                return name
    return "(gap)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A", help="A, B, C or D")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--no-strict-crutch",
        action="store_true",
        help="allow construction even if the crutch force API is broken",
    )
    args = ap.parse_args()

    env_id = scv4.env_id_for(args.stage)
    print("=" * 78)
    print("v4 environment smoke test:", env_id)
    print("=" * 78)

    env = gym.make(env_id, strict_crutch=not args.no_strict_crutch)
    u = env.unwrapped
    spec = u.stage_spec

    print("stage            :", spec.describe())
    print("dofs / actuators :", len(u.model.dofs()), "/", len(u.model.actuators()))
    print("muscles          :", len(u.model.muscles()))
    print("body weight (N)  :", u._body_weight_n)
    print("action space     :", env.action_space.shape, env.action_space.low[0], env.action_space.high[0])
    print("observation space:", env.observation_space.shape)
    print("target velocity  :", spec.target_vel, "sigma", spec.velocity_sigma)
    print("crutch force ok  :", u._crutch_force_ok, u._crutch_probe_error or "")
    print("needs crutch     : force=%s pose=%s" % (spec.needs_crutch_force, spec.needs_crutch_pose))
    print("reward safety    :", spec.reward.termination_report()["verdict"])
    if spec.rsi is not None:
        indent = chr(10) + " " * 19
        print("RSI              : %s" % u.trajectory.describe().replace(chr(10), indent))
        print("  velocity_scale : %.2f" % spec.rsi.velocity_scale)
        print("  posture ref    : %s" % spec.rsi.posture_reference)
        n = u.trajectory.n_frames
        if spec.rsi.phase_window_groups:
            groups = spec.rsi.phase_window_groups
            print("  sampling       : %d sub-movements, each equally likely"
                  % len(groups))
            for group in groups:
                label = keyframe_label(*group[0])
                spans = " ".join(
                    "%3d-%3d" % (int(lo * (n - 1)), int(hi * (n - 1)) + 1)
                    for lo, hi in group
                )
                print("      %-9s %d window(s)  frames %s"
                      % (label, len(group), spans))
        elif spec.rsi.phase_windows:
            print("  sampling       : %d keyframe windows (phase_range ignored)"
                  % len(spec.rsi.phase_windows))
            for lo, hi in spec.rsi.phase_windows:
                label = keyframe_label(lo, hi)
                print("      %.4f-%.4f  frames %3d-%3d  %s"
                      % (lo, hi, int(lo * (n - 1)), int(hi * (n - 1)) + 1, label))
        else:
            print("  sampling       : whole cycle, phase_range %s"
                  % (spec.rsi.phase_range,))
    else:
        print("RSI              : disabled (resets to the neutral pose)")
    print()

    assert env.action_space.shape == (9,)
    assert len(u.model.dofs()) == 16
    assert len(u.model.actuators()) == 9
    assert len(u.model.muscles()) == 0

    # --- the Markov fix ---------------------------------------------------
    obs = env.reset(seed=0)
    base_dim = np.asarray(gym.make(env_id, include_prev_action=False).reset(seed=0)).size
    print("obs dim with prev_action   :", obs.size)
    print("obs dim without            :", base_dim)
    assert obs.size == base_dim + 9, (
        "prev_action is not reaching the observation: %d vs %d + 9" % (obs.size, base_dim)
    )

    probe = np.full(9, 0.7, dtype=np.float32)
    for _ in range(3):
        obs, _, _, _ = env.step(probe)
    tail = np.asarray(obs, dtype=np.float32)[-9:]
    assert np.allclose(tail, u.prev_action, atol=1e-5), (
        "observation tail %r does not match prev_action %r" % (tail, u.prev_action)
    )
    print("obs tail matches prev_action: OK", np.round(tail, 4))
    print()

    # --- reset diversity --------------------------------------------------
    states = []
    for seed in range(args.seeds):
        env.reset(seed=seed)
        q = np.asarray(u.model.dof_position_array(), dtype=float)
        dq = np.asarray(u.model.dof_velocity_array(), dtype=float)
        states.append(np.concatenate([q, dq]))
        frame_label = ""
        if u.rsi_frame is not None and u.trajectory is not None:
            frac = u.rsi_frame / float(u.trajectory.n_frames - 1)
            frame_label = frame_keyframe(frac)
        print(
            "reset %d: frame=%-5s %-9s com vx=%+.5f pelvis y=%.5f tilt=%+.4f hipR=%+.4f kneeR=%+.4f"
            % (
                seed,
                u.rsi_frame if u.rsi_frame is not None else "-",
                frame_label,
                u.model.com_vel().x,
                q[u._dof_index["pelvis_ty"]],
                q[u._dof_index["pelvis_tilt"]],
                q[u._dof_index["hip_flexion_r"]],
                q[u._dof_index["knee_angle_r"]],
            )
        )
    distinct = sum(1 for s in states[1:] if not np.allclose(states[0], s))
    assert distinct >= max(1, args.seeds - 2), "reset diversity unexpectedly low"
    print()

    # --- rollout invariants ----------------------------------------------
    env.reset(seed=123)
    rng = np.random.RandomState(0)
    min_rew, max_rew, neg = np.inf, -np.inf, 0
    term_lo = {}
    term_hi = {}
    crutch_forces = []
    steps_done = 0

    for i in range(args.steps):
        action = rng.uniform(-0.3, 0.3, size=9).astype(np.float32)
        obs, rew, done, info = env.step(action)
        steps_done += 1
        min_rew = min(min_rew, rew)
        max_rew = max(max_rew, rew)
        if rew < 0.0 and not done:
            neg += 1
        for name, value in u.term_values.items():
            term_lo[name] = min(term_lo.get(name, 1.0), value)
            term_hi[name] = max(term_hi.get(name, 0.0), value)
            assert -1e-9 <= value <= 1.0 + 1e-9, "term %s out of [0,1]: %r" % (name, value)
        if spec.needs_crutch_force:
            crutch_forces.append(u.crutch_contact_force())
        if done:
            break

    print("steps run        :", steps_done)
    print("reward range     : %.4f .. %.4f" % (min_rew, max_rew))
    print("negative non-terminal steps:", neg)
    assert neg == 0, "per-step reward went negative, which should be impossible"

    print("term ranges observed:")
    for name in sorted(term_hi):
        print("  %-16s %.4f .. %.4f" % (name, term_lo[name], term_hi[name]))

    if crutch_forces:
        arr = np.asarray(crutch_forces)
        print(
            "crutch force (N) : min %.2f mean %.2f max %.2f"
            % (arr.min(), arr.mean(), arr.max())
        )
        if arr.max() <= 0.0:
            print()
            print("WARNING: crutch force was zero for every step of this rollout.")
            print("The crutch reward term is therefore contributing nothing. Verify the")
            print("contact model in the .hfd before trusting any crutch numbers.")

    print()
    print("PASS")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
