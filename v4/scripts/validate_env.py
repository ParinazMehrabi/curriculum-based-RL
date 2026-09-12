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

import numpy as np

import gym
import sconegym  # noqa: F401  (registers the base envs)

import sconegym_crutch_v4 as scv4


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
        print(
            "reset %d: com vx=%+.5f pelvis y=%.5f tilt=%+.4f"
            % (seed, u.model.com_vel().x, q[u._dof_index["pelvis_ty"]], q[u._dof_index["pelvis_tilt"]])
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
