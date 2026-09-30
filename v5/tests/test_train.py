"""Tests for the PPO trainer's error-prone parts.

Recurrent PPO fails quietly rather than loudly, so what is covered here is the
machinery that corrupts training without raising anything: bootstrapping, the
return scaler, and whether the vector env hands back the observation the
episode actually ended on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

torch = pytest.importorskip("torch")

from _train_import import RunningStd, SyncVecEnv, compute_gae  # noqa: E402


def zeros(t, n):
    return torch.zeros(t, n)


# -- GAE -------------------------------------------------------------------


def test_no_done_matches_the_plain_recursion():
    t, n = 5, 2
    rew = torch.ones(t, n)
    val = torch.full((t, n), 0.5)
    adv = compute_gae(rew, val, zeros(t, n), zeros(t, n), zeros(t, n),
                      torch.full((n,), 0.5), gamma=0.99, lam=0.95)
    expected = torch.zeros(t, n)
    last = torch.zeros(n)
    for i in reversed(range(t)):
        nxt = torch.full((n,), 0.5) if i == t - 1 else val[i + 1]
        delta = rew[i] + 0.99 * nxt - val[i]
        last = delta + 0.99 * 0.95 * last
        expected[i] = last
    assert torch.allclose(adv, expected, atol=1e-6)


def test_terminated_bootstraps_from_zero():
    """A fall really is the end; the future is worth nothing."""
    t, n = 3, 1
    val = torch.full((t, n), 2.0)
    term = zeros(t, n)
    term[1, 0] = 1.0
    adv = compute_gae(torch.zeros(t, n), val, term, zeros(t, n), zeros(t, n),
                      torch.zeros(n), gamma=0.9, lam=1.0)
    # delta = 0 + 0.9 * 0 - 2
    assert adv[1, 0] == pytest.approx(-2.0, abs=1e-6)


def test_truncated_bootstraps_from_the_final_value():
    """Running out of time is not a terminal state."""
    t, n = 3, 1
    val = torch.full((t, n), 2.0)
    trunc = zeros(t, n)
    trunc[1, 0] = 1.0
    final = zeros(t, n)
    final[1, 0] = 10.0
    adv = compute_gae(torch.zeros(t, n), val, zeros(t, n), trunc, final,
                      torch.zeros(n), gamma=0.9, lam=1.0)
    # delta = 0 + 0.9 * 10 - 2
    assert adv[1, 0] == pytest.approx(7.0, abs=1e-6)


def test_the_two_done_flags_are_not_interchangeable():
    """The bug this guards against: treating truncation as termination."""
    t, n = 4, 1
    val = torch.full((t, n), 1.0)
    final = zeros(t, n)
    final[2, 0] = 5.0
    term = zeros(t, n)
    term[2, 0] = 1.0
    trunc = zeros(t, n)
    trunc[2, 0] = 1.0
    as_term = compute_gae(torch.zeros(t, n), val, term, zeros(t, n), final,
                          torch.zeros(n), 0.99, 0.95)
    as_trunc = compute_gae(torch.zeros(t, n), val, zeros(t, n), trunc, final,
                           torch.zeros(n), 0.99, 0.95)
    assert not torch.allclose(as_term, as_trunc)
    assert as_trunc[2, 0] > as_term[2, 0]


def test_both_flags_end_the_recursion():
    """Advantage must not leak backwards across an episode boundary."""
    t, n = 6, 1
    rew = torch.zeros(t, n)
    rew[4, 0] = 100.0
    val = torch.zeros(t, n)
    for flag in ("terminated", "truncated"):
        term, trunc = zeros(t, n), zeros(t, n)
        (term if flag == "terminated" else trunc)[2, 0] = 1.0
        adv = compute_gae(rew, val, term, trunc, zeros(t, n), torch.zeros(n),
                          0.99, 0.95)
        assert adv[1, 0] == pytest.approx(0.0, abs=1e-6), flag
        assert adv[3, 0] > 0.0, flag


def test_reward_at_the_last_step_reaches_the_first():
    """Stage W pays up to 700 terminally; it has to propagate back."""
    t, n = 20, 1
    rew = torch.zeros(t, n)
    rew[-1, 0] = 700.0
    adv = compute_gae(rew, torch.zeros(t, n), zeros(t, n), zeros(t, n),
                      zeros(t, n), torch.zeros(n), 0.99, 0.95)
    assert adv[0, 0] > 1.0


# -- return scaling --------------------------------------------------------


def test_running_std_tracks_the_return_scale():
    rs = RunningStd(gamma=0.99)
    rng = np.random.default_rng(0)
    for _ in range(50):
        rew = rng.normal(0.0, 5.0, size=(16, 4)).astype(np.float32)
        rs.update(rew, np.zeros((16, 4), dtype=np.float32))
    assert rs.std > 1.0
    assert np.isfinite(rs.std)


def test_running_std_survives_episode_boundaries():
    rs = RunningStd(gamma=0.99)
    rew = np.ones((10, 2), dtype=np.float32)
    resets = np.zeros((10, 2), dtype=np.float32)
    resets[5] = 1.0
    rs.update(rew, resets)
    assert np.isfinite(rs.std) and rs.std > 0


# -- vector env ------------------------------------------------------------


def test_vec_env_returns_the_true_final_observation():
    """The reset observation is not the one the episode ended on.

    Bootstrapping truncation from the reset state instead is silent and wrong.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("myosuite")
    envs = SyncVecEnv("W", 2, seed=0)
    try:
        envs.reset(0)
        rng = np.random.default_rng(0)
        for _ in range(400):
            action = np.full((2, envs.act_dim), -0.6, dtype=np.float32)
            obs, _, term, trunc, final, _, finished = envs.step(
                action, rng.integers(0, 2**31 - 1, size=2)
            )
            done = term | trunc
            if done.any():
                i = int(np.flatnonzero(done)[0])
                assert not np.allclose(final[i], obs[i]), (
                    "final and reset observation should differ"
                )
                assert np.isfinite(final[i]).all()
                assert finished, "a finished episode should be reported"
                return
        pytest.fail("no episode ended within 400 steps")
    finally:
        envs.close()
