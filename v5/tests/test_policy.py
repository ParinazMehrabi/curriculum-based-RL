"""Tests for the phase-gated mixture-of-experts policy.

These need torch but not the environment, except where a test says otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from _myosuite_data import require_model  # noqa: E402

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

torch = pytest.importorskip("torch")

from myo_curriculum.policy import (  # noqa: E402
    MixtureActor,
    PhaseGate,
    PhaseGatedActorCritic,
    balance_loss,
    confidence_loss,
    gait_state_dim,
    phase_alignment,
    switching_loss,
)

OBS, ACT, GAIT, K = 610, 290, 30, 4


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return PhaseGatedActorCritic(OBS, ACT, GAIT, n_phases=K)


# -- the gait slice --------------------------------------------------------


def test_gait_state_dim_stops_before_the_activation_block():
    """The gate must not be fed 580 dims of activation and prev_action."""
    layout = [
        ("pelvis_height", 1), ("pelvis_tilt_sin_cos", 2), ("joint_q", 9),
        ("root_dq", 3), ("joint_dq", 9), ("com_vel_xz", 2),
        ("contact_load_heel_toe", 4),
        ("muscle_activation", 290), ("prev_action", 290),
    ]
    assert gait_state_dim(layout) == 30


def test_gait_state_dim_tracks_the_env_not_a_constant():
    pytest.importorskip("mujoco")
    require_model()
    from myo_curriculum.env import MyoLocomotionEnv

    env = MyoLocomotionEnv(stage="W", seed=0)
    try:
        assert gait_state_dim(env.obs_layout()) == 30
        assert env.observation_space.shape[0] == 610
    finally:
        env.close()


def test_gate_reads_only_the_gait_slice(net):
    """Changing the activation block must not move the belief."""
    obs = torch.randn(1, OBS)
    state = net.initial_state(1)
    _, _, _, b1, _ = net.act(obs, state, torch.ones(1))
    obs2 = obs.clone()
    obs2[:, GAIT:] = torch.randn(1, OBS - GAIT) * 5.0
    _, _, _, b2, _ = net.act(obs2, state, torch.ones(1))
    assert torch.allclose(b1, b2, atol=1e-6)


# -- the transition prior --------------------------------------------------


def test_transition_matrix_starts_sticky_and_cyclic(net):
    """Structure, not content: phases persist and follow one another.

    Nothing here says which phase is which -- only that they last and are
    ordered. Every entry is free to learn.
    """
    T = net.gate.transition_matrix()
    assert T.shape == (K, K)
    assert torch.allclose(T.sum(-1), torch.ones(K), atol=1e-6)
    for i in range(K):
        assert T[i, i] > 0.5, "should persist"
        assert T[i, i] > T[i, (i + 1) % K] > T[i, (i + 2) % K]


def test_transition_matrix_is_learnable(net):
    assert net.gate.transition_logits.requires_grad


# -- the filter ------------------------------------------------------------


def test_belief_is_a_distribution(net):
    obs = torch.randn(7, OBS)
    state = net.initial_state(7)
    _, _, _, belief, _ = net.act(obs, state, torch.ones(7))
    assert belief.shape == (7, K)
    assert torch.allclose(belief.sum(-1), torch.ones(7), atol=1e-5)
    assert (belief >= 0).all()


def test_initial_belief_is_uniform(net):
    _, log_b = net.initial_state(3)
    assert torch.allclose(log_b.exp(), torch.full((3, K), 1.0 / K), atol=1e-6)


def test_reset_clears_the_recurrent_state(net):
    """Otherwise the gate carries a phase across an episode boundary."""
    gate = net.gate
    seq = torch.randn(6, 2, GAIT)
    hidden = gate.initial_hidden(2)
    log_b = gate.initial_log_belief(2)

    resets = torch.zeros(6, 2)
    resets[0] = 1.0
    a, _, _ = gate(seq, hidden, log_b, resets, tau=1.0)

    # Same sequence, but the second half restarted: the tail must match a run
    # that began there.
    resets2 = torch.zeros(3, 2)
    resets2[0] = 1.0
    b, _, _ = gate(seq[3:], gate.initial_hidden(2), gate.initial_log_belief(2),
                   resets2, tau=1.0)
    resets3 = torch.zeros(6, 2)
    resets3[0] = 1.0
    resets3[3] = 1.0
    c, _, _ = gate(seq, hidden, log_b, resets3, tau=1.0)
    assert torch.allclose(c[3:], b, atol=1e-5)
    assert not torch.allclose(a[3:], b, atol=1e-3)


def test_temperature_sharpens_the_belief(net):
    """Annealing tau toward 0 drives b_t to one-hot, shortening the blend."""
    seq = torch.randn(12, 4, GAIT)
    gate = net.gate
    out = {}
    for tau in (2.0, 1.0, 0.2):
        log_b, _, _ = gate(seq, gate.initial_hidden(4), gate.initial_log_belief(4),
                           None, tau=tau)
        out[tau] = -(log_b.exp() * log_b).sum(-1).mean().item()
    assert out[2.0] > out[1.0] > out[0.2], "lower tau should be more decisive"


# -- the mixture actor -----------------------------------------------------


def test_all_experts_are_evaluated_and_blended():
    torch.manual_seed(1)
    actor = MixtureActor(OBS, ACT, n_experts=K, hidden=32)
    obs = torch.randn(5, OBS)
    means = actor.expert_means(obs)
    assert means.shape == (5, K, ACT)
    # Experts must differ, or the mixture is decorative.
    spread = means.std(dim=1).mean().item()
    assert spread > 0.0

    onehot = torch.zeros(5, K)
    onehot[:, 2] = 1.0
    blended, _ = actor(obs, onehot)
    assert torch.allclose(blended, means[:, 2], atol=1e-6)


def test_blend_is_a_convex_combination():
    torch.manual_seed(2)
    actor = MixtureActor(16, 3, n_experts=K, hidden=8)
    obs = torch.randn(4, 16)
    belief = torch.softmax(torch.randn(4, K), dim=-1)
    blended, means = actor(obs, belief)
    manual = (belief.unsqueeze(-1) * means).sum(1)
    assert torch.allclose(blended, manual, atol=1e-6)


def test_every_expert_receives_gradient(net):
    """Soft gating exists so all K experts learn; check they actually do."""
    obs = torch.randn(8, OBS)
    belief = torch.softmax(torch.randn(8, K), dim=-1)
    mean, _ = net.actor(obs, belief)
    mean.sum().backward()
    grad = net.actor.w3.grad
    assert grad is not None
    per_expert = grad.abs().sum(dim=(1, 2))
    assert (per_expert > 0).all(), "an expert with no gradient is dead weight"
    net.zero_grad(set_to_none=True)


# -- actor-critic ----------------------------------------------------------


def test_act_shapes(net):
    obs = torch.randn(6, OBS)
    action, logp, value, belief, state = net.act(obs, net.initial_state(6),
                                                 torch.ones(6))
    assert action.shape == (6, ACT)
    assert logp.shape == (6,)
    assert value.shape == (6,)
    assert belief.shape == (6, K)
    assert state[0].shape == (1, 6, net.gate.hidden)


def test_evaluate_reproduces_the_rollout_beliefs(net):
    """The update recomputes hidden state; it must match what acting produced.

    Replaying stale hidden states instead is the usual way a recurrent PPO
    goes quietly off-policy.
    """
    steps, batch = 9, 3
    obs = torch.randn(steps, batch, OBS)
    resets = torch.zeros(steps, batch)
    resets[0] = 1.0

    state = net.initial_state(batch)
    rollout, actions = [], []
    for t in range(steps):
        a, _, _, belief, state = net.act(obs[t], state, resets[t])
        rollout.append(belief)
        actions.append(a)
    rollout = torch.stack(rollout)

    out = net.evaluate(obs, torch.stack(actions), net.initial_state(batch),
                       resets)
    assert torch.allclose(out["belief"], rollout, atol=1e-5)


def test_evaluate_is_differentiable(net):
    steps, batch = 5, 2
    obs = torch.randn(steps, batch, OBS)
    actions = torch.randn(steps, batch, ACT)
    out = net.evaluate(obs, actions, net.initial_state(batch))
    (out["log_prob"].sum() + out["value"].sum()).backward()
    assert net.gate.transition_logits.grad is not None
    assert net.actor.w1.grad is not None
    net.zero_grad(set_to_none=True)


def test_critic_is_separate_from_the_gate(net):
    """Sharing a trunk makes the value objective fight the clustering."""
    gate_params = {id(p) for p in net.gate.parameters()}
    critic_params = {id(p) for p in net.critic.parameters()}
    assert gate_params.isdisjoint(critic_params)


# -- auxiliary losses ------------------------------------------------------


def test_switching_loss_punishes_flicker():
    steady = torch.log(torch.tensor([[[0.97, 0.01, 0.01, 0.01]]]).repeat(8, 1, 1))
    flicker = torch.log(torch.stack([
        torch.tensor([[0.97, 0.01, 0.01, 0.01]]) if t % 2 else
        torch.tensor([[0.01, 0.97, 0.01, 0.01]])
        for t in range(8)
    ]))
    assert switching_loss(flicker) > switching_loss(steady)
    assert switching_loss(steady) < 1e-3


def test_balance_loss_punishes_a_collapsed_gate():
    balanced = torch.log(torch.full((10, 2, K), 1.0 / K))
    collapsed = torch.log(
        torch.tensor([[0.97, 0.01, 0.01, 0.01]]).repeat(10, 2, 1)
    )
    assert balance_loss(collapsed) > balance_loss(balanced)


def test_confidence_loss_punishes_a_smeared_gate():
    smeared = torch.log(torch.full((10, 2, K), 1.0 / K))
    decisive = torch.log(
        torch.tensor([[0.97, 0.01, 0.01, 0.01]]).repeat(10, 2, 1)
    )
    assert confidence_loss(smeared) > confidence_loss(decisive)


def test_balance_and_confidence_pull_against_each_other():
    """That tension is the clustering objective: decisive, but all used."""
    uniform = torch.log(torch.full((10, 4, K), 1.0 / K))
    onehot = torch.log(
        torch.tensor([[0.97, 0.01, 0.01, 0.01]]).repeat(10, 4, 1)
    )
    assert confidence_loss(uniform) > confidence_loss(onehot)
    assert balance_loss(uniform) < balance_loss(onehot)


# -- diagnostic ------------------------------------------------------------


def test_phase_alignment_is_a_diagnostic_not_a_loss():
    """Perfect discovery shows up as a permutation-like matrix."""
    steps = 400
    phase = torch.rand(steps)
    bins = torch.clamp((phase * K).long(), 0, K - 1)
    beliefs = torch.nn.functional.one_hot(bins, K).float()
    aligned = phase_alignment(beliefs, phase, n_bins=K)
    assert aligned.shape == (K, K)
    assert torch.allclose(aligned.diagonal(), torch.ones(K), atol=1e-6)

    scrambled = phase_alignment(torch.full((steps, K), 1.0 / K), phase, n_bins=K)
    assert torch.allclose(scrambled, torch.full((K, K), 1.0 / K), atol=1e-6)
