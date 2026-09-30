"""A phase-gated mixture-of-experts policy for muscle locomotion.

The architecture, as agreed:

    gait state g_t (30)  ->  GRU(64)  ->  emission logits e_t (K)
                                              |
                                  differentiable HMM filter
                          b_t = softmax( log(b_{t-1}^T T) + log_softmax(e_t / tau) )
                                              |
                                        belief b_t (K)
                                              |
    full obs o_t (610) ---------------------- +  ->  K experts, 610-256-256-290
                                                     a = sum_k b_t[k] * expert_k(o_t)

Four decisions in that diagram are load-bearing.

**The gate sees only the gait state, not the full observation.** 580 of the
610 observation dimensions are muscle activation and prev_action; phase does
not depend on them, and feeding them to a recurrent phase detector is
parameters and noise. `gait_state_dim` is derived from the environment's own
`obs_layout()`, so it tracks the observation rather than being hard-coded.

**The latent is a transition, not a classifier.** `p(z_t | z_{t-1}, o_t)`
rather than `p(z_t | o_t)`. A memoryless softmax re-decides every 10 ms and
flickers; the transition matrix is what makes it segment. The matrix is
initialised diagonal-dominant with a mild bias toward the next index, which
encodes "phases persist and follow one another" without saying what any phase
*is*. It is free to learn, and for a periodic gait it should converge toward a
cycle -- which you can read off `transition_matrix()` afterwards.

**Nothing supervises the phase.** No labels enter the loss. What keeps the
gate from collapsing onto one expert is the pair of entropy terms below --
decisive per step, balanced over a batch -- plus the fact that reference-state
initialisation samples the gait cycle uniformly, so the batch is balanced by
construction.

**Blending is annealed toward hard.** Averaging two experts' 290-dimensional
activation vectors is not averaging their behaviours: muscle force is
nonlinear in activation, and two experts that both extend the knee through
different muscles average into co-contraction. Lowering `tau` over training
sharpens `b_t` toward one-hot and keeps the blend window short.

The critic is a separate network. Sharing a trunk with the phase gate makes
the value objective compete with the clustering.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Observation blocks that come *after* the gait state. Everything before these
# is what the phase gate sees.
_NON_GAIT_BLOCKS = ("muscle_activation", "prev_action")


def gait_state_dim(obs_layout: Sequence[Tuple[str, int]]) -> int:
    """Width of the leading observation slice the phase gate reads.

    Everything up to `muscle_activation` -- pelvis height, trunk tilt, joint
    angles and velocities, COM velocity and the heel/toe contact split. On the
    planar env that is 30 of 610.
    """
    total = 0
    for name, width in obs_layout:
        if name in _NON_GAIT_BLOCKS:
            break
        total += width
    if total == 0:
        raise ValueError("no gait-state blocks found in %r" % (list(obs_layout),))
    return total


def _mlp(sizes: Sequence[int], activation=nn.Tanh) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class PhaseGate(nn.Module):
    """GRU + differentiable HMM filter over K discrete phases."""

    def __init__(
        self,
        gait_dim: int,
        n_phases: int = 4,
        hidden: int = 64,
        sticky: float = 3.0,
        cyclic_bias: float = 1.0,
    ):
        super().__init__()
        self.n_phases = int(n_phases)
        self.hidden = int(hidden)
        self.gru = nn.GRU(gait_dim, hidden)
        self.emission = nn.Linear(hidden, n_phases)

        # Diagonal-dominant with a mild push toward the next index. This says
        # phases persist and follow one another; it does not say which phase
        # is which, and every entry is free to move.
        init = torch.zeros(n_phases, n_phases)
        for i in range(n_phases):
            init[i, i] = sticky
            init[i, (i + 1) % n_phases] = cyclic_bias
        self.transition_logits = nn.Parameter(init)

    def log_transition(self) -> torch.Tensor:
        """Row-normalised log transition matrix, `log p(z_t | z_{t-1})`."""
        return F.log_softmax(self.transition_logits, dim=-1)

    def transition_matrix(self) -> torch.Tensor:
        """The learned transition probabilities, for inspection."""
        return self.log_transition().exp().detach()

    def initial_log_belief(self, batch: int, device=None) -> torch.Tensor:
        """Uniform over phases -- the gate has seen nothing yet."""
        return torch.full(
            (batch, self.n_phases), -math.log(self.n_phases), device=device
        )

    def initial_hidden(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden, device=device)

    def forward(
        self,
        gait_seq: torch.Tensor,          # (T, B, gait_dim)
        hidden: torch.Tensor,            # (1, B, hidden)
        log_belief: torch.Tensor,        # (B, K)
        resets: Optional[torch.Tensor] = None,   # (T, B) 1.0 where an episode began
        tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter a sequence. Returns (log_beliefs (T,B,K), hidden, log_belief)."""
        steps = gait_seq.shape[0]
        log_T = self.log_transition()
        out: List[torch.Tensor] = []

        for t in range(steps):
            if resets is not None:
                # A new episode has no history: clear both the recurrent state
                # and the belief, or the gate carries the previous episode's
                # phase across the boundary.
                keep = (1.0 - resets[t]).view(1, -1, 1)
                hidden = hidden * keep
                fresh = self.initial_log_belief(log_belief.shape[0], log_belief.device)
                mask = resets[t].view(-1, 1) > 0.5
                log_belief = torch.where(mask, fresh, log_belief)

            step_out, hidden = self.gru(gait_seq[t : t + 1], hidden)
            emission = F.log_softmax(self.emission(step_out[0]) / tau, dim=-1)
            # predict:  log sum_j b_{t-1}[j] T[j, k]
            predicted = torch.logsumexp(
                log_belief.unsqueeze(-1) + log_T.unsqueeze(0), dim=1
            )
            log_belief = F.log_softmax(predicted + emission, dim=-1)
            out.append(log_belief)

        return torch.stack(out), hidden, log_belief


class MixtureActor(nn.Module):
    """K expert MLPs over the full observation, blended by the belief.

    The experts are held as batched weight tensors rather than a ModuleList so
    all K are evaluated in one einsum -- soft gating needs every expert's
    output anyway, to give them all gradient.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_experts: int = 4,
        hidden: int = 256,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.n_experts = int(n_experts)
        self.act_dim = int(act_dim)

        def weight(fan_in: int, fan_out: int, gain: float) -> nn.Parameter:
            w = torch.empty(n_experts, fan_in, fan_out)
            for k in range(n_experts):
                nn.init.orthogonal_(w[k], gain=gain)
            return nn.Parameter(w)

        self.w1 = weight(obs_dim, hidden, math.sqrt(2))
        self.b1 = nn.Parameter(torch.zeros(n_experts, hidden))
        self.w2 = weight(hidden, hidden, math.sqrt(2))
        self.b2 = nn.Parameter(torch.zeros(n_experts, hidden))
        # Small output gain: the policy starts near the middle of the
        # activation range rather than saturating 290 muscles on step one.
        self.w3 = weight(hidden, act_dim, 0.01)
        self.b3 = nn.Parameter(torch.zeros(n_experts, act_dim))

        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def expert_means(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, K, act_dim) -- every expert's action mean."""
        h = torch.tanh(torch.einsum("bo,koh->bkh", obs, self.w1) + self.b1)
        h = torch.tanh(torch.einsum("bkh,khg->bkg", h, self.w2) + self.b2)
        return torch.einsum("bkh,kha->bka", h, self.w3) + self.b3

    def forward(self, obs: torch.Tensor, belief: torch.Tensor):
        """Blended mean and the per-expert means, for diagnostics."""
        means = self.expert_means(obs)
        mean = torch.einsum("bk,bka->ba", belief, means)
        return mean, means

    def distribution(self, obs: torch.Tensor, belief: torch.Tensor):
        mean, _ = self.forward(obs, belief)
        return torch.distributions.Normal(mean, self.log_std.exp())


class PhaseGatedActorCritic(nn.Module):
    """The policy: phase gate, mixture actor, and a separate critic."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        gait_dim: int,
        n_phases: int = 4,
        gate_hidden: int = 64,
        expert_hidden: int = 256,
        critic_hidden: int = 256,
        sticky: float = 3.0,
        cyclic_bias: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.gait_dim = int(gait_dim)
        self.n_phases = int(n_phases)

        self.gate = PhaseGate(gait_dim, n_phases, gate_hidden, sticky, cyclic_bias)
        self.actor = MixtureActor(obs_dim, act_dim, n_phases, expert_hidden)
        # Separate, and it sees the belief: the value of a state depends on
        # where in the gait you are.
        self.critic = _mlp(
            [obs_dim + n_phases, critic_hidden, critic_hidden, 1]
        )

    # -- recurrent state ---------------------------------------------------

    def initial_state(self, batch: int, device=None):
        return (
            self.gate.initial_hidden(batch, device),
            self.gate.initial_log_belief(batch, device),
        )

    def gait_slice(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., : self.gait_dim]

    # -- rollout -----------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: torch.Tensor, state, reset: Optional[torch.Tensor] = None,
            tau: float = 1.0, deterministic: bool = False):
        """One step. `obs` is (B, obs_dim); `reset` is (B,) 1.0 at episode starts."""
        hidden, log_belief = state
        resets = None if reset is None else reset.view(1, -1)
        log_b, hidden, log_belief = self.gate(
            self.gait_slice(obs).unsqueeze(0), hidden, log_belief, resets, tau
        )
        belief = log_b[0].exp()
        dist = self.actor.distribution(obs, belief)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.critic(torch.cat([obs, belief], dim=-1)).squeeze(-1)
        return action, logp, value, belief, (hidden, log_belief)

    # -- update ------------------------------------------------------------

    def evaluate(
        self,
        obs_seq: torch.Tensor,        # (T, B, obs_dim)
        actions: torch.Tensor,        # (T, B, act_dim)
        state,
        resets: Optional[torch.Tensor] = None,   # (T, B)
        tau: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Re-run the sequence with gradients.

        The hidden state is recomputed here rather than replayed from the
        rollout: stale hidden states make PPO's importance ratio quietly
        off-policy, which is the usual way a recurrent PPO goes subtly wrong.
        """
        steps, batch = obs_seq.shape[0], obs_seq.shape[1]
        hidden, log_belief = state
        log_b, _, _ = self.gate(
            self.gait_slice(obs_seq), hidden, log_belief, resets, tau
        )
        beliefs = log_b.exp()

        flat_obs = obs_seq.reshape(steps * batch, -1)
        flat_belief = beliefs.reshape(steps * batch, -1)
        dist = self.actor.distribution(flat_obs, flat_belief)
        logp = dist.log_prob(actions.reshape(steps * batch, -1)).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(torch.cat([flat_obs, flat_belief], dim=-1)).squeeze(-1)

        return {
            "log_prob": logp.view(steps, batch),
            "entropy": entropy.view(steps, batch),
            "value": value.view(steps, batch),
            "belief": beliefs,
            "log_belief": log_b,
        }


# -- auxiliary losses ------------------------------------------------------
#
# None of these use a label. Together they are what makes a phase-like
# partition emerge rather than the gate collapsing onto one expert.


def switching_loss(log_beliefs: torch.Tensor) -> torch.Tensor:
    """Mean KL between consecutive beliefs -- the temporal-consistency term.

    The one that matters most. A per-step softmax re-decides every 10 ms and
    flickers; penalising change is what turns it into segments, which is what
    a gait phase is.
    """
    if log_beliefs.shape[0] < 2:
        return log_beliefs.new_zeros(())
    current, previous = log_beliefs[1:], log_beliefs[:-1].detach()
    return (current.exp() * (current - previous)).sum(-1).mean()


def balance_loss(log_beliefs: torch.Tensor) -> torch.Tensor:
    """Negative entropy of batch-averaged usage -- keeps all experts in play.

    Minimising this maximises `H(E[b])`, so no expert is starved. With RSI
    sampling the gait cycle uniformly the batch is already close to balanced,
    so this can usually be given a small weight.
    """
    mean = log_beliefs.exp().reshape(-1, log_beliefs.shape[-1]).mean(0)
    return (mean * torch.log(mean.clamp_min(1e-8))).sum()


def confidence_loss(log_beliefs: torch.Tensor) -> torch.Tensor:
    """Mean per-step entropy -- keeps the gate decisive rather than smeared.

    The counterweight to `balance_loss`: confident for a given state, balanced
    across the batch. That pair is the clustering objective.
    """
    return -(log_beliefs.exp() * log_beliefs).sum(-1).mean()


def phase_alignment(beliefs: torch.Tensor, reference_phase: torch.Tensor,
                    n_bins: int = 4) -> torch.Tensor:
    """How the discovered phases line up with the reference cycle.

    **A diagnostic, never a loss.** Returns a (K, n_bins) matrix of mean belief
    per bin of reference phase. If the gate found the gait's structure the
    matrix is close to a permutation; if it collapsed or split on something
    else, that shows immediately.
    """
    beliefs = beliefs.reshape(-1, beliefs.shape[-1])
    phase = reference_phase.reshape(-1)
    bins = torch.clamp((phase * n_bins).long(), 0, n_bins - 1)
    out = beliefs.new_zeros(beliefs.shape[-1], n_bins)
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            out[:, b] = beliefs[mask].mean(0)
    return out
