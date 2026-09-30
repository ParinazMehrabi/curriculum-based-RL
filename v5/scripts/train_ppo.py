"""PPO for the phase-gated mixture-of-experts policy.

    python scripts/train_ppo.py --stage W --total-steps 200000
    python scripts/train_ppo.py --phases 1        # plain-MLP baseline, no gate
    python scripts/train_ppo.py --resume runs/W-.../ckpt_latest.pt

Each run writes `ckpt_latest.pt` for resuming and keeps its ten
highest-return checkpoints as `ckpt_best_it<N>_ret<R>.pt`, pruning its own
lower-scoring ones as better iterations arrive. Nothing else in `runs/` is
ever deleted.

Recurrent PPO is where implementations usually go quietly wrong, so the three
things that matter are handled explicitly and each is noted where it happens:

* **Hidden state is recomputed during the update, not replayed.** Minibatches
  are over *environments*, keeping whole sequences intact, and every epoch
  re-runs the gate from the segment's stored initial state. Replaying stored
  hidden states makes the importance ratio off-policy by an amount that grows
  with epoch count.
* **`terminated` and `truncated` bootstrap differently.** A fall is a real
  terminal state and bootstraps from zero; hitting the step limit is not, and
  bootstraps from the value of the final observation. Conflating them teaches
  the policy that the episode ending is worth zero, which is a silent bias
  toward exactly the behaviour stage W's fall penalty is trying to avoid.
* **The recurrent state is cleared at episode boundaries.** Otherwise the gate
  carries one episode's phase into the next.

Returns are normalised by a running standard deviation. Stage W pays up to 700
at the end against at most 0.30 per step, so without it the value function has
to span four orders of magnitude and the advantage estimates are dominated by
one number per episode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from myo_curriculum.env import MyoLocomotionEnv  # noqa: E402
from myo_curriculum.policy import (  # noqa: E402
    PhaseGatedActorCritic,
    balance_loss,
    confidence_loss,
    gait_state_dim,
    phase_alignment,
    switching_loss,
)
from myo_curriculum.stages import STAGE_ORDER  # noqa: E402


# -- vector env ------------------------------------------------------------


class SyncVecEnv:
    """Synchronous vector env that keeps the true final observation.

    Written out rather than using gymnasium's vector API because autoreset
    semantics differ across versions, and getting the terminal observation
    wrong silently corrupts truncation bootstrapping.
    """

    def __init__(self, stage: str, n: int, seed: int = 0, **overrides):
        self.envs = [
            MyoLocomotionEnv(stage=stage, seed=seed + i, **overrides) for i in range(n)
        ]
        self.n = n
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.act_dim = self.envs[0].n_act
        self.spec = self.envs[0].stage_spec
        self.layout = self.envs[0].obs_layout()
        self.dt = self.envs[0].dt
        self._returns = np.zeros(n)
        self._lengths = np.zeros(n, dtype=int)

    def reset(self, seed: int = 0):
        obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])
        self._returns[:] = 0.0
        self._lengths[:] = 0
        return obs

    def step(self, actions, seeds):
        obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        final = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        reward = np.zeros(self.n, dtype=np.float32)
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.zeros(self.n, dtype=bool)
        phase = np.zeros(self.n, dtype=np.float32)
        finished = []

        for i, env in enumerate(self.envs):
            o, r, term, trunc, _ = env.step(actions[i])
            reward[i] = r
            terminated[i] = term
            truncated[i] = trunc
            phase[i] = env.ref_phase
            self._returns[i] += r
            self._lengths[i] += 1
            if term or trunc:
                final[i] = o                      # the real next state
                finished.append((self._returns[i], self._lengths[i],
                                 env.travel, env.terminal_bonus))
                o = env.reset(seed=int(seeds[i]))[0]
                self._returns[i] = 0.0
                self._lengths[i] = 0
            obs[i] = o
        return obs, reward, terminated, truncated, final, phase, finished

    def close(self):
        for e in self.envs:
            e.close()


class RunningStd:
    """Welford estimate of the discounted-return scale."""

    def __init__(self, gamma: float, eps: float = 1e-8):
        self.gamma = gamma
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self._acc = None

    def update(self, rewards: np.ndarray, resets: np.ndarray) -> None:
        if self._acc is None:
            self._acc = np.zeros(rewards.shape[1])
        for t in range(rewards.shape[0]):
            self._acc = self._acc * self.gamma * (1.0 - resets[t]) + rewards[t]
            batch = self._acc
            n = batch.size
            delta = batch.mean() - self.mean
            tot = self.count + n
            self.mean += delta * n / tot
            self.var = (
                self.var * self.count + batch.var() * n + delta**2 * self.count * n / tot
            ) / tot
            self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + self.eps)


def save_checkpoint(payload: dict, path: Path, attempts: int = 5) -> bool:
    """Write a checkpoint atomically, and never let a failure kill the run.

    Two things go wrong otherwise, both seen in practice on Windows:

    * Writing in place fails with `ERROR_USER_MAPPED_FILE` (1224) while any
      reader has the file mapped -- `eval_checkpoint.py` reading the latest
      checkpoint is enough. Writing to a temporary file and renaming means the
      write itself never touches a file anyone else holds.
    * An interrupted in-place write leaves a truncated checkpoint, so a crash
      during saving costs the run rather than one iteration. `os.replace` is
      atomic, so readers see either the old file or the new one.

    Returns True on success. A failure is reported and training continues:
    losing one checkpoint is an inconvenience, losing 200 iterations to a file
    lock is not.
    """
    tmp = path.with_name(path.name + ".tmp")
    for i in range(attempts):
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            return True
        except Exception as exc:  # noqa: BLE001 - torch and os raise several
            last = exc
            time.sleep(0.5 * (i + 1))
    print("   [warn] could not save %s after %d attempts (%r) -- continuing"
          % (path.name, attempts, last))
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return False


class BestCheckpoints:
    """Keep a run's `keep` highest-scoring checkpoints, and no more.

    Only files this object itself wrote are ever deleted, and only while they
    are still in the run directory it was constructed with and still carry the
    `ckpt_best_` prefix. Pruning by globbing a directory is how two of this
    project's runs were destroyed; a list of files this process created cannot
    match anything it did not create. `ckpt_latest.pt` is not in that list and
    is never touched -- resuming needs the newest state, not the best-scoring
    one.

    Scores are offered at the save cadence rather than every iteration: a
    checkpoint is ~17 MB, and mean return over one 2048-step batch is noisy
    enough that the per-iteration maximum would mostly select for luck.
    """

    PREFIX = "ckpt_best_"

    def __init__(self, run: Path, keep: int = 10):
        self.run = run
        self.keep = max(1, int(keep))
        self.entries: list = []          # (score, path), worst score first

    def filename(self, score: float, iteration: int) -> str:
        return "%sit%06d_ret%+09.2f.pt" % (self.PREFIX, iteration, score)

    def _own(self, path: Path) -> bool:
        """Whether deleting `path` is something this object is allowed to do."""
        return (path.parent == self.run
                and path.name.startswith(self.PREFIX)
                and path.suffix == ".pt")

    def offer(self, score: float, iteration: int, payload: dict):
        """Save `payload` if `score` makes the top `keep`. Returns its path."""
        if score != score:               # NaN never displaces a real score
            return None
        if len(self.entries) >= self.keep and score <= self.entries[0][0]:
            return None

        path = self.run / self.filename(score, iteration)
        if not save_checkpoint(payload, path):
            return None

        self.entries.append((float(score), path))
        self.entries.sort(key=lambda e: e[0])
        while len(self.entries) > self.keep:
            _, drop = self.entries.pop(0)
            if drop == path or not self._own(drop):
                continue
            try:
                drop.unlink(missing_ok=True)
            except OSError as exc:       # a reader may hold it; try again later
                print("   [warn] could not prune %s (%r)" % (drop.name, exc))
                self.entries.insert(0, (float("-inf"), drop))
        return path

    def summary(self) -> str:
        if not self.entries:
            return "no best checkpoints"
        return "best %d of %d kept: ret %.2f .. %.2f" % (
            len(self.entries), self.keep,
            self.entries[0][0], self.entries[-1][0],
        )


def compute_gae(
    rewards: torch.Tensor,       # (T, N)
    values: torch.Tensor,        # (T, N)
    terminated: torch.Tensor,    # (T, N) 1.0 where the episode really ended
    truncated: torch.Tensor,     # (T, N) 1.0 where it hit the step limit
    final_values: torch.Tensor,  # (T, N) V(final obs) where truncated
    next_value: torch.Tensor,    # (N,)   V(obs after the last rollout step)
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Generalised advantage estimation, distinguishing the two done flags.

    `terminated` is a real terminal state -- a fall -- so the future is worth
    zero. `truncated` is the step limit, where the future is worth `V(final
    observation)` because the episode would have continued. Treating them
    alike teaches the policy that running out of time is as bad as falling,
    which biases it toward exactly the behaviour stage W's fall penalty exists
    to discourage.

    Either flag ends the GAE recursion, since the next step belongs to a
    different episode.
    """
    steps, n = rewards.shape
    adv = torch.zeros_like(values)
    last = torch.zeros(n, dtype=values.dtype)
    for t in reversed(range(steps)):
        bootstrap = next_value if t == steps - 1 else values[t + 1]
        bootstrap = torch.where(
            terminated[t] > 0.5, torch.zeros_like(bootstrap), bootstrap
        )
        bootstrap = torch.where(
            (truncated[t] > 0.5) & (terminated[t] < 0.5), final_values[t], bootstrap
        )
        done = torch.maximum(terminated[t], truncated[t])
        delta = rewards[t] + gamma * bootstrap - values[t]
        last = delta + gamma * lam * (1.0 - done) * last
        adv[t] = last
    return adv


# -- training --------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", default="W", choices=list(STAGE_ORDER))
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--minibatch-envs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.03)
    # phase gate
    p.add_argument("--phases", type=int, default=4,
                   help="1 gives a plain-MLP baseline with no gating")
    p.add_argument("--lambda-switch", type=float, default=0.01)
    p.add_argument("--lambda-balance", type=float, default=0.01)
    p.add_argument("--lambda-confidence", type=float, default=0.001)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.2)
    p.add_argument("--no-normalize-returns", action="store_true")
    # stage overrides, validated by StageSpec.with_overrides
    p.add_argument("--max-tracking-error", type=float, default=None,
                   help="RMS joint deviation that ends an episode (stage default 1.5)")
    p.add_argument("--forward-bonus", type=float, default=None,
                   help="per-step-equivalent weight of the terminal forward payment")
    p.add_argument("--fall-penalty", type=float, default=None)
    # bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=V5 / "runs")
    p.add_argument("--save-every", type=int, default=20, help="iterations")
    p.add_argument("--keep-best", type=int, default=10,
                   help="how many highest-return checkpoints a run keeps")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--torch-threads", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    overrides = {}
    if args.max_tracking_error is not None:
        overrides["max_tracking_error"] = args.max_tracking_error
    if args.forward_bonus is not None:
        overrides["forward_bonus"] = args.forward_bonus
    if args.fall_penalty is not None:
        overrides["fall_penalty"] = args.fall_penalty
    envs = SyncVecEnv(args.stage, args.num_envs, args.seed, **overrides)
    net = PhaseGatedActorCritic(
        envs.obs_dim, envs.act_dim, gait_state_dim(envs.layout), n_phases=args.phases
    )
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    ret_norm = None if args.no_normalize_returns else RunningStd(args.gamma)

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        # A different expert count is a different network: load_state_dict
        # would fail with a shape error that says nothing about the cause.
        saved_phases = int(ckpt.get("args", {}).get("phases", args.phases))
        if saved_phases != args.phases:
            raise SystemExit(
                "%s was trained with --phases %d, not %d"
                % (args.resume, saved_phases, args.phases)
            )
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        start_iter = ckpt["iteration"]
        if ret_norm and ckpt.get("ret_norm"):
            ret_norm.__dict__.update(ckpt["ret_norm"])
        print("resumed from %s at iteration %d" % (args.resume, start_iter))

    run = args.out / ("%s-p%d-%s" % (args.stage, args.phases,
                                     time.strftime("%y%m%d.%H%M%S")))
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )
    log_path = run / "log.csv"
    log_rows = []
    keeper = BestCheckpoints(run, args.keep_best)

    batch = args.num_envs * args.num_steps
    iterations = max(1, args.total_steps // batch)
    print("=" * 100)
    print(envs.envs[0].describe())
    print(envs.spec.describe())
    print("net: %.2fM params | %d phases%s"
          % (sum(p.numel() for p in net.parameters()) / 1e6, args.phases,
             "  (no gating: plain MLP baseline)" if args.phases == 1 else ""))
    print("batch %d = %d envs x %d steps | %d iterations | run %s"
          % (batch, args.num_envs, args.num_steps, iterations, run.name))
    print("=" * 100)

    obs = torch.as_tensor(envs.reset(args.seed), dtype=torch.float32)
    state = net.initial_state(args.num_envs)
    reset_flag = torch.ones(args.num_envs)
    ep_returns, ep_lengths, ep_travel = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)
    global_step = start_iter * batch
    t_start = time.perf_counter()

    for iteration in range(start_iter, start_iter + iterations):
        frac = iteration / max(1, start_iter + iterations - 1)
        tau = args.tau_start + frac * (args.tau_end - args.tau_start)
        for group in opt.param_groups:
            group["lr"] = args.lr * (1.0 - frac)

        # -- rollout -------------------------------------------------------
        buf_obs = torch.zeros(args.num_steps, args.num_envs, envs.obs_dim)
        buf_act = torch.zeros(args.num_steps, args.num_envs, envs.act_dim)
        buf_logp = torch.zeros(args.num_steps, args.num_envs)
        buf_val = torch.zeros(args.num_steps, args.num_envs)
        buf_rew = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
        buf_term = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
        buf_trunc = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
        buf_reset = torch.zeros(args.num_steps, args.num_envs)
        buf_phase = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
        buf_final_val = torch.zeros(args.num_steps, args.num_envs)
        state0 = (state[0].clone(), state[1].clone())

        for t in range(args.num_steps):
            buf_obs[t] = obs
            buf_reset[t] = reset_flag
            action, logp, value, _, state = net.act(obs, state, reset_flag, tau=tau)
            buf_act[t], buf_logp[t], buf_val[t] = action, logp, value

            seeds = rng.integers(0, 2**31 - 1, size=args.num_envs)
            nxt, rew, term, trunc, final, phase, finished = envs.step(
                action.numpy(), seeds
            )
            buf_rew[t], buf_term[t], buf_trunc[t] = rew, term, trunc
            buf_phase[t] = phase

            # Truncation is not a terminal state: bootstrap from the value of
            # the observation the episode actually ended on, not the reset one.
            if trunc.any():
                idx = np.flatnonzero(trunc & ~term)
                if idx.size:
                    with torch.no_grad():
                        fin = torch.as_tensor(final[idx], dtype=torch.float32)
                        # belief at the final step is approximated by the
                        # current one; the alternative is another gate pass
                        _, _, fv, _, _ = net.act(
                            fin, (state[0][:, idx], state[1][idx]),
                            torch.zeros(idx.size), tau=tau
                        )
                    buf_final_val[t, idx] = fv

            for ret, length, travel, _bonus in finished:
                ep_returns.append(ret)
                ep_lengths.append(length)
                ep_travel.append(travel)

            obs = torch.as_tensor(nxt, dtype=torch.float32)
            reset_flag = torch.as_tensor(
                (term | trunc).astype(np.float32), dtype=torch.float32
            )
        global_step += batch

        # -- advantages ----------------------------------------------------
        rewards = buf_rew.copy()
        if ret_norm is not None:
            ret_norm.update(rewards, np.maximum(buf_term, buf_trunc))
            rewards = rewards / ret_norm.std

        with torch.no_grad():
            _, _, next_value, _, _ = net.act(obs, state, reset_flag, tau=tau)
        rew_t = torch.as_tensor(rewards)
        term_t = torch.as_tensor(buf_term)
        trunc_t = torch.as_tensor(buf_trunc)
        final_v = buf_final_val / (ret_norm.std if ret_norm else 1.0)

        adv = compute_gae(
            rew_t, buf_val, term_t, trunc_t, final_v, next_value,
            args.gamma, args.gae_lambda,
        )
        returns = adv + buf_val

        # -- update --------------------------------------------------------
        env_ids = np.arange(args.num_envs)
        stats = {k: [] for k in ("pg", "v", "ent", "kl", "clip",
                                 "switch", "balance", "conf")}
        stop = False
        for _ in range(args.epochs):
            rng.shuffle(env_ids)
            for start in range(0, args.num_envs, args.minibatch_envs):
                mb = env_ids[start : start + args.minibatch_envs]
                # Whole sequences, recomputed from the segment's start state:
                # this is what keeps the ratio on-policy across epochs.
                sub_state = (state0[0][:, mb].contiguous(), state0[1][mb])
                out = net.evaluate(
                    buf_obs[:, mb], buf_act[:, mb], sub_state,
                    buf_reset[:, mb], tau=tau
                )
                logratio = out["log_prob"] - buf_logp[:, mb]
                ratio = logratio.exp()
                mb_adv = adv[:, mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip, 1 + args.clip),
                ).mean()
                v_loss = 0.5 * (out["value"] - returns[:, mb]).pow(2).mean()
                ent = out["entropy"].mean()

                lb = out["log_belief"]
                sw = switching_loss(lb)
                bal = balance_loss(lb)
                conf = confidence_loss(lb)

                loss = (
                    pg
                    + args.value_coef * v_loss
                    - args.entropy_coef * ent
                    + args.lambda_switch * sw
                    + args.lambda_balance * bal
                    + args.lambda_confidence * conf
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfrac = ((ratio - 1).abs() > args.clip).float().mean().item()
                stats["pg"].append(pg.item()); stats["v"].append(v_loss.item())
                stats["ent"].append(ent.item()); stats["kl"].append(approx_kl)
                stats["clip"].append(clipfrac); stats["switch"].append(sw.item())
                stats["balance"].append(bal.item()); stats["conf"].append(conf.item())
                if args.target_kl and approx_kl > args.target_kl:
                    stop = True
                    break
            if stop:
                break

        # -- log -----------------------------------------------------------
        mean = lambda k: float(np.mean(stats[k])) if stats[k] else float("nan")
        with torch.no_grad():
            diag = net.evaluate(buf_obs, buf_act, state0, buf_reset, tau=tau)
            belief = diag["belief"]
            usage = belief.reshape(-1, args.phases).mean(0).numpy()
            b_ent = float(-(belief * belief.clamp_min(1e-8).log()).sum(-1).mean())
        sps = int(global_step / (time.perf_counter() - t_start + 1e-9))

        row = {
            "iteration": iteration, "steps": global_step, "sps": sps, "tau": tau,
            "return_mean": float(np.mean(ep_returns)) if ep_returns else float("nan"),
            "return_max": float(np.max(ep_returns)) if ep_returns else float("nan"),
            "ep_len": float(np.mean(ep_lengths)) if ep_lengths else float("nan"),
            "travel": float(np.mean(ep_travel)) if ep_travel else float("nan"),
            "pg": mean("pg"), "value": mean("v"), "entropy": mean("ent"),
            "kl": mean("kl"), "clipfrac": mean("clip"),
            "switch": mean("switch"), "balance": mean("balance"),
            "confidence": mean("conf"), "belief_entropy": b_ent,
        }
        for k in range(args.phases):
            row["use_%d" % k] = float(usage[k])
        log_rows.append(row)

        if iteration % args.log_every == 0:
            print(
                "it %4d  %7d steps  %4d sps | ret %8.2f (max %8.2f)  len %6.1f  "
                "travel %+.3f m | pg %+.4f  v %.4f  kl %.4f  clip %.2f | "
                "tau %.2f  H(b) %.3f  use %s"
                % (iteration, global_step, sps, row["return_mean"], row["return_max"],
                   row["ep_len"], row["travel"], row["pg"], row["value"], row["kl"],
                   row["clipfrac"], tau, b_ent,
                   " ".join("%.2f" % u for u in usage))
            )

        if iteration % args.save_every == 0 or iteration == start_iter + iterations - 1:
            payload = {"net": net.state_dict(), "opt": opt.state_dict(),
                       "iteration": iteration, "args": vars(args),
                       "ret_norm": ret_norm.__dict__ if ret_norm else None}
            save_checkpoint(payload, run / "ckpt_latest.pt")
            kept = keeper.offer(row["return_mean"], iteration, payload)
            if kept is not None:
                print("   [best] %s  (%s)" % (kept.name, keeper.summary()))
            import csv

            try:
                with log_path.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
                    w.writeheader()
                    w.writerows(log_rows)
            except OSError as exc:
                print("   [warn] could not write %s (%r) -- continuing"
                      % (log_path.name, exc))

            if args.phases > 1:
                align = phase_alignment(
                    belief, torch.as_tensor(buf_phase), n_bins=args.phases
                )
                print("   phase vs reference cycle (rows = expert, cols = cycle bin):")
                for k in range(args.phases):
                    print("     %d  %s" % (k, "  ".join("%.2f" % v for v in align[k])))
                print("   transition matrix:")
                for k, r_ in enumerate(net.gate.transition_matrix().numpy()):
                    print("     %d  %s" % (k, "  ".join("%.2f" % v for v in r_)))

    envs.close()
    print("done. run directory: %s" % run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
