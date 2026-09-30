"""Run a trained checkpoint and print the reward detail of the best episode.

    python scripts/eval_checkpoint.py
    python scripts/eval_checkpoint.py --episodes 40 --every 10
    python scripts/eval_checkpoint.py --ckpt runs/W-p4-260930.090758/ckpt_latest.pt

Runs `--episodes` episodes and prints the per-step reward breakdown of the one
with the highest return. Nothing else.

The best episode is rendered automatically to `render/<run>_it<N>_ret<R>.mp4`,
under a name no existing file has, so repeated evaluations never overwrite each
other. `--no-render` skips it.

Safe to run against a training job: the checkpoint is copied before loading,
so this process never holds the trainer's file open.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from myo_curriculum.env import MyoLocomotionEnv  # noqa: E402
from myo_curriculum.policy import PhaseGatedActorCritic, gait_state_dim  # noqa: E402


def newest_checkpoint(root: Path) -> Path:
    runs = sorted(root.glob("*/ckpt_latest.pt"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(
            "no checkpoints under %s -- pass --ckpt explicitly" % root
        )
    return runs[-1]


def load_checkpoint(path: Path, attempts: int = 5):
    """Load a copy, so the trainer's own file is never held open.

    Loading in place is not safe against a running job on Windows: torch.load
    keeps a mapped section on the file and the trainer's next save then fails
    with ERROR_USER_MAPPED_FILE (1224).
    """
    last = None
    for i in range(attempts):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
                tmp = Path(fh.name)
            shutil.copy2(path, tmp)
            return torch.load(tmp, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - torch and shutil both raise
            last = exc
            time.sleep(0.4 * (i + 1))
        finally:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    raise RuntimeError("could not read %s (%r)" % (path, last))


def render_name(ckpt_path: Path, iteration, ret: float) -> str:
    """A name carrying which run, which checkpoint and how well it scored."""
    run = ckpt_path.parent.name or "run"
    it = "it%06d" % int(iteration) if iteration is not None else ckpt_path.stem
    return "%s_%s_ret%+09.2f" % (run, it, ret)


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """`directory/stem+suffix`, with a counter appended if that name is taken.

    Evaluations of the same checkpoint would otherwise overwrite the previous
    video, which is the one case where the name alone is not unique.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (stem + suffix)
    n = 2
    while path.exists():
        path = directory / ("%s_%d%s" % (stem, n, suffix))
        n += 1
    return path


def render_episode(env, qpos_frames, path: Path, fps: int, width: int, height: int):
    """Replay a recorded episode into the model and write it out.

    The episode is replayed from its stored `qpos` rather than re-simulated,
    so the video is exactly the episode that was scored -- re-running would
    diverge on any nondeterminism.
    """
    import mujoco

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    options = mujoco.MjvOption()
    options.geomgroup[4] = 1  # MyoSuite's collision group: the contact balls
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.6
    camera.azimuth = 90.0     # sagittal: the plane the model moves in
    camera.elevation = -6.0

    frames = []
    for qpos in qpos_frames:
        env.data.qpos[:] = qpos
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        camera.lookat[:] = env.data.subtree_com[0]
        camera.lookat[2] = 0.9        # keep the body framed, not the feet
        renderer.update_scene(env.data, camera=camera, scene_option=options)
        frames.append(renderer.render())
    renderer.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=int(1000 / fps), loop=0)
    else:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
    return len(frames)


def contributions(env, term_values):
    """Each term's contribution to the step reward, and its maximum."""
    spec = env.stage_spec.reward
    weights = spec.active_weights
    total_w = sum(weights.values()) or 1.0
    out = [("alive", spec.alive, spec.alive)]
    for name in sorted(weights):
        cap = spec.shaping_scale * weights[name] / total_w
        out.append((name, cap * term_values.get(name, 0.0), cap))
    return out


def pct(value, cap):
    return 100.0 * value / cap if cap > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--runs", type=Path, default=V5 / "runs")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--every", type=int, default=1, help="print every Nth step")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--stage", default=None, help="override the checkpoint's stage")
    ap.add_argument("--sample", action="store_true",
                    help="sample actions instead of using the distribution mean")
    ap.add_argument("--render", type=Path, default=None,
                    help="write the best episode here instead of render/")
    ap.add_argument("--render-dir", type=Path, default=V5 / "render",
                    help="where auto-named videos go")
    ap.add_argument("--no-render", action="store_true",
                    help="print the reward detail only")
    ap.add_argument("--fps", type=int, default=0, help="default is realtime (1/dt)")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=640)
    args = ap.parse_args(argv)

    ckpt_path = args.ckpt or newest_checkpoint(args.runs)
    ckpt = load_checkpoint(ckpt_path)
    saved = ckpt.get("args", {})
    stage = args.stage or saved.get("stage", "W")
    phases = int(saved.get("phases", 4))
    tau = float(saved.get("tau_end", 0.2))

    env = MyoLocomotionEnv(stage=stage, seed=args.seed)
    net = PhaseGatedActorCritic(
        env.observation_space.shape[0], env.n_act,
        gait_state_dim(env.obs_layout()), n_phases=phases,
    )
    net.load_state_dict(ckpt["net"])
    net.eval()

    rendering = not args.no_render

    max_step = env.stage_spec.reward.alive + env.stage_spec.reward.shaping_scale
    max_terminal = env.stage_spec.forward_bonus * env.stage_spec.episode_steps
    max_episode = max_step * env.stage_spec.episode_steps + max_terminal

    best = None
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = net.initial_state(1)
        reset_flag = torch.ones(1)
        record, total = [], 0.0
        while True:
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, state = net.act(
                    o, state, reset_flag, tau=tau, deterministic=not args.sample
                )
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
            total += reward
            record.append({
                "step": env.steps, "reward": reward,
                "terms": dict(env.term_values), "terminal": env.terminal_bonus,
                # Forward progress is paid once at the end, so per step show
                # what it is worth so far -- otherwise the biggest term in the
                # reward is invisible until the last line.
                "forward": env.forward_bonus(), "travel": env.travel,
                "velocity": env.forward_velocity(),
                "qpos": env.data.qpos.copy() if rendering else None,
            })
            reset_flag = torch.zeros(1)
            if terminated or truncated:
                break
        if best is None or total > best["return"]:
            best = {"seed": args.seed + ep, "return": total,
                    "length": env.steps, "travel": env.travel,
                    "terminal": env.terminal_bonus, "record": record}

    caps = contributions(env, {})
    print("best of %d episodes (seed %d): return %.2f of %.0f, %d steps"
          % (args.episodes, best["seed"], best["return"], max_episode, best["length"]))
    print("max:  per step " + "  ".join("%s %.2f" % (n, c) for n, _, c in caps)
          + " = %.2f   |   at the end  forward %.0f" % (max_step, max_terminal))
    print("-" * 112)
    for row in best["record"]:
        if row["step"] % args.every and row is not best["record"][-1]:
            continue
        parts = "  ".join(
            "%s %.3f (%5.1f%%)" % (name, value, pct(value, cap))
            for name, value, cap in contributions(env, row["terms"])
        )
        shown = row["reward"] - row["terminal"]
        print("step %4d: reward %+.4f (%5.1f%%): %s | forward %7.2f (%5.1f%%)  "
              "travel %+.3f m  v %+.3f m/s"
              % (row["step"], shown, pct(shown, max_step), parts,
                 row["forward"], pct(row["forward"], max_terminal),
                 row["travel"], row["velocity"]))
    print("-" * 112)
    print("return %.2f = %.2f per-step + %.2f forward  (%.1f%% of %.0f)"
          % (best["return"], best["return"] - best["terminal"], best["terminal"],
             pct(best["return"], max_episode), max_episode))

    if rendering:
        if args.render is not None:
            out = args.render
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out = unique_path(
                args.render_dir,
                render_name(ckpt_path, ckpt.get("iteration"), best["return"]),
                ".mp4",
            )
        fps = args.fps or int(round(1.0 / env.dt))
        n = render_episode(env, [r["qpos"] for r in best["record"]],
                           out, fps, args.width, args.height)
        print("wrote %s  (%d frames, %d fps, %.2f s)"
              % (out, n, fps, n / fps))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
