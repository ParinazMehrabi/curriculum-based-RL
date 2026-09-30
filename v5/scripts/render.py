"""Render the muscle model, as a still figure or a rollout video.

    python v5/scripts/render.py                      # multi-view still
    python v5/scripts/render.py --mode rollout --stage B --steps 200
    python v5/scripts/render.py --out somewhere.png

Uses the model's own meshes and MyoSuite's scene, so the muscle paths show as
red tendons over the skeleton. The free camera is fine here: unlike the v4
MJCF, this model is z-up, which is what MuJoCo's camera assumes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

import mujoco  # noqa: E402

from myo_curriculum.env import MyoLocomotionEnv  # noqa: E402

DEFAULT_OUT = V5 / "figures" / "myo_model.png"


def make_camera(env, distance=2.4, azimuth=110.0, elevation=-6.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = env.data.subtree_com[0]
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def still(env, width, height):
    """Views of the reset stance, including a close-up of the foot contacts."""
    env.reset(seed=0)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    cam = make_camera(env)
    frames, labels = [], []
    # The model faces ~109 degrees, so offset the azimuths from that to get
    # recognisable front / side / back views rather than arbitrary ones.
    facing = np.degrees(env._heading_ref)
    views = (
        (0, 0.95, 2.4, -6, "front"),
        (90, 0.95, 2.4, -6, "side"),
        (180, 0.95, 2.4, -6, "back"),
        # Close on the feet: the orange discs are contact points, and they sit
        # under the heel as well as the forefoot. Before the stance solve the
        # model started on its toes with both heels 23 mm off the floor.
        (90, 0.10, 0.75, -12, "foot contacts"),
    )
    for azimuth, look_z, distance, elevation, label in views:
        cam.lookat[:] = env.data.subtree_com[0]
        cam.lookat[2] = look_z
        cam.distance = distance
        cam.azimuth = facing + azimuth
        cam.elevation = elevation
        renderer.update_scene(env.data, camera=cam, scene_option=options)
        frames.append(renderer.render())
        labels.append(label)
    renderer.close()
    return frames, labels


def rollout(env, width, height, steps, policy="low", seed=0):
    """Frames along a rollout, sampled evenly."""
    env.reset(seed=seed)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    rng = np.random.default_rng(seed)
    frames, labels = [], []
    keep = max(1, steps // 6)
    for i in range(steps):
        if policy == "zero":
            a = np.zeros(env.n_act, np.float32)
        elif policy == "random":
            a = rng.uniform(-1, 1, env.n_act).astype(np.float32)
        else:
            a = np.full(env.n_act, -0.6, np.float32)
        _, _, term, trunc, _ = env.step(a)
        if i % keep == 0:
            cam = make_camera(env, azimuth=np.degrees(env._heading_ref) + 90)
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())
            labels.append("t=%.2fs" % (env.steps * env.dt))
        if term or trunc:
            break
    renderer.close()
    return frames, labels


def compose(frames, labels, title, subtitle, footer, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, FG, MUTED = "#14161a", "#e8eaed", "#9aa0a6"
    n = len(frames)
    fig = plt.figure(figsize=(3.6 * n, 7.4), facecolor=BG)
    gs = fig.add_gridspec(1, n, wspace=0.02, top=0.845, bottom=0.26,
                          left=0.015, right=0.985)
    for i, (img, label) in enumerate(zip(frames, labels)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2c3036")
        ax.set_title(label, color=FG, fontsize=12, pad=9)
    fig.suptitle(title, color=FG, fontsize=17, y=0.955, fontweight="bold")
    fig.text(0.5, 0.895, subtitle, color=MUTED, fontsize=10.5, ha="center")
    fig.text(0.5, 0.115, footer, color=MUTED, fontsize=10.5, ha="center",
             linespacing=1.9, family="monospace")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=112, facecolor=BG)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="A")
    ap.add_argument("--mode", default="still", choices=("still", "rollout"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--policy", default="low", choices=("zero", "random", "low"))
    ap.add_argument("--width", type=int, default=620)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    env = MyoLocomotionEnv(stage=args.stage, seed=0)
    if args.mode == "still":
        frames, labels = still(env, args.width, args.height)
        title = "MyoSuite MyoFullBody — muscle-actuated locomotion model"
        sub = "myo_sim/body/myobody.xml (Apache-2.0)   ·   red strands are Hill-type muscle paths"
    else:
        frames, labels = rollout(env, args.width, args.height, args.steps, args.policy, 0)
        title = "MyoFullBody rollout — stage %s, %s policy" % (args.stage, args.policy)
        sub = "no trained policy: the model is falling, which is what an untrained muscle body does"

    footer = (
        "%d bodies · %d qpos · %d Hill-type muscles · %d activation states · %.1f kg\n"
        "17 independent joints; 29 more follow equality constraints (knee rolling contact, lumbar distribution)\n"
        "reset stance solved plantigrade to %.0e m, COM over the base of support\n"
        "contact load   heel_R %.2f   toe_R %.2f   heel_L %.2f   toe_L %.2f   (body weight)"
        % (env.model.nbody, env.model.nq, env.n_muscle, env.model.na,
           env.body_weight / 9.81,
           max(env.stance_residual, env.seat_residual),
           *env.contact_loads())
    )
    out = args.out or DEFAULT_OUT
    if args.mode == "rollout":
        out = out.with_name(out.stem + "_rollout" + out.suffix)
    path = compose(frames, labels, title, sub, footer, out)
    env.close()
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
