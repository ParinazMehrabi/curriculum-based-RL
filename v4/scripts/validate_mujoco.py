"""Check the MuJoCo model against the .hfd it was generated from.

`validate_env.py` exercises the environment; this checks the *model* -- that
the translation preserved what it should have, and that the places where
MuJoCo and Hyfydy genuinely differ are visible rather than buried.

    python v4/scripts/validate_mujoco.py
    python v4/scripts/validate_mujoco.py --steps 200

Needs numpy and mujoco. Needs neither gym, sconegym, sconepy nor a Hyfydy
licence, so it runs on any machine that can run the MuJoCo backend at all.

Exit code is 0 when every check passes, 1 otherwise. Warnings (lines marked
WARN) do not fail the run: they flag things worth knowing that are not defects
in the translation, including one pre-existing problem in the reference
trajectory that predates this backend and affects the SCONE runs equally.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_V4 = Path(__file__).resolve().parents[1]
REPO_ROOT = REPO_V4.parent
if str(REPO_V4) not in sys.path:
    sys.path.insert(0, str(REPO_V4))

import mujoco  # noqa: E402

from sconegym_crutch_v4.mujoco_backend import MujocoModel  # noqa: E402
from sconegym_crutch_v4.trajectory import load_sto  # noqa: E402

MJCF = REPO_ROOT / "models" / "mjcf" / "rajagopal_crutch_2d.xml"
INIT_STATE = REPO_ROOT / "models" / "init_states" / "InitState_A0_walk_003_v2.zml"
REFERENCE = REPO_ROOT / "models" / "reference" / "gaitTracking_solution_raw.sto"

# What the v4 README records for Hyfydy, from the neutral pose under zero
# torque. The only measurement of the original simulator available here.
HYFYDY_START_HEIGHT = 0.922
HYFYDY_HEIGHT_AT_36 = 0.858
HYFYDY_FALL_STEP = 73


def _converter():
    spec = importlib.util.spec_from_file_location(
        "_hfd_to_mjcf", REPO_V4 / "scripts" / "hfd_to_mjcf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if not ok:
            self.failures += 1
        print("  %-5s %-42s %s" % ("ok" if ok else "FAIL", label, detail))
        return ok

    def warn(self, label: str, detail: str = "") -> None:
        self.warnings += 1
        print("  %-5s %-42s %s" % ("WARN", label, detail))

    def note(self, label: str, detail: str = "") -> None:
        print("  %-5s %-42s %s" % ("", label, detail))


def section(title: str) -> None:
    print()
    print(title)
    print("-" * 78)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate the generated MuJoCo model.")
    ap.add_argument("--steps", type=int, default=140)
    args = ap.parse_args(argv)

    conv = _converter()
    hfd = conv.HfdModel(conv.parse_hfd(conv.DEFAULT_HFD))
    r = Report()

    print("=" * 78)
    print("MuJoCo model validation")
    print("=" * 78)
    print("mjcf      : %s" % MJCF)
    print("from      : %s" % conv.DEFAULT_HFD.name)
    print("init state: %s" % INIT_STATE.name)
    print("mujoco    : %s" % mujoco.__version__)

    # -- the generated file is current ------------------------------------
    section("1. the checked-in MJCF matches the .hfd")
    regenerated, _, _ = conv.build_mjcf(hfd, conv.DEFAULT_HFD)
    r.check(
        MJCF.read_text(encoding="utf-8") == regenerated,
        "XML is what the converter produces",
        "" if MJCF.is_file() else "missing",
    )

    model = MujocoModel(MJCF, init_state_path=INIT_STATE)

    # -- what must be preserved exactly -----------------------------------
    section("2. quantities the translation must preserve exactly")
    r.check(
        model.mass() == round(hfd.total_mass(), 6)
        or abs(model.mass() - hfd.total_mass()) < 1e-6,
        "total mass",
        "%.4f kg vs %.4f kg in the .hfd" % (model.mass(), hfd.total_mass()),
    )
    dof_names = [d.name() for d in model.dofs()]
    r.check(
        dof_names == [d.name for d in hfd.dofs],
        "dof table matches the .hfd order",
        "%d dofs" % len(dof_names),
    )
    r.check(
        tuple(a.name() for a in model.actuators()) == conv.ACTUATOR_ORDER,
        "actuator names and order",
        "%d actuators" % len(model.actuators()),
    )
    r.check(len(model.muscles()) == 0, "torque-only, no muscles")
    r.check(
        np.allclose(model.m.opt.gravity, [0.0, -9.81, 0.0]),
        "gravity is -Y (OpenSim frame kept)",
        "%s" % np.asarray(model.m.opt.gravity),
    )

    hfd_geoms = {g.name: g for g in hfd.geoms if g.type == "sphere"}
    mismatched = []
    for name, geom in hfd_geoms.items():
        gid = mujoco.mj_name2id(model.m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            mismatched.append("%s missing" % name)
            continue
        if abs(float(model.m.geom_size[gid][0]) - geom.radius) > 1e-9:
            mismatched.append("%s radius" % name)
        if not np.allclose(model.m.geom_pos[gid], geom.pos, atol=1e-9):
            mismatched.append("%s pos" % name)
    r.check(
        not mismatched,
        "contact geometry copied verbatim",
        "%d spheres%s" % (len(hfd_geoms), "" if not mismatched else "; " + ", ".join(mismatched)),
    )

    # -- planarity ---------------------------------------------------------
    section("3. the model stays planar")
    rng = np.random.RandomState(0)
    z_before = np.asarray([b.com_pos().z for b in model.bodies()])
    for _ in range(60):
        model.set_actuator_inputs(rng.uniform(-80, 80, len(model.actuators())))
        model.advance_simulation_to(model.time + 0.01)
    z_after = np.asarray([b.com_pos().z for b in model.bodies()])
    r.check(
        np.allclose(z_before, z_after, atol=1e-9),
        "no out-of-plane motion under torque",
        "max |dz| = %.2e m" % float(np.abs(z_after - z_before).max()),
    )

    # -- contact sensing ---------------------------------------------------
    section("4. contact sensing")
    model.reset()
    achieved = model.adjust_state_for_load(1.0)
    r.check(
        abs(achieved - 1.0) < 0.05,
        "adjust_state_for_load(1.0) carries body weight",
        "%.3f BW" % achieved,
    )
    feet = sum(model.find_body(b).contact_force().y for b in ("calcn_r", "calcn_l"))
    r.check(feet > 0.0, "ground reaction pushes the feet up", "%.1f N" % feet)
    # The onset floor is pose-dependent, so one measurement would mislead.
    traj_for_onset = load_sto(REFERENCE, dof_names, require_all=False)
    locked = set(model.locked_dofs())
    onsets = []
    for frame in (0, 55, 102, 134, 142, 200, 260):
        if frame >= traj_for_onset.n_frames:
            continue
        model.reset()
        q, _ = traj_for_onset.frame(frame)
        q = np.asarray(q, dtype=float).copy()
        for name in locked:
            q[dof_names.index(name)] = 0.0
        q[dof_names.index("pelvis_tx")] = 0.0
        model.set_dof_positions(q)
        model.set_dof_velocities(np.zeros(len(dof_names)))
        model.init_state_from_dofs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.adjust_state_for_load(0.5)
        if model._onset_fraction is not None:
            onsets.append(model._onset_fraction)
    if onsets:
        r.note(
            "contact onset floor, across poses",
            "%.2f .. %.2f BW -- init_load below this is unreachable"
            % (min(onsets), max(onsets)),
        )
        r.note(
            "",
            "init_load=0.5 reachable from %d of %d sampled reset poses"
            % (sum(1 for o in onsets if o <= 0.5), len(onsets)),
        )

    # -- the Hyfydy cross-check -------------------------------------------
    section("5. zero-torque collapse vs the Hyfydy measurement")
    model.reset()
    heights = [model.com_pos().y]
    fell_at = None
    for step in range(1, args.steps + 1):
        model.set_actuator_inputs(np.zeros(len(model.actuators())))
        model.advance_simulation_to(model.time + 0.01)
        heights.append(model.com_pos().y)
        if fell_at is None and heights[-1] < 0.55:
            fell_at = step
    r.note("", "%-22s %10s %10s" % ("", "mujoco", "hyfydy"))
    r.check(
        abs(heights[0] - HYFYDY_START_HEIGHT) < 0.03,
        "COM height at rest",
        "%-22s %10.3f %10.3f" % ("", heights[0], HYFYDY_START_HEIGHT),
    )
    r.check(
        len(heights) > 36 and abs(heights[36] - HYFYDY_HEIGHT_AT_36) < 0.05,
        "COM height after 36 steps",
        "%-22s %10.3f %10.3f" % ("", heights[36], HYFYDY_HEIGHT_AT_36),
    )
    r.check(
        fell_at is not None and 55 <= fell_at <= 95,
        "step at which it falls",
        "%-22s %10s %10d" % ("", fell_at, HYFYDY_FALL_STEP),
    )

    # -- the reference trajectory vs the model's own joint limits ----------
    section("6. reference trajectory against the model's joint limits")
    print("  The RSI stages reset the model to frames of this file, so a frame")
    print("  outside a joint's range starts the episode fighting a limit.")
    print()
    traj = load_sto(REFERENCE, dof_names, require_all=False)
    q = np.asarray(traj.q)
    offenders = []
    for i, name in enumerate(dof_names):
        jid = mujoco.mj_name2id(model.m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or not model.m.jnt_limited[jid]:
            continue
        lo, hi = model.m.jnt_range[jid]
        col = q[:, i]
        below = int((col < lo - 1e-9).sum())
        above = int((col > hi + 1e-9).sum())
        if below or above:
            worst = max(float(lo - col.min()), float(col.max() - hi))
            offenders.append((name, lo, hi, col.min(), col.max(), below + above, worst))

    if not offenders:
        r.check(True, "every frame lies inside the joint limits")
    else:
        print("  %-17s %8s %8s %8s %8s %7s %8s" %
              ("dof", "lim_lo", "lim_hi", "ref_min", "ref_max", "frames", "worst"))
        for name, lo, hi, cmin, cmax, n, worst in offenders:
            print("  %-17s %8.3f %8.3f %8.3f %8.3f %7d %8.3f"
                  % (name, lo, hi, cmin, cmax, n, worst))
        print()
        r.warn(
            "reference violates joint limits",
            "%d dofs, up to %.3f rad past the stop"
            % (len(offenders), max(o[6] for o in offenders)),
        )
        knee = [o for o in offenders if o[0].startswith("knee")]
        if knee and all(o[3] > 0 for o in knee):
            print("  The knees are the striking case: the reference is positive")
            print("  throughout (Rajagopal2015's convention, positive = flexion)")
            print("  while this model's range is -120..5 deg (negative = flexion).")
            print("  The two disagree about the sign of knee flexion, so every RSI")
            print("  reset hyperextends the knee instead of flexing it.")
            print()
            print("  This is not an artefact of the MuJoCo port -- the .hfd and the")
            print("  .sto are both inputs, and the SCONE runs read the same pair.")
            print("  Worth resolving before reading much into any stage's posture")
            print("  term. Nothing here changes it, because flipping the sign would")
            print("  redefine the whole curriculum and invalidate existing runs.")

    # -- summary -----------------------------------------------------------
    print()
    print("=" * 78)
    if r.failures:
        print("FAIL -- %d check(s) failed, %d warning(s)" % (r.failures, r.warnings))
    else:
        print("PASS -- all checks passed, %d warning(s)" % r.warnings)
    print("=" * 78)
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
