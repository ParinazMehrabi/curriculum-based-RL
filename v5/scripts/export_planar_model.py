"""Write the restructured model out as flat XML, for a MuJoCo without MjSpec.

    python scripts/export_planar_model.py
    python scripts/export_planar_model.py --check

Needs MuJoCo 3.2 or newer to run, and produces a model file that much older
versions can compile. `models/mjcf/myobody_planar.xml` is the result and is
committed, so a machine that cannot run MjSpec does not have to generate it.

**Why this exists.** On one 2014 Xeon, every MuJoCo from 3.2 up dies with an
access violation compiling even a single-sphere model, while 3.1.6 and 3.0.1
compile fine -- measured by `scripts/find_working_mujoco.py`, with nothing
injected into the process and a stripped PATH. But MjSpec, which
`MyoLocomotionEnv._planar_spec` uses to restructure MyoFullBody into a planar
model, only exists from 3.2. So the machine can run MuJoCo and can run this
project's physics; it just cannot perform the edits.

Editing the XML directly instead was the obvious alternative and is worse:
`myobody.xml` is 30 lines of `<include>`, and every joint, geom and body this
touches lives in one of seven included files, so it would mean writing an
include-inliner and keeping a second definition of the restructuring in step
with the first. Exporting from the one definition avoids both.

**One attribute has to be stripped.** `spec.to_xml()` writes `colorspace` on
`<texture>`, which MuJoCo 3.1 rejects as a schema violation. It is a colour
management hint with no effect on physics. `NEWER_ATTRIBUTES` lists what gets
removed, and `--check` reports whether the export still matches what MjSpec
builds, field by field.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from myo_curriculum.env import (  # noqa: E402
    EXPORTED_MODEL,
    MyoLocomotionEnv,
    _default_model_path,
)

# Attributes newer MuJoCo versions added that older ones reject. Physics is
# unaffected by every one of these; add to the list rather than to a version
# check, so one export serves every older version.
NEWER_ATTRIBUTES = ("colorspace",)

# Model fields that must agree between the two routes. Sizes first, because a
# mismatch there makes the rest meaningless.
COUNTS = ("nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nmesh", "neq",
          "ntendon", "nsite", "nkey")
ARRAYS = ("body_mass", "body_inertia", "jnt_type", "jnt_axis", "jnt_range",
          "geom_type", "geom_size", "geom_pos", "geom_contype",
          "geom_conaffinity", "geom_group", "eq_type", "eq_data",
          "actuator_gaintype", "actuator_gainprm", "dof_damping")


def export(source: Path) -> str:
    """The restructured model as flat XML, with newer attributes removed."""
    spec = MyoLocomotionEnv._planar_spec(source)
    spec.compile()                       # to_xml wants a compiled spec
    xml = spec.to_xml()
    for attr in NEWER_ATTRIBUTES:
        xml = re.sub(r'\s+%s="[^"]*"' % re.escape(attr), "", xml)
    return xml


# XML carries numbers as decimal text, so the two routes agree to the
# precision MuJoCo writes rather than exactly. Measured worst cases on this
# model: geom_size 4.4e-07, jnt_range 5.0e-07, geom_pos 7.1e-09,
# body_inertia 3.1e-11. A millimetre is 1e-3, so 1e-5 is loose enough to
# ignore serialisation and tight enough to catch a real edit going missing.
TOLERANCE = 1e-5

# eq_data is 11 wide and only the first five columns mean anything to a joint
# equality -- the polynomial. Column 10 is the weld torque scale, which the
# XML writer defaults to 1 and MjSpec leaves at 0, and which MuJoCo never
# reads for mjEQ_JOINT. Comparing it would report a difference that does not
# exist.
EQ_JOINT_COLUMNS = 5


def compare(a, b) -> list:
    """Every field in COUNTS and ARRAYS where two models really disagree."""
    bad = []
    for name in COUNTS:
        if getattr(a, name) != getattr(b, name):
            bad.append("%s: %s vs %s" % (name, getattr(a, name), getattr(b, name)))
    if bad:
        return bad                       # shapes differ; array comparison is moot

    for name in ARRAYS:
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if x.shape != y.shape:
            bad.append("%s shape: %s vs %s" % (name, x.shape, y.shape))
            continue
        if name == "eq_data":
            joints = np.asarray(a.eq_type) == int(mujoco.mjtEq.mjEQ_JOINT)
            x, y = x.copy(), y.copy()
            x[joints, EQ_JOINT_COLUMNS:] = 0.0
            y[joints, EQ_JOINT_COLUMNS:] = 0.0
        worst = float(np.abs(x - y).max()) if x.size else 0.0
        if worst > TOLERANCE:
            bad.append("%s differs, max |delta| %g" % (name, worst))
    return bad


def rollout_agrees(a, b, steps: int = 400, seed: int = 0):
    """Step both models under identical controls and compare the states.

    The decisive check: the fields can match and the model still behave
    differently if a constraint or a contact were dropped. Measured here the
    two agree to 5e-11 in qpos after 400 steps, which is double-precision
    drift from the XML's decimal text, not structure.
    """
    controls = np.random.default_rng(seed).uniform(0.0, 1.0, size=a.nu)
    states = []
    for model in (a, b):
        data = mujoco.MjData(model)
        adr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_ty")
        ]
        data.qpos[adr] = 0.95
        data.act[:] = 0.05
        data.ctrl[:] = controls
        for _ in range(steps):
            mujoco.mj_step(model, data)
        states.append((data.qpos.copy(), data.qvel.copy()))
    qpos = float(np.abs(states[0][0] - states[1][0]).max())
    qvel = float(np.abs(states[0][1] - states[1][1]).max())
    return qpos, qvel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=EXPORTED_MODEL)
    ap.add_argument("--check", action="store_true",
                    help="compare against MjSpec without writing")
    args = ap.parse_args(argv)

    if not hasattr(mujoco, "MjSpec"):
        raise SystemExit(
            "MuJoCo %s has no MjSpec; run this on 3.2 or newer"
            % mujoco.__version__
        )

    source = _default_model_path()
    print("source   %s" % source)
    print("mujoco   %s" % mujoco.__version__)

    xml = export(source)
    print("exported %d bytes, %d lines" % (len(xml), xml.count("\n") + 1))

    # Both routes, compared field by field. The exported file is what the
    # fallback actually compiles, so it is the thing to check, not the string.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    previous = args.out.read_bytes() if args.out.is_file() else None
    args.out.write_text(xml, encoding="utf-8")
    try:
        from_spec = MyoLocomotionEnv._planar_spec(source).compile()
        from_xml = MyoLocomotionEnv._compile_exported(source)
        differences = compare(from_spec, from_xml)
    finally:
        if args.check:
            if previous is None:
                args.out.unlink(missing_ok=True)
            else:
                args.out.write_bytes(previous)

    print("MjSpec   nq %d nv %d nu %d ngeom %d neq %d"
          % (from_spec.nq, from_spec.nv, from_spec.nu, from_spec.ngeom,
             from_spec.neq))
    print("flat XML nq %d nv %d nu %d ngeom %d neq %d"
          % (from_xml.nq, from_xml.nv, from_xml.nu, from_xml.ngeom, from_xml.neq))

    if differences:
        print("\n%d field(s) disagree:" % len(differences))
        for line in differences:
            print("  %s" % line)
        return 2

    qpos, qvel = rollout_agrees(from_spec, from_xml)
    print("\nidentical across %d counts and %d arrays (tolerance %g)"
          % (len(COUNTS), len(ARRAYS), TOLERANCE))
    print("400 steps under identical controls: max |dqpos| %.2e, |dqvel| %.2e"
          % (qpos, qvel))
    if qpos > 1e-6:
        print("that is too large to be serialisation -- something differs")
        return 2
    if args.check:
        print("--check: nothing written")
    else:
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
