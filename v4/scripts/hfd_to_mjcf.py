"""Translate the Hyfydy .hfd model into MuJoCo MJCF.

The .hfd file stays the source of truth. Run this to regenerate the XML
whenever the model changes; do not hand-edit models/mjcf/*.xml.

    python v4/scripts/hfd_to_mjcf.py

Frame convention
----------------
The MJCF keeps OpenSim's frame -- **X forward, Y up, Z to the subject's right**
-- rather than converting to MuJoCo's usual Z-up. Gravity is set to
(0, -9.81, 0) and the floor plane's normal to +Y.

This is deliberate, and it is why the translation is exact. Every position,
every joint axis and every sign copies over verbatim: hip flexion stays
positive-forward, knee flexion stays negative, pelvis tilt stays
negative-for-forward-lean. It keeps `_vec_x` meaning "forward" and `_vec_y`
meaning "height" in env.py, and it lets the OpenSim Moco reference trajectory
(models/reference/*.sto) be written straight into qpos untransformed.
Converting to Z-up would have meant re-deriving all of that for no gain.

The only cost is cosmetic: MuJoCo's free camera assumes Z-up, so the emitted
file carries an explicit camera and a Y-up floor.

Body frames are COM-centred
---------------------------
In the .hfd, joint offsets are given relative to each segment's COM
(`pos_in_parent` is the joint in the parent frame minus the parent COM;
`pos_in_child` likewise), and contact geometry uses the same COM-centred
frame. So every MuJoCo body frame here sits at the segment COM, which makes
`<inertial pos="0 0 0">` correct and lets contact positions copy across
unchanged. A child body's offset within its parent is then
`pos_in_parent - pos_in_child`, and its joint sits at `pos_in_child`.

Deliberate departures from the .hfd, each reported when applied
---------------------------------------------------------------
* **Locked joints become rigid.** ankle_r/l and mtp_r/l have `limits 0..0` and
  no motor, so they are emitted as jointless (welded) bodies rather than as
  zero-width hinges the solver would have to police every step. The four dofs
  still appear in the backend's dof table as constant zeros, so N_DOF stays 16
  and env.py's LOCKED_DOFS indexing is unchanged.
* **Two inertia tensors are rebalanced.** MuJoCo requires A+B>=C; forearm_r/l
  (0.010, 0.002, 0.021) and toes_r/l (0.0001, 0.0002, 0.0010) violate it. The
  two smaller components are scaled up uniformly until the tensor is valid,
  which preserves their ratio and leaves the largest -- I_z, the sagittal axis
  and the *only* one carrying dynamics in a planar model -- exactly as written.
  The correction is therefore dynamically inert.
* **Meshes become capsules.** The .vtp/.STL assets are not in this repository.
  Visual geoms are generated from joint anchors and carry contype="0"
  conaffinity="0", so they are cosmetic and cannot affect physics. All contact
  geometry is declared explicitly in the .hfd and is reproduced exactly.
* **Contact compliance is not matched.** Hyfydy's material (stiffness 11006.4,
  damping 1) and MuJoCo's soft-constraint solref/solimp are different models
  with no exact correspondence. Friction carries over as the static
  coefficient, the closest analogue of MuJoCo's friction-cone bound. Expect
  sim-to-sim differences here before anywhere else.

Joint limits, by contrast, *are* translated: `model_options.joint_limit_stiffness`
(500 N.m/rad) becomes a negative `solreflimit`, which MuJoCo reads as an
explicit (stiffness, damping) pair, instead of MuJoCo's near-rigid default.
This matters more than it looks. The reference trajectory that the RSI stages
reset from lies **outside** the knee and hip limits on most frames -- the knees
on every frame, by up to 0.59 rad, because the .sto uses Rajagopal2015's
positive-is-flexion convention while the .hfd's -120..5 range is
negative-is-flexion. A rigid MuJoCo limit would fight every reset with
effectively unbounded torque, where Hyfydy's soft limit pushes back with a
finite one. `validate_mujoco.py` reports the conflict in full; resolving it is
a decision about the model, not about this translation.
The limit damping is the one number here chosen rather than translated, since
the .hfd gives none; `--limit-damping` overrides it.

Angle units: the .hfd gives joint `limits` and dof `range` in **degrees** but
dof `default` values in **radians** (lumbar -0.05, elbow 0.80). Both are
handled. Translational dofs (pelvis_tx/ty) are metres throughout.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HFD = (
    REPO_ROOT
    / "models"
    / "hfd"
    / "Rajagopal2015_crutch_2D_ankles_locked_mesh_lumbar.hfd"
)
DEFAULT_OUT = REPO_ROOT / "models" / "mjcf" / "rajagopal_crutch_2d.xml"

# model_options.joint_limit_stiffness in the .hfd, in N.m/rad. Hyfydy's joint
# limits are soft springs, not hard stops, and that matters here: the reference
# trajectory drives the knees and hips past their declared limits on every
# frame (see the module docstring), so a rigid MuJoCo limit would fight every
# RSI reset with effectively unbounded torque. The .hfd's own value is used.
DEFAULT_LIMIT_STIFFNESS = 500.0
# The .hfd gives no limit damping. This is near-critical for a limb segment of
# ~0.2-0.5 kg.m^2 against that stiffness (2*sqrt(k*I) ~ 20-32) and is the one
# contact/limit number here that is chosen rather than translated.
DEFAULT_LIMIT_DAMPING = 25.0

WORLD_BODY = "ground"
ROOT_BODY = "pelvis"

ROOT_DOFS = ("pelvis_tx", "pelvis_ty", "pelvis_tilt")
TRANSLATIONAL_DOFS = {"pelvis_tx", "pelvis_ty"}

# Emitted in this order so the model's actuator list matches
# CrutchCurriculumGym.ACTUATOR_NAMES exactly; _validate_model() asserts it.
ACTUATOR_ORDER = (
    "lumbar_extension",
    "hip_flexion_r",
    "knee_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "arm_flex_r",
    "elbow_flex_r",
    "arm_flex_l",
    "elbow_flex_l",
)


# --------------------------------------------------------------------------
# .hfd parsing
# --------------------------------------------------------------------------

Block = List[Tuple[str, object]]  # value is either a str or a nested Block


def tokenize(text: str) -> List[str]:
    text = re.sub(r"#[^\n]*", " ", text)
    for ch in "{}=":
        text = text.replace(ch, " " + ch + " ")
    return text.split()


def parse_block(tokens: List[str], i: int) -> Tuple[Block, int]:
    """Parse tokens from i until the matching '}'."""
    out: Block = []
    while i < len(tokens):
        if tokens[i] == "}":
            return out, i + 1
        key = tokens[i]
        i += 1
        if i < len(tokens) and tokens[i] == "=":
            i += 1
        if i < len(tokens) and tokens[i] == "{":
            sub, i = parse_block(tokens, i + 1)
            out.append((key, sub))
        else:
            out.append((key, tokens[i]))
            i += 1
    return out, i


def parse_hfd(path: Path) -> Block:
    tokens = tokenize(path.read_text(encoding="utf-8"))
    if not tokens or tokens[0] != "model":
        raise ValueError("%s does not start with a `model` block" % path)
    start = 2 if tokens[1] == "{" else 3
    block, _ = parse_block(tokens, start)
    return block


def items(block: Block, key: str) -> List[object]:
    return [v for k, v in block if k == key]


def one(block: Optional[Block], key: str, default=None):
    if block is None:
        return default
    vals = items(block, key)
    return vals[0] if vals else default


def vec3(block, default=(0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    if block is None:
        return default
    return (
        float(one(block, "x", 0.0)),
        float(one(block, "y", 0.0)),
        float(one(block, "z", 0.0)),
    )


def parse_range(text: str) -> Tuple[float, float]:
    lo, hi = text.split("..")
    return float(lo), float(hi)


def fmt(*values: float) -> str:
    return " ".join(("%.6g" % v) for v in values)


# --------------------------------------------------------------------------
# model structures
# --------------------------------------------------------------------------


class Joint:
    def __init__(self, block: Block):
        self.name = str(one(block, "name"))
        self.parent = str(one(block, "parent"))
        self.pos_in_parent = vec3(one(block, "pos_in_parent"))
        self.pos_in_child = vec3(one(block, "pos_in_child"))
        limits = one(block, "limits")
        self.range_z = parse_range(str(one(limits, "z", "0..0")))

    @property
    def locked(self) -> bool:
        return self.range_z[0] == self.range_z[1]


class Geom:
    def __init__(self, block: Block):
        self.name = str(one(block, "name"))
        self.type = str(one(block, "type"))
        self.body = str(one(block, "body", WORLD_BODY))
        self.radius = float(one(block, "radius", 0.0))
        self.pos = vec3(one(block, "pos"))


class Body:
    def __init__(self, block: Block):
        self.name = str(one(block, "name"))
        self.mass = float(one(block, "mass", 0.0))
        self.inertia = vec3(one(block, "inertia"))
        jb = one(block, "joint")
        self.joint: Optional[Joint] = Joint(jb) if jb else None
        self.children: List["Body"] = []
        self.geoms: List[Geom] = []
        self.inertia_note: Optional[str] = None

    def balanced_inertia(self) -> Tuple[float, float, float]:
        """Scale the two smaller components up until A+B>=C.

        The largest component is left untouched. For both bodies that need
        this, the largest is I_z -- the sagittal axis, and the only one that
        carries dynamics in this planar model -- so the correction exists
        purely to satisfy MuJoCo's validator and changes no trajectory.
        """
        order = sorted(range(3), key=lambda k: self.inertia[k])
        a, b, c = (self.inertia[k] for k in order)
        if a + b >= c or c <= 0.0:
            return self.inertia
        scale = c / (a + b)
        fixed = list(self.inertia)
        fixed[order[0]] = a * scale
        fixed[order[1]] = b * scale
        self.inertia_note = "(%s) -> (%s); largest component preserved" % (
            fmt(*self.inertia),
            fmt(*fixed),
        )
        return (fixed[0], fixed[1], fixed[2])


class Dof:
    def __init__(self, block: Block):
        self.name = str(one(block, "name"))
        self.source = str(one(block, "source"))
        self.range = parse_range(str(one(block, "range", "0..0")))
        default = one(block, "default")
        self.default = float(default) if default is not None else 0.0

    @property
    def translational(self) -> bool:
        return self.name in TRANSLATIONAL_DOFS

    @property
    def range_rad(self) -> Tuple[float, float]:
        """Ranges are degrees in the .hfd; defaults are already radians."""
        if self.translational:
            return self.range
        return (math.radians(self.range[0]), math.radians(self.range[1]))

    def joint_name(self) -> Optional[str]:
        """The .hfd joint this dof drives, or None for the floating root."""
        if self.source.startswith("pelvis_"):
            return None
        m = re.match(r"^(?P<j>.+?)_(?:rx|ry|rz)(?:_(?P<side>[rl]))?$", self.source)
        if not m:
            raise ValueError("cannot map dof source %r onto a joint" % self.source)
        side = m.group("side")
        return m.group("j") + ("_" + side if side else "")


class HfdModel:
    def __init__(self, block: Block):
        self.material = one(block, "material") or []
        self.options = one(block, "model_options") or []
        self.bodies = [Body(b) for b in items(block, "body")]
        self.geoms = [Geom(g) for g in items(block, "geometry")]
        self.dofs = [Dof(d) for d in items(block, "dof")]
        self.motors = {str(one(m, "joint")): m for m in items(block, "joint_motor")}
        self.by_name = {b.name: b for b in self.bodies}

        for g in self.geoms:
            if g.body in self.by_name:
                self.by_name[g.body].geoms.append(g)
        for b in self.bodies:
            if b.joint is not None:
                self.by_name[b.joint.parent].children.append(b)

        self.root = self.by_name[ROOT_BODY]
        self.dof_by_name = {d.name: d for d in self.dofs}
        self.dof_for_joint: Dict[str, Dof] = {}
        self.body_for_joint: Dict[str, Body] = {}
        for d in self.dofs:
            jn = d.joint_name()
            if jn is not None:
                self.dof_for_joint[jn] = d
        for b in self.bodies:
            if b.joint is not None:
                self.body_for_joint[b.joint.name] = b

    def motor_for(self, joint_name: Optional[str]):
        return self.motors.get(joint_name) if joint_name else None

    def is_locked_dof(self, dof: Dof) -> bool:
        jn = dof.joint_name()
        if jn is None:
            return False
        body = self.body_for_joint.get(jn)
        return bool(body and body.joint and body.joint.locked)

    def total_mass(self) -> float:
        return sum(b.mass for b in self.bodies)


# --------------------------------------------------------------------------
# MJCF emission
# --------------------------------------------------------------------------


def capsule_radius(mass: float) -> float:
    return min(0.09, max(0.012, 0.045 * (mass / 8.0) ** (1.0 / 3.0)))


def visual_geoms(body: Body) -> List[str]:
    """Stand-in geometry drawn between the body's anchors.

    Cosmetic only -- the "visual" class sets contype/conaffinity to 0, so these
    cannot collide with anything. Real contact geometry comes from the .hfd's
    own `geometry` blocks and is emitted separately.
    """
    proximal = body.joint.pos_in_child if body.joint is not None else (0.0, 0.0, 0.0)
    anchors = [c.joint.pos_in_parent for c in body.children if c.joint is not None]
    anchors += [g.pos for g in body.geoms]

    radius = capsule_radius(body.mass)
    out: List[str] = []
    for target in anchors:
        span = math.sqrt(sum((proximal[k] - target[k]) ** 2 for k in range(3)))
        if span < 1e-4:
            continue
        out.append(
            '<geom type="capsule" size="%s" fromto="%s" class="visual"/>'
            % (fmt(radius), fmt(*(tuple(proximal) + tuple(target))))
        )
    if not out:
        out.append(
            '<geom type="sphere" size="%s" pos="0 0 0" class="visual"/>' % fmt(radius)
        )
    return out


def emit_body(
    model: HfdModel,
    body: Body,
    indent: int,
    notes: List[str],
    joint_order: List[str],
    limit_stiffness: float,
    limit_damping: float,
) -> List[str]:
    """Emit one body and its subtree, recording MJCF joint order as it goes.

    `joint_order` accumulates joint names in the order MuJoCo will lay them out
    in qpos -- which follows the body tree, not the .hfd's dof declaration
    order. The keyframe and the backend's index map both depend on it.
    """
    pad = "  " * indent
    lines: List[str] = []

    if body.joint is not None:
        offset = tuple(
            body.joint.pos_in_parent[k] - body.joint.pos_in_child[k] for k in range(3)
        )
    else:
        offset = (0.0, 0.0, 0.0)

    lines.append('%s<body name="%s" pos="%s">' % (pad, body.name, fmt(*offset)))

    if body.name == ROOT_BODY:
        # Floating root restricted to the sagittal plane. The slide joints hold
        # absolute world position, so qpos[pelvis_ty] *is* the pelvis height --
        # what env.py's height term and the .sto reference both assume.
        for dof_name, kind, axis in (
            ("pelvis_tx", "slide", "1 0 0"),
            ("pelvis_ty", "slide", "0 1 0"),
            ("pelvis_tilt", "hinge", "0 0 1"),
        ):
            lines.append(
                '%s  <joint name="%s" type="%s" axis="%s"/>' % (pad, dof_name, kind, axis)
            )
            joint_order.append(dof_name)
    elif body.joint is not None and not body.joint.locked:
        dof = model.dof_for_joint[body.joint.name]
        lo, hi = dof.range_rad
        motor = model.motor_for(body.joint.name)
        damping = float(one(motor, "damping", 0.0)) if motor else 0.0
        lines.append(
            '%s  <joint name="%s" type="hinge" axis="0 0 1" pos="%s" range="%s"'
            ' damping="%s" solreflimit="%s"/>'
            % (
                pad,
                dof.name,
                fmt(*body.joint.pos_in_child),
                fmt(lo, hi),
                fmt(damping),
                fmt(-limit_stiffness, -limit_damping),
            )
        )
        joint_order.append(dof.name)
    elif body.joint is not None:
        dof = model.dof_for_joint.get(body.joint.name)
        # Two kinds of rigid joint here: ankle/mtp are locked dofs that still
        # have to appear in the 16-entry dof table, while the crutch welds
        # never had a dof at all.
        notes.append(
            "locked joint %-14s -> rigid weld (%s)"
            % (
                body.joint.name,
                "dof %s stays a constant 0" % dof.name if dof else "no dof, pure weld",
            )
        )

    ix, iy, iz = body.balanced_inertia()
    if body.inertia_note:
        notes.append("inertia %-11s -> %s" % (body.name, body.inertia_note))
        lines.append("%s  <!-- inertia rebalanced: %s -->" % (pad, body.inertia_note))
    lines.append(
        '%s  <inertial pos="0 0 0" mass="%s" diaginertia="%s"/>'
        % (pad, fmt(body.mass), fmt(ix, iy, iz))
    )

    for geom in visual_geoms(body):
        lines.append("%s  %s" % (pad, geom))

    for geom in body.geoms:
        if geom.type != "sphere":
            raise ValueError(
                "unsupported contact geometry type %r on %s" % (geom.type, body.name)
            )
        lines.append(
            '%s  <geom name="%s" type="sphere" size="%s" pos="%s" class="contact"/>'
            % (pad, geom.name, fmt(geom.radius), fmt(*geom.pos))
        )

    for child in body.children:
        lines.extend(
            emit_body(
                model, child, indent + 1, notes, joint_order,
                limit_stiffness, limit_damping,
            )
        )

    lines.append("%s</body>" % pad)
    return lines


def build_mjcf(
    model: HfdModel,
    source: Path,
    limit_damping: float = DEFAULT_LIMIT_DAMPING,
) -> Tuple[str, List[str], List[str]]:
    notes: List[str] = []
    joint_order: List[str] = []
    friction = float(one(model.material, "static_friction", 0.9))
    limit_stiffness = float(
        one(model.options, "joint_limit_stiffness", DEFAULT_LIMIT_STIFFNESS)
    )

    body_lines = emit_body(
        model, model.root, 2, notes, joint_order, limit_stiffness, limit_damping
    )
    notes.append(
        "joint limits  -> solreflimit=(-%g, -%g): the .hfd's joint_limit_stiffness, "
        "with a damping chosen for near-critical response"
        % (limit_stiffness, limit_damping)
    )

    # qpos follows MJCF joint order, not the .hfd's dof declaration order.
    neutral = [model.dof_by_name[name].default for name in joint_order]

    lines: List[str] = []
    lines.append('<mujoco model="rajagopal_crutch_2d">')
    lines.append("  <!--")
    lines.append("    GENERATED by v4/scripts/hfd_to_mjcf.py from")
    lines.append("      %s" % source.name)
    lines.append("    Do not edit by hand -- edit the .hfd and regenerate.")
    lines.append("")
    lines.append("    Frame: X forward, Y up, Z to the subject's right (OpenSim's, not")
    lines.append("    MuJoCo's Z-up). Gravity is -Y and the floor normal is +Y, so every")
    lines.append("    offset, axis and sign is copied from the .hfd unchanged.")
    lines.append("")
    lines.append("    Body frames sit at segment COMs, matching the .hfd's convention")
    lines.append("    for joint offsets and contact geometry.")
    lines.append("  -->")
    lines.append('  <compiler angle="radian" coordinate="local" inertiafromgeom="false"/>')
    lines.append(
        '  <option timestep="0.001" gravity="0 -9.81 0" integrator="implicitfast"/>'
    )
    lines.append("")
    lines.append("  <default>")
    lines.append('    <geom rgba="0.72 0.74 0.8 1"/>')
    lines.append('    <default class="visual">')
    lines.append('      <geom contype="0" conaffinity="0" group="1" density="0"/>')
    lines.append("    </default>")
    lines.append('    <default class="contact">')
    # contype 1 / conaffinity 2 against a floor of contype 2 / conaffinity 1:
    # the six contact spheres touch the floor and nothing else, so feet and
    # crutch tips can never collide with each other or with the body.
    lines.append(
        '      <geom contype="1" conaffinity="2" group="3" friction="%s 0.005 0.0001"'
        ' solref="0.01 1" rgba="0.9 0.5 0.2 0.6"/>' % fmt(friction)
    )
    lines.append("    </default>")
    lines.append("  </default>")
    lines.append("")
    lines.append("  <visual>")
    lines.append('    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35"/>')
    # offwidth/offheight size the offscreen framebuffer; MuJoCo's default
    # 640x480 makes mujoco.Renderer refuse anything larger.
    lines.append(
        '    <global azimuth="90" elevation="-10" offwidth="1920" offheight="1080"/>'
    )
    lines.append("  </visual>")
    lines.append("")
    lines.append("  <asset>")
    lines.append(
        '    <texture name="grid" type="2d" builtin="checker" width="512" height="512"'
        ' rgb1="0.24 0.26 0.3" rgb2="0.3 0.32 0.36"/>'
    )
    lines.append(
        '    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>'
    )
    lines.append("  </asset>")
    lines.append("")
    lines.append("  <worldbody>")
    lines.append('    <light pos="0 3 0" dir="0 -1 0" directional="true"/>')
    # A true sagittal view, tracking the COM. Use this camera rather than the
    # free one: MuJoCo's free-camera azimuth/elevation assume a Z-up world, so
    # orbiting a Y-up model with it feels rotated. `pos` is an offset from the
    # tracked COM, so this sits 3 m to the subject's right looking along -Z.
    lines.append(
        '    <camera name="side" pos="0 0 3" xyaxes="1 0 0 0 1 0" mode="trackcom"/>'
    )
    lines.append(
        '    <geom name="floor" type="plane" size="50 50 0.1" zaxis="0 1 0" pos="0 0 0"'
        ' contype="2" conaffinity="1" friction="%s 0.005 0.0001" solref="0.01 1"'
        ' material="grid"/>' % fmt(friction)
    )
    lines.extend(body_lines)
    lines.append("  </worldbody>")
    lines.append("")
    lines.append("  <actuator>")
    for name in ACTUATOR_ORDER:
        dof = model.dof_by_name[name]
        motor = model.motor_for(dof.joint_name())
        max_torque = float(one(motor, "max_torque", 1000.0)) if motor else 1000.0
        lines.append(
            '    <motor name="%s" joint="%s" gear="1" ctrllimited="true"'
            ' ctrlrange="%s"/>' % (name, name, fmt(-max_torque, max_torque))
        )
    lines.append("  </actuator>")
    lines.append("")
    lines.append("  <keyframe>")
    lines.append("    <!-- the .hfd's dof defaults, in MJCF joint order -->")
    lines.append('    <key name="neutral" qpos="%s"/>' % fmt(*neutral))
    lines.append("  </keyframe>")
    lines.append("")
    lines.append("  <custom>")
    lines.append("    <!--")
    lines.append("      The canonical 16-entry dof table, in the .hfd's declaration order.")
    lines.append("      env.py indexes dofs by name against this; four of them are locked")
    lines.append("      and have no MJCF joint, so the backend reads them as constant 0.")
    lines.append("      Carried in the file rather than hard-coded in Python so the two")
    lines.append("      cannot drift apart.")
    lines.append("    -->")
    lines.append(
        '    <text name="dof_order" data="%s"/>' % " ".join(d.name for d in model.dofs)
    )
    lines.append(
        '    <text name="locked_dofs" data="%s"/>'
        % " ".join(d.name for d in model.dofs if model.is_locked_dof(d))
    )
    lines.append("  </custom>")
    lines.append("</mujoco>")
    return "\n".join(lines) + "\n", notes, joint_order


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Translate the .hfd model into MJCF.")
    ap.add_argument("--hfd", type=Path, default=DEFAULT_HFD)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--limit-damping",
        type=float,
        default=DEFAULT_LIMIT_DAMPING,
        help="damping for the soft joint limits, N.m.s/rad (default %(default)s)",
    )
    args = ap.parse_args(argv)

    model = HfdModel(parse_hfd(args.hfd))
    xml, notes, joint_order = build_mjcf(model, args.hfd, args.limit_damping)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(xml, encoding="utf-8")

    locked = [d.name for d in model.dofs if model.is_locked_dof(d)]
    print("read  %s" % args.hfd)
    print("wrote %s" % args.out)
    print("  bodies    %d" % len(model.bodies))
    print(
        "  dofs      %d  (%d as MJCF joints, %d locked into rigid welds)"
        % (len(model.dofs), len(joint_order), len(locked))
    )
    print("  actuators %d" % len(ACTUATOR_ORDER))
    print(
        "  mass      %.3f kg  (weight %.1f N)"
        % (model.total_mass(), model.total_mass() * 9.81)
    )
    print("  contacts  %s" % ", ".join(g.name for g in model.geoms if g.type == "sphere"))
    if notes:
        print("  departures from the .hfd:")
        for note in notes:
            print("    %s" % note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
