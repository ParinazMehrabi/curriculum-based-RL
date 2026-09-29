"""A MuJoCo model object shaped like sconepy's.

`env.py` talks to the simulator through a small, stable surface -- roughly
twenty calls, all of them on `self.model` or on a body handle. This module
implements that surface on top of MuJoCo so the same environment class can run
under either simulator without the reward code knowing which.

The surface, for reference:

    model.dofs() / .actuators() / .muscles() / .mass() / .bodies()
    model.find_body(name)                 -> Body or None
    model.reset()
    model.set_dof_positions(q) / .set_dof_velocities(dq)
    model.init_state_from_dofs()
    model.adjust_state_for_load(fraction)
    model.set_actuator_inputs(torques)    # newton-metres
    model.advance_simulation_to(t)
    model.dof_position_array() / .dof_velocity_array()
    model.com_pos() / .com_vel()
    model.set_store_data(flag) / .write_results(dir, name)
    body.name() / .com_pos() / .contact_force()

Two things worth knowing before reading further.

**The dof table is not MuJoCo's.** env.py indexes dofs by name against a
16-entry table in the .hfd's declaration order, four of whose entries
(ankle_angle_r/l, mtp_angle_r/l) are locked. The MJCF has only 12 joints,
because the converter turns locked joints into rigid welds rather than making
the solver police zero-width ranges every step. So this module keeps the
canonical 16-name order -- read from the MJCF's own `<custom>` block, written
there by the converter, so the two cannot drift -- and maps it onto MuJoCo's
qpos. The locked four read as constant 0.0 and ignore writes of 0.0.

**Frames are OpenSim's, not MuJoCo's.** X forward, Y up, Z to the subject's
right; gravity is -Y. See v4/scripts/hfd_to_mjcf.py for why. Every vector this
module returns is in that frame, so `.y` is height and `.x` is forward, which
is what env.py's `_vec_x`/`_vec_y` assume.
"""
from __future__ import annotations

import math
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised only without mujoco
    raise ImportError(
        "the mujoco backend needs the `mujoco` package: pip install mujoco"
    ) from exc

GRAVITY = 9.81

# Vectors carry named components so callers can use either `v.y` or `v[1]`.
# env.py's _vec_x/_vec_y try the attribute first and fall back to indexing, so
# both paths work; sconepy's own vectors expose .x/.y/.z.
Vec3 = namedtuple("Vec3", "x y z")


def _vec(arr) -> Vec3:
    a = np.asarray(arr, dtype=np.float64).ravel()
    return Vec3(float(a[0]), float(a[1]), float(a[2]))


class Named:
    """A dof or actuator handle. sconepy exposes `.name()`; so do we."""

    __slots__ = ("_name", "index")

    def __init__(self, name: str, index: int):
        self._name = name
        self.index = index

    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return "<%s %s>" % (type(self).__name__, self._name)


class Dof(Named):
    __slots__ = ()


class Actuator(Named):
    __slots__ = ()


class Body:
    """A body handle exposing the two queries env.py makes of one."""

    __slots__ = ("_name", "id", "_model", "_geom_ids")

    def __init__(self, model: "MujocoModel", name: str, body_id: int):
        self._model = model
        self._name = name
        self.id = body_id
        self._geom_ids = [
            g
            for g in range(model.m.ngeom)
            if model.m.geom_bodyid[g] == body_id and model.m.geom_contype[g] != 0
        ]

    def name(self) -> str:
        return self._name

    def com_pos(self) -> Vec3:
        """World position of this body's centre of mass."""
        return _vec(self._model.d.xipos[self.id])

    def com_vel(self) -> Vec3:
        return _vec(self._model.subtree_linvel(self.id))

    def contact_force(self) -> Vec3:
        """Net contact force on this body, in world coordinates and newtons.

        Sums every active contact involving one of this body's colliding geoms.
        `.y` is the vertical component, which is what the crutch-load and
        ground-reaction terms read.
        """
        return _vec(self._model.body_contact_force(self._geom_ids))

    def __repr__(self) -> str:
        return "<Body %s>" % self._name


class MujocoModel:
    """sconepy-shaped wrapper around an MjModel/MjData pair."""

    def __init__(
        self,
        mjcf_path,
        init_state_path=None,
        substeps: Optional[int] = None,
    ):
        self.mjcf_path = Path(mjcf_path)
        if not self.mjcf_path.is_file():
            raise FileNotFoundError("MJCF not found: %s" % self.mjcf_path)
        self.m = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.d = mujoco.MjData(self.m)

        self._dof_names = self._read_custom_text("dof_order")
        self._locked = set(self._read_custom_text("locked_dofs"))
        if not self._dof_names:
            raise RuntimeError(
                "%s has no `dof_order` custom text. Regenerate it with "
                "v4/scripts/hfd_to_mjcf.py." % self.mjcf_path.name
            )

        # canonical dof index -> qpos/qvel address, or -1 when locked
        self._qpos_adr: List[int] = []
        self._qvel_adr: List[int] = []
        for name in self._dof_names:
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                if name not in self._locked:
                    raise RuntimeError(
                        "dof %r is in dof_order but is neither a MJCF joint nor "
                        "declared locked" % name
                    )
                self._qpos_adr.append(-1)
                self._qvel_adr.append(-1)
            else:
                self._qpos_adr.append(int(self.m.jnt_qposadr[jid]))
                self._qvel_adr.append(int(self.m.jnt_dofadr[jid]))

        self._dofs = [Dof(n, i) for i, n in enumerate(self._dof_names)]
        self._actuators = [
            Actuator(
                mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "", i
            )
            for i in range(self.m.nu)
        ]

        self._bodies: List[Body] = []
        self._body_by_name: Dict[str, Body] = {}
        for i in range(self.m.nbody):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name or name == "world":
                continue
            body = Body(self, name, i)
            self._bodies.append(body)
            self._body_by_name[name] = body

        self._mass = float(np.sum(self.m.body_mass))

        # The .hfd's dof defaults come through as the MJCF's "neutral"
        # keyframe; a SCONE .zml init state, when given, overrides them. That
        # mirrors the .scone file, which names both.
        self.init_q = self._keyframe_positions()
        self.init_dq = np.zeros(len(self._dof_names), dtype=np.float64)
        self.init_state_path = None
        if init_state_path is not None:
            self.load_init_state(init_state_path)

        # The MJCF integrates at its own timestep; advance_simulation_to steps
        # until it reaches the requested time.
        self.timestep = float(self.m.opt.timestep)
        if substeps is not None:
            self.timestep = float(substeps) * self.timestep

        self._store = False
        self._recorded: List[np.ndarray] = []
        self._subtree_vel_stale = True

        self.reset()

    # -- MJCF metadata ----------------------------------------------------

    def _read_custom_text(self, key: str) -> List[str]:
        tid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_TEXT, key)
        if tid < 0:
            return []
        start = int(self.m.text_adr[tid])
        size = int(self.m.text_size[tid])
        raw = bytes(self.m.text_data[start : start + size])
        return raw.decode("utf-8").strip("\x00").strip().split()

    def _keyframe_positions(self) -> np.ndarray:
        q = np.zeros(len(self._dof_names), dtype=np.float64)
        if self.m.nkey == 0:
            return q
        key_qpos = np.asarray(self.m.key_qpos[0], dtype=np.float64)
        for i, adr in enumerate(self._qpos_adr):
            if adr >= 0:
                q[i] = key_qpos[adr]
        return q

    def load_init_state(self, path) -> None:
        """Read a SCONE .zml init state into init_q / init_dq.

        The .zml is the same file the .scone model points at, so the MuJoCo and
        Hyfydy runs start from the same pose rather than from the .hfd defaults
        (which differ: the .zml drops the arms to arm_flex=0, elbow_flex=0.30).
        Unknown dof names raise rather than being skipped -- a typo there would
        silently change the starting pose.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("init state not found: %s" % path)
        values, velocities = _parse_zml_state(path)
        index = {n: i for i, n in enumerate(self._dof_names)}

        for label, table, target in (
            ("values", values, self.init_q),
            ("velocities", velocities, self.init_dq),
        ):
            unknown = sorted(set(table) - set(index))
            if unknown:
                raise KeyError(
                    "%s block of %s names dofs this model does not have: %s\n"
                    "model dofs: %s"
                    % (label, path.name, ", ".join(unknown), ", ".join(self._dof_names))
                )
            for name, value in table.items():
                target[index[name]] = value

        self.init_state_path = path

    # -- introspection ----------------------------------------------------

    def dofs(self) -> List[Dof]:
        return list(self._dofs)

    def actuators(self) -> List[Actuator]:
        return list(self._actuators)

    def muscles(self) -> List:
        """Always empty: this is a torque-actuated model."""
        return []

    def bodies(self) -> List[Body]:
        return list(self._bodies)

    def find_body(self, name: str) -> Optional[Body]:
        return self._body_by_name.get(name)

    def mass(self) -> float:
        return self._mass

    @property
    def body_weight(self) -> float:
        return self._mass * GRAVITY

    def dof_names(self) -> List[str]:
        return list(self._dof_names)

    def locked_dofs(self) -> List[str]:
        return [n for n in self._dof_names if n in self._locked]

    # -- state ------------------------------------------------------------

    def reset(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.set_dof_positions(self.init_q)
        self.set_dof_velocities(self.init_dq)
        self.d.ctrl[:] = 0.0
        self.d.time = 0.0
        self._recorded = []
        self.init_state_from_dofs()

    def set_dof_positions(self, q: Sequence[float]) -> None:
        q = np.asarray(q, dtype=np.float64).ravel()
        if q.size != len(self._dof_names):
            raise ValueError(
                "expected %d dof positions, got %d" % (len(self._dof_names), q.size)
            )
        for i, adr in enumerate(self._qpos_adr):
            if adr >= 0:
                self.d.qpos[adr] = q[i]
            elif abs(q[i]) > 1e-9:
                warnings.warn(
                    "ignoring a non-zero value (%g) written to locked dof %r"
                    % (q[i], self._dof_names[i]),
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._subtree_vel_stale = True

    def set_dof_velocities(self, dq: Sequence[float]) -> None:
        dq = np.asarray(dq, dtype=np.float64).ravel()
        if dq.size != len(self._dof_names):
            raise ValueError(
                "expected %d dof velocities, got %d" % (len(self._dof_names), dq.size)
            )
        for i, adr in enumerate(self._qvel_adr):
            if adr >= 0:
                self.d.qvel[adr] = dq[i]
        self._subtree_vel_stale = True

    def init_state_from_dofs(self) -> None:
        """Bring derived quantities into line with qpos/qvel.

        In Hyfydy this builds the full state from the dof values. MuJoCo's
        state *is* qpos/qvel, so there is nothing to build -- only kinematics
        and contacts to refresh.
        """
        mujoco.mj_forward(self.m, self.d)
        self._subtree_vel_stale = True

    def dof_position_array(self) -> np.ndarray:
        q = np.zeros(len(self._dof_names), dtype=np.float64)
        for i, adr in enumerate(self._qpos_adr):
            if adr >= 0:
                q[i] = self.d.qpos[adr]
        return q

    def dof_velocity_array(self) -> np.ndarray:
        dq = np.zeros(len(self._dof_names), dtype=np.float64)
        for i, adr in enumerate(self._qvel_adr):
            if adr >= 0:
                dq[i] = self.d.qvel[adr]
        return dq

    # -- settling ---------------------------------------------------------

    def total_vertical_contact_force(self) -> float:
        total = 0.0
        for i in range(self.d.ncon):
            total += float(self._contact_force_world(i)[1])
        return abs(total)

    # The smallest load MuJoCo will report at the instant of contact, measured
    # once per model and cached, as a fraction of body weight. See
    # adjust_state_for_load for why it is not zero.
    _onset_fraction: Optional[float] = None

    def adjust_state_for_load(self, fraction: float, tol: float = 1e-3) -> float:
        """Lower the model until ground contact carries `fraction` of body weight.

        Hyfydy's routine of the same name. The point is to begin an episode
        already bearing weight rather than dropping into contact a few steps
        in, which would put an impact transient at the start of every episode.

        Only the pelvis height moves -- joint angles are left exactly as the
        caller set them, so a reset pose or an RSI frame survives the
        adjustment untouched.

        **There is a floor on what this can achieve, and it is not zero.**
        MuJoCo resolves contact with a constraint solver rather than a penalty
        spring, so the normal force does not rise continuously from zero with
        penetration depth: the moment a contact becomes active it already
        carries a finite load.

        That floor depends on the pose, because it depends on how much of the
        body the first contact has to arrest. Measured on this model it ranges
        from about 0.20 to 0.73 of body weight across reset poses (0.40 from
        the neutral .zml pose). So the shipped `init_load=0.5` is reachable
        from some reset poses and not from others. It is barely sensitive to
        contact stiffness -- widening solref's time constant from 0.005 to 0.08
        moves the neutral-pose floor only from 0.445 to 0.362, while taking
        penetration from 0.04 mm to 21 mm -- so it cannot be tuned away, and
        the stiff setting is kept for the shallow penetration.

        When the target is below the floor the model is settled at the
        shallowest real contact rather than left hovering, which is the useful
        behaviour and keeps the impact transient out of the episode either way.
        A warning is issued once per model when that happens.

        Hyfydy's compliant contact has no such floor, so this is a genuine
        sim-to-sim difference in how `init_load` behaves.

        Returns the achieved load as a fraction of body weight.
        """
        target = float(fraction) * self.body_weight
        if target <= 0.0:
            return 0.0
        adr = self._qpos_adr[self._dof_names.index("pelvis_ty")]
        y0 = float(self.d.qpos[adr])

        def load_at(y: float) -> float:
            self.d.qpos[adr] = y
            mujoco.mj_forward(self.m, self.d)
            return self.total_vertical_contact_force()

        # 1. Bracket contact: `clear` is above the floor, `touching` below it.
        clear, touching = y0 + 0.30, y0
        for _ in range(80):
            if load_at(touching) > 0.0:
                break
            touching -= 0.005
        else:
            self.init_state_from_dofs()
            return 0.0
        if load_at(clear) > 0.0:
            clear = touching + 0.30

        # 2. Find the onset height -- the shallowest height that is in contact.
        for _ in range(60):
            mid = 0.5 * (clear + touching)
            if load_at(mid) > 0.0:
                touching = mid
            else:
                clear = mid
            if clear - touching < 1e-9:
                break
        onset_load = load_at(touching)
        self._onset_fraction = onset_load / self.body_weight

        # 3. Either the target is below the onset load, or bisect down to it.
        if onset_load >= target:
            if onset_load - target > 0.02 * self.body_weight:
                self._warn_once_load_floor(fraction, self._onset_fraction)
            best_y = touching
        else:
            lo, hi = touching, touching
            for _ in range(80):
                hi -= 0.002
                if load_at(hi) >= target:
                    break
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                load = load_at(mid)
                if abs(load - target) <= tol * self.body_weight:
                    break
                if load < target:
                    lo = mid
                else:
                    hi = mid
            best_y = mid

        achieved = load_at(best_y)
        self.init_state_from_dofs()
        return achieved / self.body_weight

    def _warn_once_load_floor(self, requested: float, floor: float) -> None:
        if getattr(self, "_load_floor_warned", False):
            return
        self._load_floor_warned = True
        warnings.warn(
            "init_load=%.3f is below what MuJoCo's contact solver can produce "
            "on this model (about %.3f of body weight at the instant of "
            "contact). The model was settled at the shallowest real contact "
            "instead of left hovering. See "
            "MujocoModel.adjust_state_for_load." % (requested, floor),
            RuntimeWarning,
            stacklevel=3,
        )

    # -- stepping ---------------------------------------------------------

    def set_actuator_inputs(self, torques: Sequence[float]) -> None:
        """Set joint torques, in newton-metres.

        env.py passes `normalised_action * torque_scales`, so these are real
        torques rather than normalised inputs. The MJCF's motors use gear=1,
        which makes ctrl newton-metres directly, and carry the .hfd's
        max_torque as their ctrlrange.
        """
        t = np.asarray(torques, dtype=np.float64).ravel()
        if t.size != self.m.nu:
            raise ValueError("expected %d actuator inputs, got %d" % (self.m.nu, t.size))
        self.d.ctrl[:] = t

    def advance_simulation_to(self, t: float) -> None:
        target = float(t)
        guard = 0
        limit = int(abs(target - self.d.time) / self.m.opt.timestep) + 16
        while self.d.time < target - 1e-12:
            mujoco.mj_step(self.m, self.d)
            if self._store:
                self._recorded.append(self._record_row())
            guard += 1
            if guard > limit:
                raise RuntimeError(
                    "advance_simulation_to(%g) did not converge from t=%g"
                    % (target, self.d.time)
                )
        self._subtree_vel_stale = True

    @property
    def time(self) -> float:
        return float(self.d.time)

    # -- measurement ------------------------------------------------------

    def com_pos(self) -> Vec3:
        return _vec(self.d.subtree_com[0])

    def com_vel(self) -> Vec3:
        return _vec(self.subtree_linvel(0))

    def subtree_linvel(self, body_id: int) -> np.ndarray:
        if self._subtree_vel_stale:
            mujoco.mj_subtreeVel(self.m, self.d)
            self._subtree_vel_stale = False
        return self.d.subtree_linvel[body_id]

    def _contact_force_world(self, i: int) -> np.ndarray:
        """World-frame force that contact `i` applies to geom2's body.

        mj_contactForce reports in the contact frame, whose rows are the normal
        and two tangents, with the normal pointing from geom1 to geom2. The
        transpose of that frame therefore maps contact-local to world, and the
        result acts on geom2 (geom1 takes the negative).
        """
        buf = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.m, self.d, i, buf)
        frame = np.asarray(self.d.contact[i].frame, dtype=np.float64).reshape(3, 3)
        return frame.T @ buf[:3]

    def body_contact_force(self, geom_ids: Sequence[int]) -> np.ndarray:
        if not len(geom_ids):
            return np.zeros(3)
        wanted = set(int(g) for g in geom_ids)
        total = np.zeros(3, dtype=np.float64)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 not in wanted and g2 not in wanted:
                continue
            force = self._contact_force_world(i)
            # The force acts on geom2's body; geom1's body takes the reaction.
            total += force if g2 in wanted else -force
        return total

    # -- result storage ---------------------------------------------------

    def set_store_data(self, flag: bool) -> None:
        self._store = bool(flag)
        if not self._store:
            self._recorded = []

    def _record_row(self) -> np.ndarray:
        return np.concatenate(
            [[self.d.time], self.dof_position_array(), self.dof_velocity_array()]
        )

    def write_results(self, output_dir, name: str) -> Optional[Path]:
        """Write the recorded rollout as an OpenSim .sto file.

        Same format the reference trajectory uses, so anything that reads those
        (trajectory.py, OpenSim, SCONE Studio) reads these too.
        """
        if not self._recorded:
            return None
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / ("%s.sto" % name)
        rows = np.asarray(self._recorded, dtype=np.float64)
        columns = (
            ["time"]
            + list(self._dof_names)
            + ["%s_u" % n for n in self._dof_names]
        )
        with path.open("w", encoding="utf-8") as fh:
            fh.write("%s\n" % name)
            fh.write("version=1\n")
            fh.write("nRows=%d\n" % rows.shape[0])
            fh.write("nColumns=%d\n" % rows.shape[1])
            fh.write("inDegrees=no\n")
            fh.write("endheader\n")
            fh.write("\t".join(columns) + "\n")
            for row in rows:
                fh.write("\t".join("%.8g" % v for v in row) + "\n")
        self._recorded = []
        return path


def _parse_zml_state(path: Path):
    """Read the `values` and `velocities` blocks of a SCONE .zml init state."""
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())

    def block(name: str) -> Dict[str, float]:
        start = text.find(name)
        if start < 0:
            return {}
        open_brace = text.find("{", start)
        depth, i = 0, open_brace
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out: Dict[str, float] = {}
        for line in text[open_brace + 1 : i].splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value:
                out[key] = float(value)
        return out

    return block("values"), block("velocities")
