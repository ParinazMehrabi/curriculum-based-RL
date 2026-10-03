"""A SCONE `.zml` initial state, mapped onto this model.

Loads `models/init/InitStateH0918Gait10ActA.zml` -- a gait10dof18musc state
captured mid-stride -- and applies its joint angles, joint velocities and
muscle activations.

**Why a captured state rather than a standing pose.** The model is exactly
left/right symmetric, and so is a policy that sees a symmetric observation. A
standing start puts both legs at identical angles with identical velocities and
identical activations, so the two sides of the network receive mirror-identical
input and emit mirror-identical output: the model can hop, and it cannot step.
Nothing in the reward breaks that tie, and reset noise only breaks it by
accident, slowly, and differently every episode. A captured state breaks it on
purpose and reproducibly, in all three places at once:

* **Pose.** Right hip flexed 0.44 rad with the knee near extension; left knee
  flexed 1.04 rad with the ankle plantarflexed 0.35 rad.
* **Velocity.** Right hip extending at -1.36 rad/s, left hip flexing at
  +3.34 rad/s. The pelvis is already travelling at 1.08 m/s.
* **Activation.** Right vasti at 0.48 against left vasti at 0.01; left
  iliopsoas at 0.29 against right at 0.05.

Read together that is right-leg stance just after contact and left leg at
toe-off, which is a phase of walking rather than a posture.

Three parts of the mapping are not identities, and each is the same correction
`reference.py` documents for the same reason -- these are OpenSim conventions,
and this model uses Rajagopal's.

* **pelvis_tilt is negated.** Negative is a forward lean in OpenSim; this
  model's hinge is positive-forward.
* **knee_angle is negated.** The file's knees are negative throughout
  (-0.39 and -1.04); MyoSuite's `knee_angle` is positive-is-flexion over
  [0, 2.0944]. Both values land inside that range once negated.
* **pelvis_ty is an absolute height, not a joint value.** Here `pelvis_ty` is
  a slide offset from the root body's base position, so the height is applied
  by solving for the offset that produces it.

`hip_flexion` and `ankle_angle` need no correction: positive is flexion and
dorsiflexion respectively in both conventions.

**Muscles are groups, not actuators.** gait10dof18musc has nine lumped muscles
per leg; this model has 290. Each group's activation is applied to every
MyoSuite actuator belonging to it, which is the honest reading of a lumped
value -- `vasti = 0.484` means the whole quadriceps group is at 0.484, so
vasint, vaslat and vasmed each start there. The 256 actuators the file says
nothing about keep the stage's `initial_activation`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INIT_STATE = REPO_ROOT / "models" / "init" / "InitStateH0918Gait10ActA.zml"

# file dof -> (model joint, sign). See the module docstring for the three
# that are not identities.
JOINT_MAP: Dict[str, Tuple[str, float]] = {
    "pelvis_tilt": ("pelvis_tilt", -1.0),
    "pelvis_tx": ("pelvis_tx", +1.0),
    "hip_flexion_r": ("hip_flexion_r", +1.0),
    "knee_angle_r": ("knee_angle_r", -1.0),
    "ankle_angle_r": ("ankle_angle_r", +1.0),
    "hip_flexion_l": ("hip_flexion_l", +1.0),
    "knee_angle_l": ("knee_angle_l", -1.0),
    "ankle_angle_l": ("ankle_angle_l", +1.0),
}

# Handled apart from JOINT_MAP: an absolute height, not a joint coordinate.
HEIGHT_DOF = "pelvis_ty"

# gait10dof18musc's nine lumped muscles -> the MyoSuite actuators that make
# them up. Verified against the compiled model: all 34 exist.
MUSCLE_MAP: Dict[str, Tuple[str, ...]] = {
    "hamstrings": ("bflh", "semimem", "semiten"),
    "bifemsh": ("bfsh",),
    "glut_max": ("glmax1", "glmax2", "glmax3"),
    "iliopsoas": ("iliacus", "psoas"),
    "rect_fem": ("recfem",),
    "vasti": ("vasint", "vaslat", "vasmed"),
    "gastroc": ("gasmed", "gaslat"),
    "soleus": ("soleus",),
    "tib_ant": ("tibant",),
}

_BLOCK = re.compile(r"(\w+)\s*\{(.*?)\}", re.S)
_ENTRY = re.compile(r"(\w+)\s*=\s*(-?[\d.eE+-]+)")


@dataclass(frozen=True)
class InitState:
    """One captured state: positions, velocities and activations, as read."""

    path: Path
    values: Dict[str, float] = field(default_factory=dict)
    velocities: Dict[str, float] = field(default_factory=dict)
    activations: Dict[str, float] = field(default_factory=dict)

    # -- the model's own convention ---------------------------------------

    def joint_positions(self) -> Dict[str, float]:
        """Sign-corrected joint angles, keyed by this model's joint names."""
        return {
            joint: sign * self.values[dof]
            for dof, (joint, sign) in JOINT_MAP.items()
            if dof in self.values
        }

    def joint_velocities(self) -> Dict[str, float]:
        """Sign-corrected joint velocities. The signs are the positions'.

        A velocity is the time derivative of a coordinate, so a flipped
        coordinate flips its velocity too. Getting this wrong and only the
        angles right would start the model in the right pose moving the wrong
        way, which is harder to notice than a wrong pose.
        """
        out = {
            joint: sign * self.velocities[dof]
            for dof, (joint, sign) in JOINT_MAP.items()
            if dof in self.velocities
        }
        if HEIGHT_DOF in self.velocities:
            # Vertical velocity needs no sign change: positive is up in both.
            out[HEIGHT_DOF] = self.velocities[HEIGHT_DOF]
        return out

    @property
    def pelvis_height(self):
        """The absolute pelvis height, or None if the file omits it."""
        return self.values.get(HEIGHT_DOF)

    @property
    def forward_velocity(self) -> float:
        return float(self.velocities.get("pelvis_tx", 0.0))

    def muscle_activations(self) -> Dict[str, float]:
        """Per-actuator activations, expanding each lumped group."""
        out: Dict[str, float] = {}
        for name, value in self.activations.items():
            group, _, side = name.rpartition("_")
            members = MUSCLE_MAP.get(group)
            if members is None or side not in ("r", "l"):
                continue
            for stem in members:
                out["%s_%s" % (stem, side)] = float(value)
        return out

    # -- reporting ---------------------------------------------------------

    def asymmetry(self) -> Dict[str, float]:
        """Left-right difference in each paired quantity.

        The reason this file is used at all, so it is worth being able to
        assert on: a state whose asymmetry is zero would reintroduce the
        problem it was brought in to solve.
        """
        out = {}
        for source in (self.values, self.velocities, self.activations):
            for key in source:
                if not key.endswith("_r"):
                    continue
                mirror = key[:-2] + "_l"
                if mirror in source:
                    out[key[:-2]] = abs(source[key] - source[mirror])
        return out

    def describe(self) -> str:
        worst = sorted(self.asymmetry().items(), key=lambda kv: -kv[1])[:3]
        return "%s | %d dofs, %d velocities, %d muscles | v %.2f m/s | %s" % (
            self.path.name, len(self.values), len(self.velocities),
            len(self.activations), self.forward_velocity,
            ", ".join("%s %+.2f" % (k, v) for k, v in worst),
        )


def load_init_state(path=None) -> InitState:
    """Parse a SCONE `.zml` state file.

    The format is three `name { key = value }` blocks. Parsed with a regex
    rather than a .zml library because that is the whole of the grammar in
    use, and a dependency for it would have to be installed on a machine that
    already fights every wheel it is given.
    """
    path = Path(path) if path else DEFAULT_INIT_STATE
    if not path.is_file():
        raise FileNotFoundError("initial state not found: %s" % path)

    blocks = {
        name: {k: float(v) for k, v in _ENTRY.findall(body)}
        for name, body in _BLOCK.findall(path.read_text(encoding="utf-8"))
    }
    missing = [b for b in ("values", "velocities", "activations") if b not in blocks]
    if missing:
        raise ValueError("%s has no %s block" % (path, ", ".join(missing)))

    state = InitState(
        path=path,
        values=blocks["values"],
        velocities=blocks["velocities"],
        activations=blocks["activations"],
    )
    if not state.asymmetry():
        raise ValueError("%s has no paired quantities to compare" % path)
    return state


def resolve_init_state(name) -> Path:
    """A stage's `init_state` value as a path.

    A bare filename means `models/init/<name>`, which is where the captured
    states live, so a stage can name one without embedding a path. Anything
    with a separator, or an absolute path, is used as given.
    """
    path = Path(name)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return DEFAULT_INIT_STATE.parent / path.name
