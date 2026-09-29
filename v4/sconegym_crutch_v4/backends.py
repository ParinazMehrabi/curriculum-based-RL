"""Which simulator `CrutchCurriculumGym` sits on top of.

The curriculum -- the stages, the reward terms, the RSI machinery -- is
simulator-agnostic; it reaches the physics through the small model API
described in `mujoco_backend`. Two implementations of that API exist:

* **scone** -- `sconegym.gaitgym.GaitGym` driving Hyfydy. The original, and the
  one the existing checkpoints were trained against. Needs sconegym, sconepy
  and an active Hyfydy licence.
* **mujoco** -- `mujoco_gym.MujocoGaitGym` driving the MJCF generated from the
  same .hfd by `v4/scripts/hfd_to_mjcf.py`. Needs only the `mujoco` wheel.

Choose with the `CRUTCH_V4_BACKEND` environment variable:

    CRUTCH_V4_BACKEND=mujoco   python -m deprl.main v4/configs/stage_A.yaml
    CRUTCH_V4_BACKEND=scone    python v4/scripts/validate_env.py --stage A
    CRUTCH_V4_BACKEND=auto     # the default: scone if importable, else mujoco

`auto` prefers scone so that an existing training machine keeps its behaviour
after this change, and falls back to mujoco elsewhere. It never silently
downgrades a machine that has sconegym installed but broken -- an import error
from a *present* sconegym propagates rather than being treated as absent.

**Checkpoints do not transfer between backends.** The observation layouts
differ in width and in meaning, so the actor's input layer will not load. Pick
one backend for a whole curriculum run.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

SCONE_MODEL = REPO_ROOT / "models" / "scone" / "Rajagopal_crutch_v3_A0_walk_003.scone"
MUJOCO_MODEL = REPO_ROOT / "models" / "mjcf" / "rajagopal_crutch_2d.xml"

ENV_VAR = "CRUTCH_V4_BACKEND"
VALID = ("auto", "scone", "mujoco")


def _load_scone():
    from sconegym.gaitgym import GaitGym

    return GaitGym, SCONE_MODEL


def _load_mujoco():
    from .mujoco_gym import MujocoGaitGym

    if not MUJOCO_MODEL.is_file():
        raise FileNotFoundError(
            "the MJCF model is missing: %s\nGenerate it with "
            "`python v4/scripts/hfd_to_mjcf.py`." % MUJOCO_MODEL
        )
    return MujocoGaitGym, MUJOCO_MODEL


def _sconegym_present() -> bool:
    """True if sconegym is installed, regardless of whether it imports cleanly.

    Checked with find_spec rather than a try/except around the import, so that
    a machine with a genuinely broken sconegym gets the real error instead of
    being quietly switched onto a different physics engine.
    """
    try:
        return importlib.util.find_spec("sconegym") is not None
    except (ImportError, ValueError):
        return False


def selected(name: str = None) -> str:
    """The backend name to use: 'scone' or 'mujoco', with 'auto' resolved."""
    requested = (name or os.environ.get(ENV_VAR) or "auto").strip().lower()
    if requested not in VALID:
        raise ValueError(
            "%s must be one of %s, got %r" % (ENV_VAR, ", ".join(VALID), requested)
        )
    if requested == "auto":
        requested = "scone" if _sconegym_present() else "mujoco"
    return requested


def ensure_simulator(name: str = None) -> str:
    """Import whatever the selected backend needs, and return its name.

    The scripts used to `import sconegym` unconditionally, which registers
    sconegym's own base environments. The v4 curriculum registers its own ids
    and does not need them, so that import is only required when the scone
    backend is actually in use -- and on a machine without a Hyfydy licence it
    is the import that fails first.
    """
    backend = selected(name)
    if backend == "scone":
        importlib.import_module("sconegym")  # registers sconegym's base envs
    return backend


def resolve(name: str = None) -> Tuple[type, Path, str]:
    """Return (base_class, default_model_file, backend_name)."""
    requested = selected(name)

    if requested == "scone":
        base, model = _load_scone()
    else:
        base, model = _load_mujoco()
    return base, model, requested
