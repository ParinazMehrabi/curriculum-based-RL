"""Reward primitives, loaded from v4 rather than copied.

v4's `rewards.py` is deliberately free of any simulator, gym or Python-version
dependency -- it is pure numpy arithmetic over dicts of floats -- and it
encodes design work worth keeping: every term lives in [0, 1], terms compose as
a weighted geometric mean so no single term can be farmed in isolation, and
`RewardSpec.termination_report` proves the per-step reward cannot go negative.
The v4 README documents why each of those matters, and 175 tests pin them.

So v5 imports that file instead of duplicating it. Copying was the specific
failure v4 was written to undo -- four 550-line sibling files that had drifted
apart -- and a second copy of the reward maths would drift the same way.

The import is by path because the two packages cannot share an interpreter:
v4 runs on Python 3.9 with gym<0.22 (sconegym's constraint) while MyoSuite
needs 3.10+. Loading the module directly also skips v4's package __init__,
which imports gym. That is the same trick v4/tests/_bootstrap.py already uses
to test the reward maths on machines without a simulator.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

V4_REWARDS = (
    Path(__file__).resolve().parents[2]
    / "v4"
    / "sconegym_crutch_v4"
    / "rewards.py"
)

_MODULE_NAME = "_v5_rewards_from_v4"


def _load():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    if not V4_REWARDS.is_file():
        raise FileNotFoundError(
            "v5 reuses v4's reward primitives, but they are not at %s.\n"
            "Either restore that file or vendor a copy into v5." % V4_REWARDS
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, V4_REWARDS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_v4 = _load()

GEOMETRIC = _v4.GEOMETRIC
ADDITIVE = _v4.ADDITIVE
RewardSpec = _v4.RewardSpec
gaussian = _v4.gaussian
smoothstep = _v4.smoothstep
weighted_geometric_mean = _v4.weighted_geometric_mean
weighted_arithmetic_mean = _v4.weighted_arithmetic_mean

__all__ = [
    "ADDITIVE",
    "GEOMETRIC",
    "RewardSpec",
    "gaussian",
    "smoothstep",
    "weighted_arithmetic_mean",
    "weighted_geometric_mean",
]
