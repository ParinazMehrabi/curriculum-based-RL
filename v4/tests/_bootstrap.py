"""Import rewards/stages without executing the package __init__.

The package __init__ imports gym in order to register the environments. The
reward maths deliberately has no simulator dependency, so the tests load the
submodules under a synthetic package name and skip __init__ entirely. That way
the reward invariants are testable on any machine, not only one with sconegym
and Hyfydy installed.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

PKG_NAME = "_scv4_under_test"
PKG_DIR = Path(__file__).resolve().parents[1] / "sconegym_crutch_v4"


def load():
    """Return (rewards, stages) module objects."""
    if PKG_NAME not in sys.modules:
        pkg = types.ModuleType(PKG_NAME)
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules[PKG_NAME] = pkg

    import importlib

    rewards = importlib.import_module(PKG_NAME + ".rewards")
    stages = importlib.import_module(PKG_NAME + ".stages")
    return rewards, stages


def env_source() -> str:
    return (PKG_DIR / "env.py").read_text(encoding="utf-8")
