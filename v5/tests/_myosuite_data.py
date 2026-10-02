"""Skip a test when MyoSuite's model file is absent, without importing it.

`pytest.importorskip("myosuite")` is the obvious way and the wrong one here.
This project reads MyoSuite's data -- one XML and the meshes beside it -- and
deliberately never imports the package: its `__init__` registers several
hundred Gym environments nothing here uses, and it pins MuJoCo tightly enough
that importing it would make its compatibility this env's problem. On a machine
whose MuJoCo must predate any MyoSuite release, it is installed with
`--no-deps` and importing it fails on a missing dependency while every file
needed is present.

So the tests check for the file, exactly as `env._default_model_path` does.
"""
from __future__ import annotations

import pytest

MESSAGE = "MyoSuite's model file is not installed"


def require_model() -> None:
    from myo_curriculum.env import _default_model_path

    try:
        path = _default_model_path()
    except FileNotFoundError as exc:
        pytest.skip("%s (%s)" % (MESSAGE, exc))
    if not path.is_file():
        pytest.skip("%s: %s" % (MESSAGE, path))
