"""Auto-named renders identify their source and never overwrite each other."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

_spec = importlib.util.spec_from_file_location(
    "_v5_eval_checkpoint", V5 / "scripts" / "eval_checkpoint.py"
)
_eval = importlib.util.module_from_spec(_spec)
sys.modules["_v5_eval_checkpoint"] = _eval
_spec.loader.exec_module(_eval)


def test_name_carries_run_iteration_and_return():
    ckpt = Path("runs/W-p4-260930.094250/ckpt_best_it000320_ret+00032.36.pt")
    assert (_eval.render_name(ckpt, 320, 32.36)
            == "W-p4-260930.094250_it000320_ret+00032.36")


def test_name_falls_back_to_the_file_when_the_iteration_is_unknown():
    ckpt = Path("runs/W-p4-260930.094250/ckpt_latest.pt")
    assert _eval.render_name(ckpt, None, 1.5) == (
        "W-p4-260930.094250_ckpt_latest_ret+00001.50"
    )


def test_repeated_renders_get_distinct_names(tmp_path):
    made = []
    for _ in range(3):
        path = _eval.unique_path(tmp_path / "render", "best", ".mp4")
        path.write_bytes(b"")       # as render_episode does, before the next call
        made.append(path)
    assert [p.name for p in made] == ["best.mp4", "best_2.mp4", "best_3.mp4"]


def test_the_render_directory_is_created_on_demand(tmp_path):
    target = tmp_path / "deep" / "render"
    assert not target.exists()
    _eval.unique_path(target, "best", ".mp4")
    assert target.is_dir()
