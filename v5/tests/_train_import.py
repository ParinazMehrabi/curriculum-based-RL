"""Import the trainer script as a module so its internals can be tested."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_ppo.py"
_NAME = "_v5_train_ppo"

if _NAME in sys.modules:
    _module = sys.modules[_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_NAME, _PATH)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    _spec.loader.exec_module(_module)

BestCheckpoints = _module.BestCheckpoints
RunningStd = _module.RunningStd
SyncVecEnv = _module.SyncVecEnv
compute_gae = _module.compute_gae
save_checkpoint = _module.save_checkpoint
parse_args = _module.parse_args
