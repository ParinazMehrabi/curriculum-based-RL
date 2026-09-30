"""Where throwaway training runs go, so real ones are never in the way.

Smoke-testing the trainer used to write into `v5/runs/` alongside real runs,
and cleaning those up afterwards with `rm -rf v5/runs/W-p4-*` destroyed two
real runs -- the glob matches every stage-W four-phase run, not just the
throwaway. `v5/runs/` is gitignored, so there was never a reason to delete
anything from it.

Throwaway runs now go to the system temp directory instead. Nothing under
`v5/runs/` should ever be deleted by tooling.

    python scripts/train_ppo.py --out $(python scripts/_scratch_run.py) ...
"""
from __future__ import annotations

import tempfile
from pathlib import Path


def scratch_runs_dir() -> Path:
    """A temp directory for runs that are not worth keeping."""
    path = Path(tempfile.gettempdir()) / "v5-scratch-runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    print(scratch_runs_dir())
