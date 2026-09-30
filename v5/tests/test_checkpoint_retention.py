"""BestCheckpoints keeps the ten best and deletes nothing else.

The deletion rule is tested harder than the retention rule on purpose: this
project has lost two real training runs to careless cleanup, so what the class
refuses to delete matters more than what it keeps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from _train_import import BestCheckpoints  # noqa: E402


def payload(i: int) -> dict:
    return {"net": {"w": torch.zeros(2)}, "iteration": i}


def best_files(run: Path):
    return sorted(p.name for p in run.glob("ckpt_best_*.pt"))


def test_keeps_only_the_best_ten(tmp_path):
    keeper = BestCheckpoints(tmp_path, keep=10)
    for i in range(1, 26):
        keeper.offer(float(i), i, payload(i))

    files = best_files(tmp_path)
    assert len(files) == 10
    scores = sorted(s for s, _ in keeper.entries)
    assert scores == [float(i) for i in range(16, 26)]
    # every kept entry is a file that exists, and nothing else is left behind
    assert {p.name for _, p in keeper.entries} == set(files)


def test_a_worse_score_is_not_saved_once_full(tmp_path):
    keeper = BestCheckpoints(tmp_path, keep=3)
    for i, score in enumerate([5.0, 6.0, 7.0], start=1):
        keeper.offer(score, i, payload(i))
    before = best_files(tmp_path)

    assert keeper.offer(1.0, 99, payload(99)) is None
    assert keeper.offer(5.0, 98, payload(98)) is None   # ties do not displace
    assert best_files(tmp_path) == before


def test_a_better_score_displaces_the_worst(tmp_path):
    keeper = BestCheckpoints(tmp_path, keep=3)
    for i, score in enumerate([5.0, 6.0, 7.0], start=1):
        keeper.offer(score, i, payload(i))
    worst = keeper.entries[0][1]

    kept = keeper.offer(9.0, 4, payload(4))
    assert kept is not None and kept.exists()
    assert not worst.exists()
    assert sorted(s for s, _ in keeper.entries) == [6.0, 7.0, 9.0]


def test_nan_never_displaces_a_real_score(tmp_path):
    keeper = BestCheckpoints(tmp_path, keep=2)
    keeper.offer(1.0, 1, payload(1))
    assert keeper.offer(float("nan"), 2, payload(2)) is None
    assert len(best_files(tmp_path)) == 1


def test_latest_and_foreign_files_are_untouched(tmp_path):
    latest = tmp_path / "ckpt_latest.pt"
    other = tmp_path / "log.csv"
    torch.save(payload(0), latest)
    other.write_text("iteration\n", encoding="utf-8")

    keeper = BestCheckpoints(tmp_path, keep=2)
    for i in range(1, 12):
        keeper.offer(float(i), i, payload(i))

    assert latest.exists(), "ckpt_latest.pt is what resuming needs"
    assert other.exists()
    assert len(best_files(tmp_path)) == 2


def test_it_refuses_to_delete_outside_its_own_run(tmp_path):
    """A path that is not this run's own best checkpoint is never unlinked."""
    mine = tmp_path / "mine"
    theirs = tmp_path / "theirs"
    mine.mkdir()
    theirs.mkdir()
    victim = theirs / "ckpt_best_it000001_ret+0001.00.pt"
    torch.save(payload(1), victim)

    keeper = BestCheckpoints(mine, keep=1)
    # smuggle another run's file into the list, as a bug elsewhere might
    keeper.entries.append((-1.0, victim))
    keeper.entries.append((-2.0, mine / "ckpt_latest.pt"))
    keeper.entries.sort(key=lambda e: e[0])
    keeper.offer(100.0, 2, payload(2))

    assert victim.exists(), "another run's checkpoint must survive"


def test_filename_sorts_and_carries_the_score(tmp_path):
    keeper = BestCheckpoints(tmp_path, keep=10)
    a = keeper.filename(12.5, 20)
    b = keeper.filename(-3.25, 300)
    assert a == "ckpt_best_it000020_ret+00012.50.pt"
    assert b == "ckpt_best_it000300_ret-00003.25.pt"
    assert keeper.filename(1.0, 20) < keeper.filename(1.0, 300)
