from __future__ import annotations

import importlib.util
import os
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sts1"
    / "sts1_armg_strategy_loop.py"
)
SPEC = importlib.util.spec_from_file_location("strategy_loop_v4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

PARENT_PID = os.getpid()


def _fake_branch(gc, **kwargs):
    return {"value": int(gc), "current_armg_index": 0}


def _child_only_failure(gc, **kwargs):
    if os.getpid() != PARENT_PID:
        raise RuntimeError("intentional child failure")
    return {"value": int(gc), "current_armg_index": 0}


def _rows():
    return [
        {"gc": 11, "current_armg_index": 0},
        {"gc": 22, "current_armg_index": 0},
        {"gc": 33, "current_armg_index": 0},
        {"gc": 44, "current_armg_index": 0},
    ]


def test_parallel_teacher_uses_two_workers_and_preserves_order(monkeypatch) -> None:
    if "fork" not in m.mp.get_all_start_methods():
        return
    monkeypatch.setattr(m, "_branch_example", _fake_branch)
    results, report = m._label_selected_candidates(
        _rows(),
        sts=object(),
        armg=object(),
        mcts_sims=2000,
        max_game_steps=600,
        max_battle_steps=1200,
        temperature=2.0,
        collection_workers=2,
        parallel_timeout_seconds=30,
    )
    assert [row["value"] for row in results] == [11, 22, 33, 44]
    assert report["mode"] == "fork_pool"
    assert report["effective_workers"] == 2
    assert report["fallback_reason"] is None


def test_parallel_teacher_fails_safe_to_sequential(monkeypatch) -> None:
    if "fork" not in m.mp.get_all_start_methods():
        return
    monkeypatch.setattr(m, "_branch_example", _child_only_failure)
    results, report = m._label_selected_candidates(
        _rows()[:2],
        sts=object(),
        armg=object(),
        mcts_sims=2000,
        max_game_steps=600,
        max_battle_steps=1200,
        temperature=2.0,
        collection_workers=2,
        parallel_timeout_seconds=30,
    )
    assert [row["value"] for row in results] == [11, 22]
    assert report["mode"] == "sequential_fallback"
    assert report["effective_workers"] == 1
    assert "RuntimeError:intentional child failure" in report["fallback_reason"]


def test_collection_workers_one_never_forks(monkeypatch) -> None:
    monkeypatch.setattr(m, "_branch_example", _fake_branch)
    results, report = m._label_selected_candidates(
        _rows()[:2],
        sts=object(),
        armg=object(),
        mcts_sims=2000,
        max_game_steps=600,
        max_battle_steps=1200,
        temperature=2.0,
        collection_workers=1,
        parallel_timeout_seconds=30,
    )
    assert [row["value"] for row in results] == [11, 22]
    assert report["mode"] == "sequential"
    assert report["effective_workers"] == 1
