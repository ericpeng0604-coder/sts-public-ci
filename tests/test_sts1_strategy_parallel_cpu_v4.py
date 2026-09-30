from __future__ import annotations

import importlib.util
import os
import sys
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
sys.modules[SPEC.name] = m
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
    assert report["mode"] == "fork_rolling"
    assert report["effective_workers"] == 2
    assert report["fallback_reason"] is None
    assert report["parallel_elapsed_seconds"] >= 0.0
    assert report["parity_elapsed_seconds"] >= 0.0


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

# Workflow trigger: validate fixed fresh-child v4 at full 2-worker load.

# Trigger real 3-worker benchmark on latest control.

# Trigger final benchmark-best control validation.

# Full-core validation trigger.


def test_zero_max_stagnation_disables_auto_pause() -> None:
    assert m._stagnation_limit_reached(count=12, max_stagnation=0) is False
    assert m._stagnation_limit_reached(count=999, max_stagnation=0) is False


def test_positive_max_stagnation_still_pauses() -> None:
    assert m._stagnation_limit_reached(count=11, max_stagnation=12) is False
    assert m._stagnation_limit_reached(count=12, max_stagnation=12) is True


def test_stagnation_rescue_uses_smaller_candidate_steps() -> None:
    normal = m._candidate_recipes_for_stagnation(11)
    rescue = m._candidate_recipes_for_stagnation(12)

    assert [row["step_scale"] for row in normal] == [0.25, 0.50, 1.00]
    assert [row["step_scale"] for row in rescue] == [0.10, 0.20, 0.35]
    assert all(row["name"].startswith("rescue_") for row in rescue)


def test_rescue_focus_weight_boosts_only_current_teacher_rows(tmp_path) -> None:
    import json

    row = {
        "schema_version": m.STRATEGY_DATASET_SCHEMA_VERSION,
        "combat_policy": "mcts_2000",
        "kind": "map",
        "obs": [0.0, 1.0],
        "descs": [[0.0], [1.0]],
        "current_armg_index": 0,
        "teacher_best_index": 1,
        "target_probs": [0.1, 0.9],
        "priority": 2.0,
        "teacher_margin": 4.0,
    }
    focus_path = tmp_path / "fresh.jsonl"
    focus_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    focus_ids = m._focus_identity_set([focus_path])

    assert m._example_focus_weight(
        row,
        focus_ids=focus_ids,
        focus_weight=6.0,
    ) == 6.0

    other = dict(row)
    other["obs"] = [9.0, 9.0]
    assert m._example_focus_weight(
        other,
        focus_ids=focus_ids,
        focus_weight=6.0,
    ) == 1.0


def test_strategy_v6_dev_confirmation_gate_is_twenty_seed_non_regression() -> None:
    gate = m.DEV_CONFIRM_20_GATE
    assert gate.expected_seed_count == 20
    assert gate.min_win_delta == 0
    assert gate.min_floor_delta == 0.0
