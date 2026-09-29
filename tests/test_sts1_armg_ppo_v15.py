from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v15_tournament.py"

SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v15_tournament", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


TRAIN_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_train_v14_sharded.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_train_v14_sharded_v15", TRAIN_SCRIPT)
assert TRAIN_SPEC and TRAIN_SPEC.loader
train_mod = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train_mod
TRAIN_SPEC.loader.exec_module(train_mod)


def row(name: str, *, passed: bool, wins: int, floor: float, safe: bool = True, idx: int = 0):
    return {
        "name": name,
        "rank_key": [
            int(safe),
            int(passed),
            wins,
            floor,
            0,
            -idx,
        ],
    }


def test_tournament_prefers_existing_dev_gate_pass():
    winner = mod.choose_winner(
        [
            row("A", passed=False, wins=5, floor=42.0, idx=0),
            row("B", passed=True, wins=4, floor=41.0, idx=1),
        ]
    )
    assert winner["name"] == "B"


def test_tournament_prefers_more_wins_then_floor():
    winner = mod.choose_winner(
        [
            row("A", passed=True, wins=4, floor=41.0, idx=0),
            row("B", passed=True, wins=5, floor=39.0, idx=1),
            row("C", passed=True, wins=5, floor=40.0, idx=2),
        ]
    )
    assert winner["name"] == "C"


def test_tournament_never_prefers_unsafe_candidate():
    winner = mod.choose_winner(
        [
            row("A", passed=True, wins=8, floor=50.0, safe=False, idx=0),
            row("B", passed=False, wins=3, floor=38.0, safe=True, idx=1),
        ]
    )
    assert winner["name"] == "B"



def _strategy_row(kind: str, *, current: int, teacher: int, priority: float = 2.0):
    return {
        "schema_version": "sts1-armg-strategy-branch-dataset-v1",
        "combat_policy": "mcts_2000",
        "kind": kind,
        "floor": 22,
        "obs": [0.0, 1.0],
        "descs": [[0.0], [1.0]],
        "current_armg_index": current,
        "teacher_best_index": teacher,
        "target_probs": [0.25, 0.75],
        "priority": priority,
        "teacher_margin": 8.0,
    }


def test_strategy_teacher_import_keeps_only_disagreements_and_balances_kinds(tmp_path):
    import json

    path = tmp_path / "strategy-replay.jsonl"
    rows = [
        _strategy_row("map", current=0, teacher=1, priority=3.0),
        _strategy_row("map", current=0, teacher=1, priority=2.0),
        _strategy_row("shop", current=0, teacher=1, priority=2.5),
        _strategy_row("rest", current=1, teacher=1, priority=3.0),
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = train_mod.load_strategy_teacher_examples(
        path,
        max_examples=3,
        combat_policy="mcts_2000",
    )

    assert len(selected) == 3
    assert all(row["current_armg_index"] != row["teacher_best_index"] for row in selected)
    assert {row["kind"] for row in selected[:2]} == {"map", "shop"}
    assert all(abs(sum(row["target_probs"]) - 1.0) < 1e-6 for row in selected)


def test_strategy_teacher_import_fails_closed_on_mcts_policy_drift(tmp_path):
    import json
    import pytest

    path = tmp_path / "strategy-replay.jsonl"
    row = _strategy_row("map", current=0, teacher=1)
    row["combat_policy"] = "mcts_50000"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="combat policy mismatch"):
        train_mod.load_strategy_teacher_examples(
            path,
            max_examples=32,
            combat_policy="mcts_2000",
        )



ADAPT_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v15_adapt.py"
ADAPT_SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v15_adapt_test", ADAPT_SCRIPT)
assert ADAPT_SPEC and ADAPT_SPEC.loader
adapt_mod = importlib.util.module_from_spec(ADAPT_SPEC)
sys.modules[ADAPT_SPEC.name] = adapt_mod
ADAPT_SPEC.loader.exec_module(adapt_mod)


def test_v15_adapt_heals_missing_teacher_parallel_control_keys():
    state = {
        "schema_version": "sts1-armg-ppo-v14-loop-state-v1",
        "round_index": 7,
        "stagnation_count": 6,
        "last_decision": "HOLD_PARENT",
        "last_dev_gate": {
            "win_delta": -1,
            "mean_paired_floor_delta": 0.1,
        },
    }
    order = [
        "attempt",
        "retry_count",
        "games_per_worker",
        "mcts_sims",
        "max_rounds",
        "ppo_version",
    ]
    control = {
        "attempt": "4",
        "retry_count": "0",
        "games_per_worker": "80",
        "mcts_sims": "2000",
        "max_rounds": "0",
        "ppo_version": "1.5",
    }

    updated, report = adapt_mod.adapt(
        state=state,
        control_order=order,
        control=control,
    )

    assert updated["strategy_teacher_coef"] == "0.01"
    assert updated["strategy_teacher_max_examples"] == "512"
    assert updated["strategy_teacher_required"] == "1"
    assert updated["candidate_threads"] == "4"
    assert updated["mcts_sims"] == "2000"
    assert updated["max_rounds"] == "0"
    assert all(
        key in order
        for key in (
            "strategy_teacher_coef",
            "strategy_teacher_max_examples",
            "strategy_teacher_required",
            "candidate_threads",
        )
    )
    assert report["production_champion_changed"] is False
