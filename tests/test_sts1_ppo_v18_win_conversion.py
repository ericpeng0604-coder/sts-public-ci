from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
ROLLOUT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_rollout_v14.py"

SPEC = importlib.util.spec_from_file_location("sts1_ppo_v18_rollout_test", ROLLOUT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def _record():
    return {
        "gc": object(),
        "kind": "rest",
        "selected_index": 0,
        "scores": [1.0, 0.5, 0.1],
        "floor": 49,
        "act": 3,
        "hp": 18,
        "max_hp": 80,
        "obs": [0.0, 1.0],
        "descs": [[0.0], [1.0], [2.0]],
    }


def test_near_boss_failure_targets_all_three_acts():
    assert mod._near_boss_failure(16)
    assert mod._near_boss_failure(32)
    assert mod._near_boss_failure(50)
    assert not mod._near_boss_failure(40)


def test_win_conversion_requires_confirmed_loss_to_confirmed_win(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_force(record, *, choice_index, sts, policy, mcts_sims):
        calls.append((choice_index, mcts_sims))
        # Original choice 0 loses at both budgets. Alternative 1 wins at both.
        return {"victory": choice_index == 1, "final_floor": 51 if choice_index == 1 else 50}

    monkeypatch.setattr(mod, "_force_choice_and_finish", fake_force)
    policy = type("P", (), {"conversion_states": [_record()]})()

    rows = mod._mine_win_conversions(
        policy=policy,
        sts=object(),
        final_floor=50,
        max_states=2,
        max_alternatives=3,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == mod.CONVERSION_SCHEMA
    assert row["current_armg_index"] == 0
    assert row["teacher_best_index"] == 1
    assert row["target_probs"] == [0.0, 1.0, 0.0]
    assert row["confidence_weight"] == 1.0
    assert row["teacher_consensus_fraction"] == 1.0
    assert row["combat_policy"] == "mcts_2000"
    assert row["confirmation_policy"] == "mcts_10000"
    assert (0, 2000) in calls and (0, 10000) in calls
    assert (1, 2000) in calls and (1, 10000) in calls


def test_no_conversion_when_original_choice_survives_confirm_budget(monkeypatch: pytest.MonkeyPatch):
    def fake_force(record, *, choice_index, sts, policy, mcts_sims):
        if choice_index == 0 and mcts_sims == 10000:
            return {"victory": True, "final_floor": 51}
        return {"victory": False, "final_floor": 50}

    monkeypatch.setattr(mod, "_force_choice_and_finish", fake_force)
    policy = type("P", (), {"conversion_states": [_record()]})()

    rows = mod._mine_win_conversions(
        policy=policy,
        sts=object(),
        final_floor=50,
    )
    assert rows == []


def test_no_conversion_mining_for_non_boss_loss(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        mod,
        "_force_choice_and_finish",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    policy = type("P", (), {"conversion_states": [_record()]})()
    assert mod._mine_win_conversions(policy=policy, sts=object(), final_floor=40) == []
