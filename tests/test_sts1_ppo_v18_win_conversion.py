from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
ROLLOUT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_rollout_v14.py"

SPEC = importlib.util.spec_from_file_location("sts1_ppo_v19_rollout_test", ROLLOUT)
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


def _result(*, passed: bool, victory: bool = False, second=None):
    return {
        "victory": victory,
        "passed_boss": passed,
        "target_boss_floor": 50,
        "final_floor": 51 if passed else 50,
        "second_intervention": second,
    }


def test_near_boss_failure_targets_all_three_acts():
    assert mod._near_boss_failure(16)
    assert mod._near_boss_failure(32)
    assert mod._near_boss_failure(50)
    assert not mod._near_boss_failure(40)
    assert mod._boss_target_floor(32) == 33


def test_one_step_boss_pass_is_valid_conversion(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_force(
        record,
        *,
        choice_index,
        sts,
        policy,
        mcts_sims,
        target_floor,
        second_alternative_rank=None,
    ):
        calls.append((choice_index, mcts_sims, second_alternative_rank))
        return _result(passed=choice_index == 1)

    monkeypatch.setattr(mod, "_force_choice_and_finish", fake_force)
    policy = type("P", (), {"conversion_states": [_record()]})()

    rows = mod._mine_win_conversions(
        policy=policy,
        sts=object(),
        final_floor=50,
        seed=123,
        max_states=6,
        max_alternatives=2,
        max_second_alternatives=2,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "critical_boss_pass_conversion"
    assert row["source"] == mod.CONVERSION_SCHEMA
    assert row["seed"] == 123
    assert row["teacher_best_index"] == 1
    assert row["priority"] == 4.0
    assert (0, 2000, None) in calls
    assert (0, 10000, None) in calls
    assert (1, 2000, None) in calls
    assert (1, 10000, None) in calls


def test_two_step_conversion_emits_both_teaching_decisions(monkeypatch: pytest.MonkeyPatch):
    second = {
        "kind": "shop",
        "floor": 49,
        "act": 3,
        "obs": [2.0, 3.0],
        "descs": [[0.0], [1.0]],
        "current_armg_index": 0,
        "teacher_best_index": 1,
        "target_probs": [0.0, 1.0],
    }

    def fake_force(
        record,
        *,
        choice_index,
        sts,
        policy,
        mcts_sims,
        target_floor,
        second_alternative_rank=None,
    ):
        if choice_index == 0:
            return _result(passed=False)
        if second_alternative_rank == 0:
            return _result(passed=True, second=second)
        return _result(passed=False)

    monkeypatch.setattr(mod, "_force_choice_and_finish", fake_force)
    policy = type("P", (), {"conversion_states": [_record()]})()

    rows = mod._mine_win_conversions(
        policy=policy,
        sts=object(),
        final_floor=50,
        seed=456,
        max_states=6,
        max_alternatives=1,
        max_second_alternatives=1,
    )

    assert len(rows) == 2
    assert rows[0]["type"] == "critical_two_step_boss_conversion"
    assert rows[0]["teacher_best_index"] == 1
    assert rows[1]["type"] == "critical_two_step_boss_conversion_followup"
    assert rows[1]["kind"] == "shop"
    assert rows[1]["teacher_best_index"] == 1
    assert all(row["seed"] == 456 for row in rows)


def test_no_conversion_when_original_choice_already_passes_boss(monkeypatch: pytest.MonkeyPatch):
    def fake_force(
        record,
        *,
        choice_index,
        sts,
        policy,
        mcts_sims,
        target_floor,
        second_alternative_rank=None,
    ):
        if choice_index == 0 and mcts_sims == 10000:
            return _result(passed=True)
        return _result(passed=False)

    monkeypatch.setattr(mod, "_force_choice_and_finish", fake_force)
    policy = type("P", (), {"conversion_states": [_record()]})()
    assert mod._mine_win_conversions(policy=policy, sts=object(), final_floor=50) == []


def test_no_conversion_mining_for_non_boss_loss(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        mod,
        "_force_choice_and_finish",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    policy = type("P", (), {"conversion_states": [_record()]})()
    assert mod._mine_win_conversions(policy=policy, sts=object(), final_floor=40) == []



def test_zero_legal_map_transition_advances_without_consuming_a_choice():
    class ScreenState:
        MAP_SCREEN = "MAP"
        REWARDS = "REWARDS"

    class Sts:
        @staticmethod
        def get_legal_game_actions(gc):
            return []

    Sts.ScreenState = ScreenState

    class Gc:
        screen_state = ScreenState.MAP_SCREEN
        floor_num = 33
        act = 2
        outcome = "UNDECIDED"

    class Agent:
        pause_on_map = True

        def playout(self, gc):
            assert self.pause_on_map is False
            gc.floor_num = 34

    gc = Gc()
    agent = Agent()
    mod._advance_zero_legal_map_transition(agent, gc, Sts)
    assert gc.floor_num == 34
    assert agent.pause_on_map is True


def test_zero_legal_map_transition_refuses_to_hide_real_choice():
    class ScreenState:
        MAP_SCREEN = "MAP"
        REWARDS = "REWARDS"

    class Sts:
        @staticmethod
        def get_legal_game_actions(gc):
            return [object()]

    Sts.ScreenState = ScreenState

    class Gc:
        screen_state = ScreenState.MAP_SCREEN
        floor_num = 33
        act = 2
        outcome = "UNDECIDED"

    class Agent:
        pause_on_map = True

    with pytest.raises(RuntimeError, match="legal player actions"):
        mod._advance_zero_legal_map_transition(Agent(), Gc(), Sts)
