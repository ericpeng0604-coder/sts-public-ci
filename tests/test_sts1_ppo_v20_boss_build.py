from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from roguelike_ai.sts1_phase3.simulator import _combat_mcts_budget_for_floor


ROOT=Path(__file__).parents[1]
BUILD_SCRIPT=ROOT / "scripts" / "sts1" / "sts1_ppo_v20_build_trajectory_diagnostic.py"
SPEC=importlib.util.spec_from_file_location("sts1_v20_build_diag_test", BUILD_SCRIPT)
assert SPEC and SPEC.loader
build_mod=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=build_mod
SPEC.loader.exec_module(build_mod)


def test_boss_only_mcts_budget_overrides_base():
    for floor in (16, 33, 50):
        assert _combat_mcts_budget_for_floor(
            floor,
            base_sims=2000,
            boss_sims=10000,
            boss_floors=(16, 33, 50),
        ) == 10000


def test_non_boss_floor_keeps_base_budget():
    assert _combat_mcts_budget_for_floor(
        32,
        base_sims=2000,
        boss_sims=50000,
        boss_floors=(16, 33, 50),
    ) == 2000


def test_boss_budget_has_priority_over_late_schedule():
    assert _combat_mcts_budget_for_floor(
        50,
        base_sims=2000,
        late_sims=10000,
        late_floor=45,
        boss_sims=50000,
        boss_floors=(16, 33, 50),
    ) == 50000
    assert _combat_mcts_budget_for_floor(
        49,
        base_sims=2000,
        late_sims=10000,
        late_floor=45,
        boss_sims=50000,
        boss_floors=(16, 33, 50),
    ) == 10000


def _record():
    return {
        "gc": object(),
        "kind": "rest",
        "selected_index": 0,
        "scores": [1.0, 0.8],
        "floor": 31,
        "act": 2,
        "obs": [0.0, 1.0],
        "descs": [[0.0], [1.0]],
    }


def _second_intervention():
    return {
        "kind": "shop",
        "floor": 32,
        "act": 2,
        "obs": [2.0, 3.0],
        "descs": [[0.0], [1.0]],
        "current_armg_index": 0,
        "teacher_best_index": 1,
        "target_probs": [0.0, 1.0],
    }


def test_early_build_falls_back_to_confirmed_two_step_rescue(monkeypatch: pytest.MonkeyPatch):
    class FakePolicy:
        def __init__(self, **kwargs):
            self.conversion_states=[_record()]

    monkeypatch.setattr(build_mod.rollout, "SamplingArmG", FakePolicy)
    monkeypatch.setattr(build_mod, "_load_sts", lambda path: object())
    monkeypatch.setattr(
        build_mod,
        "run_simulator_game",
        lambda **kwargs: {
            "result":"PASS_SIMULATOR_COMPLETE_RUN",
            "outcome":"defeat",
            "final_floor":33,
        },
    )

    calls=[]
    def fake_finish(
        record,
        *,
        choice_index,
        sts,
        policy,
        mcts_sims,
        boss_mcts_sims,
        target_floor,
        second_alternative_rank=None,
    ):
        calls.append((choice_index,boss_mcts_sims,second_alternative_rank))
        passed=(choice_index==1 and second_alternative_rank==0)
        return {
            "passed_boss":passed,
            "victory":False,
            "final_floor":34 if passed else 33,
            "second_intervention":_second_intervention() if passed else None,
        }

    monkeypatch.setattr(build_mod.rollout, "_force_choice_and_finish", fake_finish)

    row=build_mod.diagnose_seed(
        seed=123,
        module_dir=Path("."),
        armg_root=Path("."),
        weight=Path("fake.pt"),
        heldout=[123],
        max_states=20,
        max_alternatives=1,
        max_two_step_states=8,
        max_second_alternatives=1,
    )

    assert row["status"]=="EARLY_TWO_STEP_BUILD_RESCUE_FOUND"
    assert row["rescue"]["alternative_index"]==1
    assert row["rescue"]["second_alternative_rank"]==0
    assert len(row["rescue"]["teachers"])==2
    assert row["rescue"]["teachers"][0]["type"]=="v20_early_two_step_boss_rescue"
    assert row["rescue"]["teachers"][1]["type"]=="v20_early_two_step_boss_rescue_followup"
    assert (1,10000,0) in calls
    assert (1,50000,0) in calls


def test_second_step_must_pass_both_boss_budgets(monkeypatch: pytest.MonkeyPatch):
    class FakePolicy:
        def __init__(self, **kwargs):
            self.conversion_states=[_record()]

    monkeypatch.setattr(build_mod.rollout, "SamplingArmG", FakePolicy)
    monkeypatch.setattr(build_mod, "_load_sts", lambda path: object())
    monkeypatch.setattr(
        build_mod,
        "run_simulator_game",
        lambda **kwargs: {
            "result":"PASS_SIMULATOR_COMPLETE_RUN",
            "outcome":"defeat",
            "final_floor":33,
        },
    )

    def fake_finish(
        record,
        *,
        choice_index,
        sts,
        policy,
        mcts_sims,
        boss_mcts_sims,
        target_floor,
        second_alternative_rank=None,
    ):
        # Two-step branch passes the 10k screen but fails 50k confirmation.
        passed=(
            choice_index==1
            and second_alternative_rank==0
            and boss_mcts_sims==10000
        )
        return {
            "passed_boss":passed,
            "victory":False,
            "final_floor":34 if passed else 33,
            "second_intervention":_second_intervention() if passed else None,
        }

    monkeypatch.setattr(build_mod.rollout, "_force_choice_and_finish", fake_finish)

    row=build_mod.diagnose_seed(
        seed=456,
        module_dir=Path("."),
        armg_root=Path("."),
        weight=Path("fake.pt"),
        heldout=[456],
        max_states=20,
        max_alternatives=1,
        max_two_step_states=8,
        max_second_alternatives=1,
    )
    assert row["status"]=="NO_EARLY_BUILD_RESCUE"
    assert row["rescue"] is None
