from __future__ import annotations

from roguelike_ai.sts1_phase3.simulator import _combat_mcts_budget_for_floor


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
