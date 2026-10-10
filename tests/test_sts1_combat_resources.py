from types import SimpleNamespace

import pytest

from roguelike_ai.sts1_phase3.simulator import (
    _combat_potion_inventory_snapshot, _diagnostic_battle_snapshot,
)
from scripts.sts1.sts1_g7_h7_potion_trace_audit import (
    _valid_potion_snapshot, EvaluationIntegrityError,
)

SOURCE = "native_battlecontext_potion_enum_names_v1"
EMPTY = ["EMPTY_POTION_SLOT"] * 5


def test_live_snapshot_does_not_use_stale_run_inventory_or_mutate_sources():
    gc = SimpleNamespace(potions=["ENERGY_POTION"] + EMPTY[1:], deck=[], relics=[])
    battle = SimpleNamespace(combat_potions=list(EMPTY), player=SimpleNamespace())
    state = _diagnostic_battle_snapshot(battle, gc)
    assert state["run"]["potions"][0] == "ENERGY_POTION"
    live = state["combat_potion_inventory"]
    assert live["potions"] == EMPTY
    assert live["potion_inventory_source"] == SOURCE
    _valid_potion_snapshot(live, "live combat", source=SOURCE)
    live["potions"][0] = "FIRE_POTION"
    assert battle.combat_potions == EMPTY


@pytest.mark.parametrize("slots", [None, EMPTY[:4], ["INVALID"] + EMPTY[1:], [None] + EMPTY[1:]])
def test_missing_or_malformed_live_inventory_fails_closed(slots):
    state = _combat_potion_inventory_snapshot(SimpleNamespace(combat_potions=slots))
    assert state["potion_inventory_complete"] is False
    with pytest.raises(EvaluationIntegrityError):
        _valid_potion_snapshot(state, "live combat", source=SOURCE)


def test_run_snapshot_cannot_substitute_for_live_source():
    state = _combat_potion_inventory_snapshot(SimpleNamespace(combat_potions=list(EMPTY)))
    state["potion_inventory_source"] = "native_gamecontext_potion_enum_names_v1"
    with pytest.raises(EvaluationIntegrityError):
        _valid_potion_snapshot(state, "live combat", source=SOURCE)
