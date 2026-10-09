from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

from roguelike_ai.sts1_phase3.simulator import (
    _diagnostic_run_snapshot,
    public_run_state,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sts1"))
from sts1_g7_h7_potion_trace_audit import (  # noqa: E402
    EvaluationIntegrityError,
    _validate_trace,
)


def _game_context(potions: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        floor_num=12,
        act=1,
        gold=83,
        screen_state="BATTLE",
        cur_map_node_x=2,
        cur_map_node_y=3,
        cur_hp=31,
        max_hp=72,
        deck=[],
        relics=[],
        potions=potions,
    )


def test_diagnostic_snapshot_preserves_exact_five_native_slots_without_policy_leak():
    slots = [
        "EMPTY_POTION_SLOT",
        "FIRE_POTION",
        "EMPTY_POTION_SLOT",
        "BLOCK_POTION",
        "EMPTY_POTION_SLOT",
    ]
    gc = _game_context(slots)

    policy_state = public_run_state(gc)
    diagnostic_state = _diagnostic_run_snapshot(gc)

    assert "potions" not in policy_state
    assert "potion_inventory_complete" not in policy_state
    assert diagnostic_state["potions"] == slots
    assert diagnostic_state["potion_inventory_complete"] is True
    assert diagnostic_state["potion_inventory_source"] == (
        "native_gamecontext_potion_enum_names_v1"
    )
    assert diagnostic_state["potion_inventory_reason"] == "exactly_five_valid_native_slots"


@pytest.mark.parametrize(
    ("slots", "expected_reason"),
    [
        (["EMPTY_POTION_SLOT"] * 4, "expected_exactly_five_slots"),
        (["EMPTY_POTION_SLOT"] * 6, "expected_exactly_five_slots"),
        (["EMPTY_POTION_SLOT"] * 4 + [None], "unknown_or_invalid_native_slot"),
        (["EMPTY_POTION_SLOT"] * 4 + ["INVALID"], "unknown_or_invalid_native_slot"),
        (["EMPTY_POTION_SLOT"] * 4 + ["UNKNOWN"], "unknown_or_invalid_native_slot"),
    ],
)
def test_diagnostic_snapshot_fails_closed_for_incomplete_or_unknown_slots(
    slots: object, expected_reason: str
):
    state = _diagnostic_run_snapshot(_game_context(slots))

    assert state["potions"] is None
    assert state["potion_inventory_complete"] is False
    assert state["potion_inventory_source"] == "native_gamecontext_potion_enum_names_v1"
    assert state["potion_inventory_reason"] == expected_reason


def test_diagnostic_snapshot_marks_missing_binding_property_unavailable():
    gc = _game_context()
    del gc.potions

    state = _diagnostic_run_snapshot(gc)

    assert state["potions"] is None
    assert state["potion_inventory_complete"] is False
    assert state["potion_inventory_source"] == "unavailable"
    assert state["potion_inventory_reason"] == "native_potion_property_missing"


def _valid_trace_state() -> dict[str, object]:
    return {
        "potions": ["EMPTY_POTION_SLOT"] * 5,
        "potion_inventory_complete": True,
        "potion_inventory_source": "native_gamecontext_potion_enum_names_v1",
        "potion_inventory_reason": "exactly_five_valid_native_slots",
        "deck": [],
        "relics": [],
    }


def _valid_trace_events(metadata: dict[str, object]) -> list[dict[str, object]]:
    state = _valid_trace_state()
    combat_state = {key: value for key, value in state.items() if key not in {"deck", "relics"}}
    return [
        {
            "type": "diagnostic_trace_header_v1",
            "trace_schema": "sts1-diagnostic-trace-v1",
            "run_metadata": metadata,
        },
        {"type": "encounter_started_v1", "state": {"run": state}},
        {
            "type": "combat_decision_trace_v1",
            "mcts_sims": 2000,
            "legal_actions_complete": True,
            "canonical_native_legal_actions": [{"kind": "end_turn"}],
            "policy_legal_actions": [{"kind": "end_turn"}],
            "selected_native_action_index": 0,
            "selected_public_action_index": 0,
            "public_state": combat_state,
        },
        {
            "type": "noncombat_decision_trace_v1",
            "screen_before": "MAP_SCREEN",
            "legal_choices_complete": True,
            "legal_choices": [{"kind": "map_choice"}],
            "state_before": state,
            "state_after": state,
            "route": {"choices_complete": True},
        },
        {
            "type": "terminal_trace_v1",
            "complete": True,
            "outcome": "defeat",
            "error": None,
            "legal_actions_complete": True,
            "illegal_action_count": 0,
            "timeout_count": 0,
            "crash_count": 0,
            "potion_inventory_snapshot_complete": True,
            "potion_inventory_source": "native_gamecontext_potion_enum_names_v1",
            "potion_inventory_reason": "exactly_five_valid_native_slots",
            "final_state": state,
        },
    ]


def test_h7_trace_validator_accepts_complete_scoped_diagnostic(tmp_path: Path):
    import json

    metadata = {"stage": "h7-test", "mcts_sims": 2000}
    path = tmp_path / "trace.ndjson"
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in _valid_trace_events(metadata)),
        encoding="utf-8",
    )

    stats = _validate_trace(path, {"outcome": "defeat"}, metadata)

    assert stats == {
        "combat_decision_count": 1,
        "encounter_count": 1,
        "noncombat_decision_count": 1,
        "route_decision_count": 1,
        "potion_snapshot_count": 5,
    }


def test_h7_trace_validator_rejects_incomplete_potion_inventory(tmp_path: Path):
    import json

    metadata = {"stage": "h7-test", "mcts_sims": 2000}
    events = _valid_trace_events(metadata)
    events[2]["public_state"] = {**_valid_trace_state(), "potion_inventory_complete": False}
    path = tmp_path / "trace.ndjson"
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationIntegrityError, match="potion-slot inventory"):
        _validate_trace(path, {"outcome": "defeat"}, metadata)


def test_h7_trace_validator_requires_deck_on_run_snapshots_not_combat_projection(tmp_path: Path):
    import json

    metadata = {"stage": "h7-test", "mcts_sims": 2000}
    events = _valid_trace_events(metadata)
    events[1]["state"] = {"run": {key: value for key, value in _valid_trace_state().items() if key != "deck"}}
    path = tmp_path / "trace.ndjson"
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationIntegrityError, match="deck or relic"):
        _validate_trace(path, {"outcome": "defeat"}, metadata)
