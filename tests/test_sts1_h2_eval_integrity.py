import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sts1"))

from sts1_g7_h2_elite_route_eval import (  # noqa: E402
    EXPECTED_POOL_SHA256,
    INVENTORY_HASH_MODE,
    POOL_ID,
    ROUND_POOL_FILES,
    SIMULATOR_SOURCE_SHA256,
    EvaluationIntegrityError,
    _check_terminal_trace,
    _inventory_sha256,
)


def test_h6_registration_uses_reserved_round005_pool_and_explicit_pool_files():
    assert POOL_ID == "round-005-20261009-train_hypothesis_2"
    assert EXPECTED_POOL_SHA256 == "79ef213f04549431937649f30c0a2087184c3fc14040e8adf48cced9b4099200"
    assert INVENTORY_HASH_MODE == "raw"
    assert SIMULATOR_SOURCE_SHA256 == "fe735348978f2886fe7b5bc3a840c743c0de6fc3e37f124ceb6644367a1e0a49"
    assert tuple(ROUND_POOL_FILES.values()) == (
        "train_hypothesis_1.json",
        "train_hypothesis_2.json",
        "train_hypothesis_3.json",
        "probe.json",
        "dev.json",
    )


def test_h6_exclusion_inventory_hash_uses_exact_registered_file_bytes(tmp_path):
    inventory_bytes = b'{ "source": "fixed", "count": 3 }\r\n'
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_bytes(inventory_bytes)

    assert _inventory_sha256(inventory_path, {"source": "fixed", "count": 3}) == (
        hashlib.sha256(inventory_bytes).hexdigest()
    )


def _trace_file(
    tmp_path: Path,
    *,
    status: str,
    hp_ratio: float | None,
    recommended_room: str | None,
    actual_room: str | None,
    overridden: bool,
    reason: str | None,
    route_rooms: list[str],
    recommended_index: int = 0,
    actual_index: int = 0,
) -> Path:
    choices = [
        {
            "legal_action_index": index,
            "target_node": {"x": index, "y": 1},
            "target_room": room,
        }
        for index, room in enumerate(route_rooms)
    ]
    trace = [
        {
            "type": "noncombat_decision_trace_v1",
            "screen_before": "MAP_SCREEN",
            "legal_choices_complete": True,
            "legal_choices": [{} for _ in choices],
            "selected_legal_action_index": actual_index,
            "recommended_legal_action_index": recommended_index,
            "map_policy_intervention": {
                "status": status,
                "reason": reason,
                "hp_ratio": hp_ratio,
                "recommended_index": recommended_index,
                "actual_index": actual_index,
                "recommended_room": recommended_room,
                "actual_room": actual_room,
                "overridden": overridden,
            },
            "route": {
                "choices_complete": True,
                "choice_count": len(choices),
                "choices": choices,
                "recommended_selected_index": recommended_index,
                "actual_selected_index": actual_index,
                "recommended_route": {
                    "legal_action_index": recommended_index,
                    "room": route_rooms[recommended_index],
                },
                "selected_route": {
                    "legal_action_index": actual_index,
                    "room": route_rooms[actual_index],
                },
            },
        },
        {
            "type": "terminal_trace_v1",
            "complete": True,
            "legal_actions_complete": True,
            "illegal_action_count": 0,
            "timeout_count": 0,
            "crash_count": 0,
        },
    ]
    path = tmp_path / "trace.ndjson"
    path.write_text("".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8")
    return path


def test_candidate_trace_accepts_high_hp_unchanged_map_choice(tmp_path):
    path = _trace_file(
        tmp_path,
        status="unchanged",
        hp_ratio=0.75,
        recommended_room=None,
        actual_room=None,
        overridden=False,
        reason="hp_not_below_half",
        route_rooms=["ELITE"],
    )

    result = _check_terminal_trace(
        path, candidate=True, expected_overrides=0, expected_fail_closed=0
    )

    assert result["map_decisions"] == 1
    assert result["low_hp_elite_recommendations"] == 0
    assert result["overrides"] == 0
    assert result["fail_closed_count"] == 0


def test_candidate_trace_accepts_low_hp_elite_override(tmp_path):
    path = _trace_file(
        tmp_path,
        status="overridden",
        hp_ratio=0.4,
        recommended_room="ELITE",
        actual_room="REST",
        overridden=True,
        reason="low_hp_elite_replaced_by_highest_scored_non_elite",
        route_rooms=["ELITE", "REST"],
        actual_index=1,
    )

    result = _check_terminal_trace(
        path, candidate=True, expected_overrides=1, expected_fail_closed=0
    )

    assert result["low_hp_elite_recommendations"] == 1
    assert result["overrides"] == 1


def test_candidate_trace_accepts_fail_closed_unknown_room_when_action_is_unchanged(tmp_path):
    path = _trace_file(
        tmp_path,
        status="fail_closed",
        hp_ratio=0.4,
        recommended_room=None,
        actual_room=None,
        overridden=False,
        reason="map_room_or_target_unknown",
        route_rooms=["INVALID"],
    )

    result = _check_terminal_trace(
        path, candidate=True, expected_overrides=0, expected_fail_closed=1
    )

    assert result["fail_closed_count"] == 1
    assert result["fail_closed_reasons"] == {"map_room_or_target_unknown": 1}


def test_candidate_trace_rejects_fail_closed_action_change(tmp_path):
    path = _trace_file(
        tmp_path,
        status="fail_closed",
        hp_ratio=0.4,
        recommended_room=None,
        actual_room=None,
        overridden=False,
        reason="map_room_or_target_unknown",
        route_rooms=["INVALID", "REST"],
        actual_index=1,
    )

    with pytest.raises(EvaluationIntegrityError, match="fail-closed map decision changed"):
        _check_terminal_trace(
            path, candidate=True, expected_overrides=0, expected_fail_closed=1
        )


def test_candidate_trace_requires_map_decision_events(tmp_path):
    path = tmp_path / "terminal-only.ndjson"
    path.write_text(
        json.dumps(
            {
                "type": "terminal_trace_v1",
                "complete": True,
                "legal_actions_complete": True,
                "illegal_action_count": 0,
                "timeout_count": 0,
                "crash_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationIntegrityError, match="candidate map-decision trace is missing"):
        _check_terminal_trace(
            path, candidate=True, expected_overrides=0, expected_fail_closed=0
        )
