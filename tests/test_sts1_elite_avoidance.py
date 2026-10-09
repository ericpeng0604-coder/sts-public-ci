import math

import pytest

from roguelike_ai.sts1_phase3.simulator import _apply_low_hp_elite_route_avoidance


KNOWN_ROOMS = {"MONSTER", "ELITE", "REST", "SHOP", "EVENT", "TREASURE", "BOSS"}


def _route(rooms, *, complete=True):
    choices = [
        {
            "legal_action_index": index,
            "target_node": {"x": index, "y": 2},
            "target_room": room,
        }
        for index, room in enumerate(rooms)
    ]
    return {
        "choices_complete": complete,
        "choice_count": len(choices),
        "choices": choices,
    }


def _apply(index, scores, rooms, *, hp=8, max_hp=20, complete=True):
    return _apply_low_hp_elite_route_avoidance(
        index,
        scores,
        _route(rooms, complete=complete),
        current_hp=hp,
        max_hp=max_hp,
        known_room_names=KNOWN_ROOMS,
    )


def test_low_hp_elite_choice_uses_highest_scored_legal_non_elite():
    actual, detail = _apply(0, [0.9, 0.7, 0.8], ["ELITE", "REST", "SHOP"])

    assert actual == 2
    assert detail["status"] == "overridden"
    assert detail["overridden"] is True
    assert detail["recommended_index"] == 0
    assert detail["actual_index"] == 2
    assert detail["recommended_room"] == "ELITE"
    assert detail["actual_room"] == "SHOP"
    assert detail["hp_ratio"] == 0.4


def test_equal_non_elite_scores_choose_lowest_index():
    actual, detail = _apply(0, [0.9, 0.7, 0.7], ["ELITE", "REST", "SHOP"])

    assert actual == 1
    assert detail["status"] == "overridden"


def test_half_hp_boundary_and_non_elite_recommendation_are_unchanged():
    at_boundary, boundary_detail = _apply(
        0, [0.9, 0.7], ["ELITE", "REST"], hp=10, max_hp=20
    )
    already_safe, safe_detail = _apply(
        1, [0.9, 0.7], ["ELITE", "REST"], hp=8, max_hp=20
    )

    assert at_boundary == 0
    assert boundary_detail["reason"] == "hp_not_below_half"
    assert already_safe == 1
    assert safe_detail["reason"] == "recommended_route_not_elite"


@pytest.mark.parametrize(
    ("index", "scores", "rooms", "hp", "max_hp", "complete", "expected_reason"),
    [
        (0, [0.9, 0.7], ["ELITE", "REST"], 8, 20, False, "map_route_incomplete"),
        (0, [0.9, 0.7], ["ELITE", "UNKNOWN"], 8, 20, True, "map_room_or_target_unknown"),
        (0, [0.9], ["ELITE", "REST"], 8, 20, True, "armg_scores_incomplete"),
        (0, [math.nan, 0.7], ["ELITE", "REST"], 8, 20, True, "armg_score_non_finite"),
        (0, [0.9, 0.7], ["ELITE", "ELITE"], 8, 20, True, "no_known_non_elite_route"),
        (0, [0.9, 0.7], ["ELITE", "REST"], None, 20, True, "player_hp_unavailable_or_invalid"),
        (3, [0.9, 0.7], ["ELITE", "REST"], 8, 20, True, "map_choice_count_or_index_invalid"),
    ],
)
def test_incomplete_or_unsafe_inputs_fail_closed(
    index, scores, rooms, hp, max_hp, complete, expected_reason
):
    actual, detail = _apply(
        index,
        scores,
        rooms,
        hp=hp,
        max_hp=max_hp,
        complete=complete,
    )

    assert actual == index
    assert detail["status"] == "fail_closed"
    assert detail["reason"] == expected_reason
