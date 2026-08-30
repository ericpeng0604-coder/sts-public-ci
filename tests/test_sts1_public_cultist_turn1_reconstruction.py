from copy import deepcopy

from roguelike_ai.sts1_teacher.cultist_turn1_reconstruction import assess_public_cultist_turn1_player
from roguelike_ai.sts1_teacher.enemy_reconstruction import assess_public_enemies


def card(card_id: str, cost: int) -> dict:
    return {"id": card_id, "upgrades": 0, "cost": cost}


def state() -> dict:
    return {
        "turn": 1,
        "ascension_level": 0,
        "energy": 3,
        "block": 0,
        "powers": [],
        "hand": [card("STRIKE_RED", 1), card("STRIKE_RED", 1), card("DEFEND_RED", 1), card("DEFEND_RED", 1), card("BASH", 2)],
        "draw_pile": [],
        "discard_pile": [card("STRIKE_RED", 1), card("STRIKE_RED", 1), card("STRIKE_RED", 1), card("DEFEND_RED", 1), card("DEFEND_RED", 1)],
        "exhaust_pile": [],
        "enemies": [
            {
                "index": 0,
                "name": "CULTIST",
                "hp": 48,
                "max_hp": 48,
                "block": 0,
                "intent": "CULTIST_DARK_STRIKE",
                "intent_damage": 6,
                "intent_hits": 1,
                "is_gone": False,
                "powers": [],
            }
        ],
    }


def test_cultist_turn1_pristine_public_boundary_is_admitted() -> None:
    s = state()
    player = assess_public_cultist_turn1_player(s)
    enemy = assess_public_enemies(s)
    assert player.allowed is True and player.reasons == ()
    assert enemy.allowed is True and enemy.reasons == ()


def test_cultist_turn1_invalid_public_energy_fails_closed() -> None:
    s = state()
    s["energy"] = 2
    result = assess_public_cultist_turn1_player(s)
    assert result.allowed is False
    assert "cultist_turn1_energy_boundary:2" in result.reasons


def test_cultist_turn1_unsupported_intent_fails_closed() -> None:
    s = state()
    s["enemies"][0]["intent"] = "CULTIST_INCANTATION"
    result = assess_public_enemies(s)
    assert result.allowed is False
    assert "cultist_turn1_intent_mismatch_v1:CULTIST_INCANTATION" in result.reasons


def test_cultist_turn2_stays_fail_closed() -> None:
    s = state()
    s["turn"] = 2
    result = assess_public_enemies(s)
    assert result.allowed is False
    assert "cultist_later_turn_unsupported_v1" in result.reasons


def test_cultist_turn1_impossible_enemy_damage_fails_closed() -> None:
    s = state()
    s["enemies"][0]["hp"] = 42
    result = assess_public_enemies(s)
    assert result.allowed is False
    assert "cultist_turn1_requires_pristine_enemy_hp_v1" in result.reasons


def test_cultist_turn1_projected_power_fails_closed() -> None:
    s = state()
    s["enemies"][0]["powers"] = [{"name": "Strength", "amount": 1}]
    result = assess_public_enemies(s)
    assert result.allowed is False
    assert "cultist_turn1_requires_no_projected_enemy_powers_v1" in result.reasons


def test_cultist_turn1_missing_required_card_pile_fails_closed() -> None:
    s = state()
    del s["discard_pile"]
    result = assess_public_cultist_turn1_player(s)
    assert result.allowed is False
    assert "cultist_turn1_card_piles_not_sequences" in result.reasons


def test_cultist_turn1_nonstarter_card_fails_closed() -> None:
    s = state()
    s["hand"][0] = card("ANGER", 0)
    result = assess_public_cultist_turn1_player(s)
    assert result.allowed is False
    assert "cultist_turn1_nonstarter_card:ANGER" in result.reasons


def test_cultist_opening_regression_still_requires_incantation() -> None:
    opening = deepcopy(state())
    opening["turn"] = 0
    opening["enemies"][0]["intent"] = "CULTIST_INCANTATION"
    assert assess_public_enemies(opening).allowed is True
    opening["enemies"][0]["intent"] = "CULTIST_DARK_STRIKE"
    result = assess_public_enemies(opening)
    assert result.allowed is False
    assert "cultist_opening_intent_mismatch_v1:CULTIST_DARK_STRIKE" in result.reasons
