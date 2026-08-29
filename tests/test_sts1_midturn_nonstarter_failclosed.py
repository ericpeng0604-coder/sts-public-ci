from copy import deepcopy

from roguelike_ai.sts1_teacher.player_reconstruction import assess_public_player


def _starter_turn_one_state() -> dict:
    return {
        "source": "simulator",
        "hp": 70,
        "max_hp": 80,
        "block": 0,
        "energy": 3,
        "gold": 99,
        "turn": 1,
        "floor": 1,
        "ascension_level": 0,
        "powers": [],
        "character": "IRONCLAD",
        "room": "COMBAT",
        "hand": [
            {"id": "STRIKE_RED"},
            {"id": "STRIKE_RED"},
            {"id": "DEFEND_RED"},
            {"id": "DEFEND_RED"},
            {"id": "BASH"},
        ],
        "draw_pile": [],
        "discard_pile": [
            {"id": "STRIKE_RED"},
            {"id": "STRIKE_RED"},
            {"id": "STRIKE_RED"},
            {"id": "DEFEND_RED"},
            {"id": "DEFEND_RED"},
        ],
        "exhaust_pile": [],
        "enemies": [{"name": "JAW_WORM"}],
    }


def _replace_hand_card(card_id: str) -> dict:
    state = deepcopy(_starter_turn_one_state())
    state["hand"][0] = {"id": card_id}
    return state


def test_pommel_strike_draw_card_fails_closed() -> None:
    result = assess_public_player(_replace_hand_card("POMMEL_STRIKE"))
    assert result.allowed is False
    assert "turn1_nonstarter_card:POMMEL_STRIKE" in result.reasons
    assert "turn1_starter_composition_mismatch" in result.reasons


def test_shrug_it_off_draw_card_fails_closed() -> None:
    result = assess_public_player(_replace_hand_card("SHRUG_IT_OFF"))
    assert result.allowed is False
    assert "turn1_nonstarter_card:SHRUG_IT_OFF" in result.reasons
    assert "turn1_starter_composition_mismatch" in result.reasons


def test_generated_card_fails_closed() -> None:
    result = assess_public_player(_replace_hand_card("ANGER"))
    assert result.allowed is False
    assert "turn1_nonstarter_card:ANGER" in result.reasons
