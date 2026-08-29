from roguelike_ai.sts1_teacher.player_reconstruction import assess_public_player
from roguelike_ai.sts1_teacher.run_reconstruction import assess_public_run_state


def base_state() -> dict:
    return {
        "hp": 70,
        "max_hp": 80,
        "block": 0,
        "energy": 3,
        "gold": 99,
        "turn": 0,
        "floor": 1,
        "ascension_level": 0,
        "powers": [],
        "character": "IRONCLAD",
        "room": "COMBAT",
        "relics": [{"id": "BURNING_BLOOD", "counter": -1}],
        "potions": [
            {"index": 0, "id": "EMPTY_POTION_SLOT", "empty": True},
            {"index": 1, "id": "EMPTY_POTION_SLOT", "empty": True},
            {"index": 2, "id": "EMPTY_POTION_SLOT", "empty": True},
        ],
    }


def test_v1_player_and_run_slice_is_admitted() -> None:
    state = base_state()
    assert assess_public_player(state).allowed is True
    run = assess_public_run_state(state)
    assert run.relics_allowed is True
    assert run.potions_allowed is True
    assert run.reasons == ()


def test_second_simulator_turn_fails_closed() -> None:
    state = base_state()
    state["turn"] = 1
    result = assess_public_player(state)
    assert result.allowed is False
    assert "turn_unsupported_v1:1" in result.reasons


def test_unimplemented_player_power_fails_closed() -> None:
    state = base_state()
    state["powers"] = [{"name": "Barricade", "amount": 1}]
    result = assess_public_player(state)
    assert result.allowed is False
    assert "player_power_unsupported_v1:BARRICADE" in result.reasons


def test_other_relic_fails_closed() -> None:
    state = base_state()
    state["relics"] = [{"id": "VAJRA", "counter": -1}]
    result = assess_public_run_state(state)
    assert result.relics_allowed is False
    assert "relic_unsupported_v1:VAJRA" in result.reasons


def test_nonempty_potion_fails_closed() -> None:
    state = base_state()
    state["potions"][0] = {"index": 0, "id": "WEAK_POTION", "empty": False}
    result = assess_public_run_state(state)
    assert result.potions_allowed is False
    assert "potion_unsupported_v1:WEAK_POTION" in result.reasons
