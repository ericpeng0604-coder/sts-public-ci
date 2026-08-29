from roguelike_ai.sts1_teacher.player_reconstruction import assess_public_player
from roguelike_ai.sts1_teacher.run_reconstruction import assess_public_run_state


def base_state() -> dict:
    return {
        "source": "simulator",
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


def jaw_worm_turn_one_boundary() -> dict:
    state = base_state()
    state.update(
        {
            "turn": 1,
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
    )
    return state


def test_v1_player_and_run_slice_is_admitted() -> None:
    state = base_state()
    assert assess_public_player(state).allowed is True
    run = assess_public_run_state(state)
    assert run.relics_allowed is True
    assert run.potions_allowed is True
    assert run.reasons == ()


def test_real_game_source_is_admitted_for_same_public_run_surface() -> None:
    state = base_state()
    state["source"] = "real_game"
    result = assess_public_run_state(state)
    assert result.relics_allowed is True
    assert result.potions_allowed is True
    assert result.reasons == ()


def test_unknown_source_still_fails_closed() -> None:
    state = base_state()
    state["source"] = "replay_file"
    result = assess_public_run_state(state)
    assert result.relics_allowed is False
    assert result.potions_allowed is False
    assert "source_unsupported_v1" in result.reasons


def test_second_simulator_turn_without_boundary_proof_fails_closed() -> None:
    state = base_state()
    state["turn"] = 1
    result = assess_public_player(state)
    assert result.allowed is False
    assert "turn1_requires_single_enemy" in result.reasons


def test_jaw_worm_second_turn_fresh_boundary_is_admitted() -> None:
    result = assess_public_player(jaw_worm_turn_one_boundary())
    assert result.allowed is True
    assert result.reasons == ()


def test_jaw_worm_second_turn_after_one_defend_is_admitted() -> None:
    state = jaw_worm_turn_one_boundary()
    state["energy"] = 2
    state["block"] = 5
    played = state["hand"].pop(3)
    state["discard_pile"].append(played)
    result = assess_public_player(state)
    assert result.allowed is True
    assert result.reasons == ()


def test_jaw_worm_second_turn_after_one_strike_is_admitted() -> None:
    state = jaw_worm_turn_one_boundary()
    state["energy"] = 2
    played = state["hand"].pop(0)
    state["discard_pile"].append(played)
    result = assess_public_player(state)
    assert result.allowed is True
    assert result.reasons == ()


def test_jaw_worm_second_turn_after_bash_is_admitted() -> None:
    state = jaw_worm_turn_one_boundary()
    state["energy"] = 1
    played = state["hand"].pop()
    state["discard_pile"].append(played)
    result = assess_public_player(state)
    assert result.allowed is True
    assert result.reasons == ()


def test_jaw_worm_second_turn_impossible_energy_spend_fails_closed() -> None:
    state = jaw_worm_turn_one_boundary()
    state["energy"] = 0
    played = state["hand"].pop(0)
    state["discard_pile"].append(played)
    result = assess_public_player(state)
    assert result.allowed is False
    assert "turn1_energy_spend_unreachable:played=1:spent=3" in result.reasons


def test_jaw_worm_second_turn_impossible_block_fails_closed() -> None:
    state = jaw_worm_turn_one_boundary()
    state["energy"] = 2
    state["block"] = 4
    played = state["hand"].pop(3)
    state["discard_pile"].append(played)
    result = assess_public_player(state)
    assert result.allowed is False
    assert "turn1_block_unreachable:4" in result.reasons


def test_jaw_worm_second_turn_wrong_pile_boundary_fails_closed() -> None:
    state = jaw_worm_turn_one_boundary()
    state["draw_pile"].append(state["discard_pile"].pop())
    result = assess_public_player(state)
    assert result.allowed is False
    assert "turn1_draw_not_empty:1" in result.reasons


def test_jaw_worm_second_turn_wrong_starter_composition_fails_closed() -> None:
    state = jaw_worm_turn_one_boundary()
    state["hand"][0]["id"] = "DEFEND_RED"
    result = assess_public_player(state)
    assert result.allowed is False
    assert "turn1_starter_composition_mismatch" in result.reasons


def test_second_turn_non_jaw_worm_fails_closed() -> None:
    state = jaw_worm_turn_one_boundary()
    state["enemies"][0]["name"] = "CULTIST"
    result = assess_public_player(state)
    assert result.allowed is False
    assert "turn1_requires_jaw_worm" in result.reasons


def test_second_turn_player_power_fails_closed_even_if_normally_supported() -> None:
    state = jaw_worm_turn_one_boundary()
    state["powers"] = [{"name": "Strength", "amount": 1}]
    result = assess_public_player(state)
    assert result.allowed is False
    assert "turn1_requires_no_player_powers" in result.reasons


def test_third_simulator_turn_still_fails_closed() -> None:
    state = jaw_worm_turn_one_boundary()
    state["turn"] = 2
    result = assess_public_player(state)
    assert result.allowed is False
    assert "turn_unsupported_v1:2" in result.reasons


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
