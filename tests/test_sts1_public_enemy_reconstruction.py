from roguelike_ai.sts1_teacher.enemy_reconstruction import assess_public_enemies


def jaw_worm_state() -> dict:
    return {
        "enemies": [
            {
                "index": 0,
                "name": "Jaw Worm",
                "hp": 40,
                "max_hp": 40,
                "block": 0,
                "intent": "CHOMP",
                "intent_damage": 11,
                "intent_hits": 1,
                "is_gone": False,
                "powers": [{"name": "Strength", "amount": 3}],
            }
        ]
    }


def test_single_jaw_worm_public_surface_is_admitted() -> None:
    result = assess_public_enemies(jaw_worm_state())
    assert result.allowed is True
    assert result.reasons == ()
    assert result.enemy_count == 1


def test_unknown_monster_fails_closed() -> None:
    state = jaw_worm_state()
    state["enemies"][0]["name"] = "Cultist"
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "enemy_unsupported_v1:CULTIST" in result.reasons


def test_missing_current_public_intent_fails_closed() -> None:
    state = jaw_worm_state()
    state["enemies"][0].pop("intent")
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "missing_public_intent:enemies[0]" in result.reasons


def test_unmodelled_enemy_power_fails_closed() -> None:
    state = jaw_worm_state()
    state["enemies"][0]["powers"] = [{"name": "Ritual", "amount": 3}]
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "enemy_power_unsupported_v1:RITUAL" in result.reasons


def test_multi_enemy_combat_is_not_silently_accepted_in_v1() -> None:
    state = jaw_worm_state()
    second = dict(state["enemies"][0])
    second["index"] = 1
    state["enemies"].append(second)
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "enemy_count_unsupported_v1:2" in result.reasons
