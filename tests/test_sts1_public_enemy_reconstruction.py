from roguelike_ai.sts1_teacher.enemy_reconstruction import assess_public_enemies


def jaw_worm_state() -> dict:
    return {
        "turn": 0,
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 40, "max_hp": 40, "block": 0, "intent": "CHOMP", "intent_damage": 11, "intent_hits": 1, "is_gone": False, "powers": [{"name": "Strength", "amount": 3}]}],
    }


def cultist_state() -> dict:
    return {
        "turn": 0,
        "enemies": [{"index": 0, "name": "Cultist", "hp": 48, "max_hp": 48, "block": 0, "intent": "CULTIST_INCANTATION", "intent_damage": 0, "intent_hits": 0, "is_gone": False, "powers": []}],
    }


def gremlin_nob_state() -> dict:
    return {
        "turn": 0,
        "enemies": [{"index": 0, "name": "Gremlin Nob", "hp": 82, "max_hp": 82, "block": 0, "intent": "GREMLIN_NOB_BELLOW", "intent_damage": 0, "intent_hits": 0, "is_gone": False, "powers": []}],
    }


def blue_slaver_state(intent: str = "BLUE_SLAVER_STAB") -> dict:
    return {
        "turn": 0,
        "enemies": [{"index": 0, "name": "Blue Slaver", "hp": 46, "max_hp": 46, "block": 0, "intent": intent, "intent_damage": 12, "intent_hits": 1, "is_gone": False, "powers": []}],
    }


def test_single_jaw_worm_public_surface_is_admitted() -> None:
    result = assess_public_enemies(jaw_worm_state())
    assert result.allowed is True
    assert result.reasons == ()
    assert result.enemy_count == 1


def test_single_cultist_opening_public_surface_is_admitted() -> None:
    result = assess_public_enemies(cultist_state())
    assert result.allowed is True
    assert result.reasons == ()
    assert result.enemy_count == 1


def test_single_gremlin_nob_opening_public_surface_is_admitted() -> None:
    result = assess_public_enemies(gremlin_nob_state())
    assert result.allowed is True
    assert result.reasons == ()
    assert result.enemy_count == 1


def test_blue_slaver_opening_stab_and_rake_are_publicly_admitted() -> None:
    for intent in ("BLUE_SLAVER_STAB", "BLUE_SLAVER_RAKE"):
        result = assess_public_enemies(blue_slaver_state(intent))
        assert result.allowed is True
        assert result.reasons == ()
        assert result.enemy_count == 1


def test_impossible_cultist_opening_dark_strike_fails_closed() -> None:
    state = cultist_state()
    state["enemies"][0]["intent"] = "CULTIST_DARK_STRIKE"
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "cultist_opening_intent_mismatch_v1:CULTIST_DARK_STRIKE" in result.reasons


def test_impossible_gremlin_nob_opening_rush_fails_closed() -> None:
    state = gremlin_nob_state()
    state["enemies"][0]["intent"] = "GREMLIN_NOB_RUSH"
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "gremlin_nob_opening_intent_mismatch_v1:GREMLIN_NOB_RUSH" in result.reasons


def test_impossible_blue_slaver_opening_intent_fails_closed() -> None:
    state = blue_slaver_state("BLUE_SLAVER_UNKNOWN")
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "blue_slaver_opening_intent_mismatch_v1:BLUE_SLAVER_UNKNOWN" in result.reasons


def test_unknown_monster_fails_closed() -> None:
    state = jaw_worm_state()
    state["enemies"][0]["name"] = "Fungi Beast"
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "enemy_unsupported_v1:FUNGI_BEAST" in result.reasons


def test_missing_current_public_intent_fails_closed() -> None:
    state = jaw_worm_state()
    state["enemies"][0].pop("intent")
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "missing_public_intent:enemies[0]" in result.reasons


def test_unmodelled_enemy_power_fails_closed() -> None:
    state = cultist_state()
    state["enemies"][0]["powers"] = [{"name": "Ritual", "amount": 3}]
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "enemy_power_unsupported_v1:RITUAL" in result.reasons


def test_gremlin_nob_enrage_is_not_silently_admitted_as_opening_power() -> None:
    state = gremlin_nob_state()
    state["enemies"][0]["powers"] = [{"name": "Enrage", "amount": 2}]
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "enemy_power_unsupported_v1:ENRAGE" in result.reasons


def test_multi_enemy_combat_is_not_silently_accepted_in_v1() -> None:
    state = jaw_worm_state()
    second = dict(state["enemies"][0])
    second["index"] = 1
    state["enemies"].append(second)
    result = assess_public_enemies(state)
    assert result.allowed is False
    assert "enemy_count_unsupported_v1:2" in result.reasons
