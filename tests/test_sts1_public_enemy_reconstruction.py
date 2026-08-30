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


def red_slaver_state(intent: str = "RED_SLAVER_STAB", turn: int = 0) -> dict:
    return {
        "turn": turn,
        "enemies": [{"index": 0, "name": "Red Slaver", "hp": 46, "max_hp": 46, "block": 0, "intent": intent, "intent_damage": 13, "intent_hits": 1, "is_gone": False, "powers": []}],
    }


def looter_state(
    *,
    intent: str = "LOOTER_MUG",
    turn: int = 0,
    ascension: int = 0,
    thievery: int | None = None,
    expose_thievery: bool = True,
    gold: int = 99,
) -> dict:
    if thievery is None:
        thievery = 20 if ascension >= 17 else 15
    powers = [{"name": "Thievery", "amount": thievery}] if expose_thievery else []
    return {
        "turn": turn,
        "gold": gold,
        "ascension_level": ascension,
        "enemies": [{"index": 0, "name": "Looter", "hp": 46, "max_hp": 46, "block": 0, "intent": intent, "intent_damage": 10 if ascension < 2 else 11, "intent_hits": 1, "is_gone": False, "powers": powers}],
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


def test_red_slaver_opening_stab_is_publicly_admitted() -> None:
    result = assess_public_enemies(red_slaver_state())
    assert result.allowed is True
    assert result.reasons == ()
    assert result.enemy_count == 1


def test_looter_opening_mug_and_public_thievery_are_admitted() -> None:
    for ascension, expected in ((0, 15), (17, 20)):
        result = assess_public_enemies(looter_state(ascension=ascension, thievery=expected))
        assert result.allowed is True
        assert result.reasons == ()
        assert result.enemy_count == 1


def test_looter_opening_without_projected_thievery_is_publicly_derivable() -> None:
    for ascension in (0, 17):
        result = assess_public_enemies(looter_state(ascension=ascension, expose_thievery=False))
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


def test_impossible_red_slaver_opening_intent_fails_closed() -> None:
    result = assess_public_enemies(red_slaver_state("RED_SLAVER_ENTANGLE"))
    assert result.allowed is False
    assert "red_slaver_opening_intent_mismatch_v1:RED_SLAVER_ENTANGLE" in result.reasons


def test_red_slaver_later_turn_fails_closed() -> None:
    result = assess_public_enemies(red_slaver_state(turn=1))
    assert result.allowed is False
    assert "red_slaver_later_turn_unsupported_v1" in result.reasons


def test_looter_impossible_opening_intent_fails_closed() -> None:
    result = assess_public_enemies(looter_state(intent="LOOTER_LUNGE"))
    assert result.allowed is False
    assert "looter_opening_intent_mismatch_v1:LOOTER_LUNGE" in result.reasons


def test_looter_turn_one_positive_gold_is_publicly_admitted() -> None:
    for ascension, expected in ((0, 15), (17, 20)):
        for expose_thievery in (False, True):
            result = assess_public_enemies(
                looter_state(
                    turn=1,
                    ascension=ascension,
                    thievery=expected,
                    expose_thievery=expose_thievery,
                    gold=84,
                )
            )
            assert result.allowed is True
            assert result.reasons == ()


def test_looter_turn_one_zero_gold_fails_closed() -> None:
    result = assess_public_enemies(looter_state(turn=1, gold=0))
    assert result.allowed is False
    assert "looter_turn1_positive_gold_required_v1" in result.reasons


def test_looter_turn_one_non_mug_fails_closed() -> None:
    result = assess_public_enemies(looter_state(turn=1, intent="LOOTER_LUNGE", gold=84))
    assert result.allowed is False
    assert "looter_turn1_intent_mismatch_v1:LOOTER_LUNGE" in result.reasons


def test_looter_turn_two_still_fails_closed() -> None:
    result = assess_public_enemies(looter_state(turn=2, gold=84))
    assert result.allowed is False
    assert "looter_later_turn_unsupported_v1" in result.reasons


def test_looter_wrong_public_thievery_fails_closed() -> None:
    for turn in (0, 1):
        result = assess_public_enemies(looter_state(turn=turn, ascension=0, thievery=20, gold=84))
        assert result.allowed is False
        assert "looter_opening_thievery_mismatch_v1:expected_15" in result.reasons


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
