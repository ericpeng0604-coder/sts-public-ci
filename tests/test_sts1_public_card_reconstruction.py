from roguelike_ai.sts1_teacher.card_reconstruction import assess_public_cards


def base_state() -> dict:
    return {
        "hand": [
            {"position": 1, "id": "STRIKE_RED", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0},
            {"position": 2, "id": "BASH", "name": "Bash", "type": "ATTACK", "cost": 2, "upgrades": 1},
        ],
        "draw_pile": [
            {"id": "DEFEND_RED", "name": "Defend", "type": "SKILL", "cost": 1, "upgrades": 0},
        ],
        "discard_pile": [],
        "exhaust_pile": [],
    }


def test_basic_public_cards_are_admitted() -> None:
    result = assess_public_cards(base_state())
    assert result.allowed is True
    assert result.reasons == ()
    assert result.card_count == 3


def test_any_card_outside_v1_slice_fails_closed() -> None:
    for card_id in ("RAMPAGE", "ANGER", "POMMEL_STRIKE", "SEARING_BLOW"):
        state = base_state()
        state["hand"][0]["id"] = card_id
        result = assess_public_cards(state)
        assert result.allowed is False
        assert f"card_unsupported_v1:{card_id}" in result.reasons


def test_missing_public_cost_fails_closed() -> None:
    state = base_state()
    state["hand"][0].pop("cost")
    result = assess_public_cards(state)
    assert result.allowed is False
    assert "invalid_card_cost:hand[0]" in result.reasons


def test_temporary_cost_change_fails_closed() -> None:
    state = base_state()
    state["hand"][0]["cost"] = 0
    result = assess_public_cards(state)
    assert result.allowed is False
    assert "temporary_or_unknown_card_cost_unsupported:hand[0]:STRIKE_RED:0!=1" in result.reasons


def test_missing_upgrade_count_fails_closed() -> None:
    state = base_state()
    state["draw_pile"][0].pop("upgrades")
    result = assess_public_cards(state)
    assert result.allowed is False
    assert "invalid_card_upgrades:draw_pile[0]" in result.reasons


def test_card_instance_identity_is_never_required() -> None:
    state = base_state()
    result = assess_public_cards(state)
    assert result.allowed is True
    assert all("unique" not in key.lower() and "uuid" not in key.lower() for pile in state.values() for card in pile for key in card)
