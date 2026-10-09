from __future__ import annotations

from roguelike_ai.sts1_phase3.simulator import _apply_last_potion_reserve


class _Action:
    def __init__(self, action_type: str, source_idx: int = -1, target_idx: int = -1) -> None:
        self.action_type = action_type
        self.source_idx = source_idx
        self.target_idx = target_idx


def _apply(
    recommended: _Action,
    legal: list[_Action],
    *,
    floor: int = 49,
    hp: int | None = 20,
    block: int | None = 0,
    incoming: int | None = 10,
) -> tuple[object, bool]:
    return _apply_last_potion_reserve(
        recommended,
        legal,
        hand=[],
        floor=floor,
        reserve_until_floor=50,
        player_hp=hp,
        player_block=block,
        incoming_damage=incoming,
    )


def test_reserves_only_usable_potion_and_selects_a_legal_card_fallback() -> None:
    potion = _Action("POTION", source_idx=0)
    card = _Action("CARD", source_idx=0)

    selected, overridden = _apply(potion, [potion, card])

    assert selected is card
    assert overridden is True


def test_does_not_replace_last_potion_with_an_empty_end_turn() -> None:
    potion = _Action("POTION", source_idx=0)
    end_turn = _Action("END_TURN")

    selected, overridden = _apply(potion, [potion, end_turn])

    assert selected is potion
    assert overridden is False


def test_allows_last_potion_when_visible_incoming_damage_is_lethal() -> None:
    potion = _Action("POTION", source_idx=0)
    card = _Action("CARD", source_idx=0)

    selected, overridden = _apply(potion, [potion, card], hp=10, incoming=10)

    assert selected is potion
    assert overridden is False


def test_deduplicates_target_actions_for_the_same_potion_slot() -> None:
    potion_target_a = _Action("POTION", source_idx=0, target_idx=0)
    potion_target_b = _Action("POTION", source_idx=0, target_idx=1)
    card = _Action("CARD", source_idx=0)

    selected, overridden = _apply(
        potion_target_a,
        [potion_target_a, potion_target_b, card],
    )

    assert selected is card
    assert overridden is True


def test_does_not_reserve_when_multiple_usable_potion_slots_remain() -> None:
    potion_a = _Action("POTION", source_idx=0)
    potion_b = _Action("POTION", source_idx=1)
    card = _Action("CARD", source_idx=0)

    selected, overridden = _apply(potion_a, [potion_a, potion_b, card])

    assert selected is potion_a
    assert overridden is False


def test_does_not_reserve_at_or_after_the_configured_floor() -> None:
    potion = _Action("POTION", source_idx=0)
    card = _Action("CARD", source_idx=0)

    selected, overridden = _apply(potion, [potion, card], floor=50)

    assert selected is potion
    assert overridden is False


def test_does_not_block_discarding_a_potion_or_unknown_threat_states() -> None:
    discard = _Action("POTION", source_idx=0, target_idx=6)
    card = _Action("CARD", source_idx=0)
    selected_discard, discard_overridden = _apply(discard, [discard, card])

    potion = _Action("POTION", source_idx=0)
    selected_unknown, unknown_overridden = _apply(
        potion,
        [potion, card],
        incoming=None,
    )

    assert selected_discard is discard
    assert discard_overridden is False
    assert selected_unknown is potion
    assert unknown_overridden is False
