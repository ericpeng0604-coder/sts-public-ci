from __future__ import annotations

from types import SimpleNamespace

from roguelike_ai.sts1_phase3.simulator import _apply_lethal_intent_defend_rescue


class _Card:
    def __init__(self, card_id: str, upgraded: bool | None = False) -> None:
        self.id = card_id
        self.upgraded = upgraded


class _Action:
    action_type = "CARD"

    def __init__(self, source_idx: int) -> None:
        self.source_idx = source_idx
        self.target_idx = -1


class _Enemy:
    cur_hp = 20
    alive = True

    def __init__(self, damage: int | None, hits: int = 1) -> None:
        self.damage = damage
        self.hits = hits

    def intent_damage(self, _battle: object) -> object:
        if self.damage is None:
            raise RuntimeError("unknown intent")
        return SimpleNamespace(damage=self.damage, attack_count=self.hits)


def _battle(*, hp: int = 8, block: int = 2, damage: int | None = 16) -> object:
    return SimpleNamespace(
        player=SimpleNamespace(cur_hp=hp, block=block),
        monsters=[_Enemy(damage)],
    )


def test_h16_selects_strongest_legal_defend_that_closes_visible_deficit() -> None:
    hand = [_Card("DEFEND_R"), _Card("DEFEND_G", upgraded=True), _Card("BASH")]
    recommended = _Action(2)
    selected, overridden, reason, detail = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1), _Action(2)],
        hand=hand,
        battle=_battle(hp=8, block=2, damage=16),
    )

    assert overridden is True
    assert reason == "legal_defend_closes_visible_lethal_deficit"
    assert selected is not recommended
    assert selected.source_idx == 1
    assert detail["projected_deficit"] == 6
    assert detail["selected_defend_base_block"] == 8
    assert detail["selected_defend_hand_index"] == 2


def test_h16_tie_breaks_by_lowest_hand_index_not_legal_action_order() -> None:
    hand = [_Card("DEFEND_R", upgraded=True), _Card("DEFEND_G", upgraded=True), _Card("BASH")]
    recommended = _Action(2)
    selected, overridden, _, detail = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(1), _Action(0), _Action(2)],
        hand=hand,
        battle=_battle(),
    )

    assert overridden is True
    assert selected.source_idx == 0
    assert detail["selected_defend_hand_index"] == 1


def test_h16_keeps_g7_action_when_intent_is_not_lethal() -> None:
    hand = [_Card("DEFEND_R"), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(hp=8, block=2, damage=9),
    )

    assert selected is recommended
    assert overridden is False
    assert reason == "visible_intent_not_lethal"


def test_h16_keeps_g7_defend_recommendation() -> None:
    hand = [_Card("DEFEND_R"), _Card("BASH")]
    recommended = _Action(0)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(),
    )

    assert selected is recommended
    assert overridden is False
    assert reason == "g7_already_selected_defend"


def test_h16_fails_closed_when_defend_cannot_close_deficit_or_identity_is_unknown() -> None:
    hand = [_Card("DEFEND_R"), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(hp=8, block=0, damage=20),
    )
    assert selected is recommended
    assert overridden is False
    assert reason == "defend_block_does_not_close_deficit"

    unknown_upgrade = [_Card("DEFEND_R", upgraded=None), _Card("BASH")]
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=unknown_upgrade,
        battle=_battle(),
    )
    assert selected is recommended
    assert overridden is False
    assert reason == "defend_upgrade_status_unknown"


def test_h16_fails_closed_when_visible_intent_is_unavailable() -> None:
    hand = [_Card("DEFEND_R"), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(damage=None),
    )

    assert selected is recommended
    assert overridden is False
    assert reason == "unknown_attack_intent"
