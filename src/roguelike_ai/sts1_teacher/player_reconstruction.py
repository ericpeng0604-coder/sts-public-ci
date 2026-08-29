"""Narrow V1 admission for public player combat state.

The pinned sts_lightspeed BattleContext is zero-based: ``turn == 0`` is the
first player turn. V1 still admits that opening surface directly. A second,
strictly audited slice admits only the *start* of Jaw Worm turn 1, where the
starter-only public state proves that per-turn player counters are still zero.
Any other later-turn state continues to fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_SUPPORTED_PLAYER_POWERS_V1 = frozenset({"STRENGTH", "DEXTERITY", "FOCUS", "ARTIFACT"})
_SUPPORTED_STARTER_CARDS_V1 = frozenset({"STRIKE_RED", "DEFEND_RED", "BASH"})


def _norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return None


def _jaw_worm_turn_one_boundary_reasons(state: Mapping[str, Any]) -> list[str]:
    """Prove the tiny turn-1 start boundary from public state only.

    Under the frozen V1 run/card slice there are exactly ten non-exhausting
    starter cards, no energy gain/draw/retain cards, Burning Blood only, and no
    usable potions. On turn 1 the first five cards have all reached discard and
    the untouched second hand is the remaining five cards. Thus the exact public
    boundary is energy 3, block 0, hand 5, draw 0, discard 5, exhaust 0.
    """

    reasons: list[str] = []
    enemies = _sequence(state.get("enemies"))
    if enemies is None or len(enemies) != 1 or not isinstance(enemies[0], Mapping):
        reasons.append("turn1_boundary_requires_single_enemy")
    elif _norm(enemies[0].get("name")) != "JAW_WORM":
        reasons.append("turn1_boundary_requires_jaw_worm")

    if state.get("energy") != 3:
        reasons.append(f"turn1_boundary_energy_not_fresh:{state.get('energy')}")
    if state.get("block") != 0:
        reasons.append(f"turn1_boundary_block_not_reset:{state.get('block')}")

    hand = _sequence(state.get("hand"))
    draw = _sequence(state.get("draw_pile"))
    discard = _sequence(state.get("discard_pile"))
    exhaust = _sequence(state.get("exhaust_pile"))
    if hand is None or draw is None or discard is None or exhaust is None:
        reasons.append("turn1_boundary_card_piles_not_sequences")
        return reasons

    if len(hand) != 5:
        reasons.append(f"turn1_boundary_hand_not_fresh:{len(hand)}")
    if len(draw) != 0:
        reasons.append(f"turn1_boundary_draw_not_empty:{len(draw)}")
    if len(discard) != 5:
        reasons.append(f"turn1_boundary_discard_not_complete:{len(discard)}")
    if len(exhaust) != 0:
        reasons.append(f"turn1_boundary_exhaust_not_empty:{len(exhaust)}")

    all_cards = [*hand, *draw, *discard, *exhaust]
    if len(all_cards) != 10:
        reasons.append(f"turn1_boundary_card_count_not_starter:{len(all_cards)}")
    for index, card in enumerate(all_cards):
        if not isinstance(card, Mapping):
            reasons.append(f"turn1_boundary_card_not_mapping:{index}")
            continue
        card_id = _norm(card.get("id"))
        if card_id not in _SUPPORTED_STARTER_CARDS_V1:
            reasons.append(f"turn1_boundary_nonstarter_card:{card_id or 'MISSING'}")

    powers = _sequence(state.get("powers"))
    if powers is None:
        reasons.append("turn1_boundary_powers_not_sequence")
    elif len(powers) != 0:
        # Keeping this boundary power-free avoids reconstructing justApplied and
        # other history-sensitive player-power metadata in this first extension.
        reasons.append("turn1_boundary_requires_no_player_powers")

    return reasons


@dataclass(frozen=True)
class PublicPlayerAdmission:
    allowed: bool
    reasons: tuple[str, ...]


def assess_public_player(state: Mapping[str, Any]) -> PublicPlayerAdmission:
    reasons: list[str] = []
    for field in ("hp", "max_hp", "block", "energy", "gold", "turn", "floor", "ascension_level"):
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            reasons.append(f"invalid_player_scalar:{field}")

    turn = state.get("turn")
    if turn == 1:
        reasons.extend(_jaw_worm_turn_one_boundary_reasons(state))
    elif turn != 0:
        reasons.append(f"turn_unsupported_v1:{turn}")

    powers = state.get("powers")
    if not isinstance(powers, Sequence) or isinstance(powers, str | bytes | bytearray):
        reasons.append("player_powers_not_sequence")
    else:
        for index, power in enumerate(powers):
            if not isinstance(power, Mapping):
                reasons.append(f"player_power_not_mapping:{index}")
                continue
            name = _norm(power.get("name"))
            if name not in _SUPPORTED_PLAYER_POWERS_V1:
                reasons.append(f"player_power_unsupported_v1:{name or 'MISSING'}")
            amount = power.get("amount")
            if isinstance(amount, bool) or not isinstance(amount, int):
                reasons.append(f"invalid_player_power_amount:{index}")

    return PublicPlayerAdmission(not reasons, tuple(sorted(set(reasons))))


__all__ = ["PublicPlayerAdmission", "assess_public_player"]
