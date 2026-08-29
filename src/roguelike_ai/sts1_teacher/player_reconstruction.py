"""Narrow V1 admission for public player combat state.

The pinned sts_lightspeed BattleContext is zero-based: ``turn == 0`` is the
first player turn. V1 admits that opening surface directly. A second, strictly
audited slice admits Jaw Worm turn 1 both at the fresh boundary and during the
same player turn, but only for the starter-only/Burning-Blood surface where the
per-turn counters are derivable from public hand, energy and block.
Any other later-turn state continues to fail closed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_SUPPORTED_PLAYER_POWERS_V1 = frozenset({"STRENGTH", "DEXTERITY", "FOCUS", "ARTIFACT"})
_SUPPORTED_STARTER_CARDS_V1 = frozenset({"STRIKE_RED", "DEFEND_RED", "BASH"})
_EXPECTED_STARTER_COUNTS_V1 = Counter({"STRIKE_RED": 5, "DEFEND_RED": 4, "BASH": 1})


def _norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return None


def _jaw_worm_turn_one_reasons(state: Mapping[str, Any]) -> list[str]:
    """Prove the tiny turn-1 surface from public state only.

    Under the frozen V1 slice there are exactly ten non-exhausting starter cards,
    no draw/energy/retain/discard cards, Burning Blood only and no usable potion.
    The second player turn therefore begins with five cards, three energy, zero
    block and five cards already in discard. During that same turn:

    * cards played = 5 - current hand size;
    * skills played = current block / 5 (starter Defend is the only block source);
    * attacks played = cards played - skills played;
    * energy spent is either one per played card, plus one extra iff Bash was
      among the current-turn plays.

    This makes the simulator bookkeeping derivable from the current public state
    without adding move history or hidden counters to the Teacher contract.
    """

    reasons: list[str] = []
    enemies = _sequence(state.get("enemies"))
    if enemies is None or len(enemies) != 1 or not isinstance(enemies[0], Mapping):
        reasons.append("turn1_requires_single_enemy")
    elif _norm(enemies[0].get("name")) != "JAW_WORM":
        reasons.append("turn1_requires_jaw_worm")

    hand = _sequence(state.get("hand"))
    draw = _sequence(state.get("draw_pile"))
    discard = _sequence(state.get("discard_pile"))
    exhaust = _sequence(state.get("exhaust_pile"))
    if hand is None or draw is None or discard is None or exhaust is None:
        reasons.append("turn1_card_piles_not_sequences")
        return reasons

    if len(draw) != 0:
        reasons.append(f"turn1_draw_not_empty:{len(draw)}")
    if len(exhaust) != 0:
        reasons.append(f"turn1_exhaust_not_empty:{len(exhaust)}")
    if not 2 <= len(hand) <= 5:
        reasons.append(f"turn1_hand_size_unreachable:{len(hand)}")
    if len(discard) != 10 - len(hand):
        reasons.append(f"turn1_discard_size_unreachable:{len(discard)}")

    all_cards = [*hand, *draw, *discard, *exhaust]
    ids: list[str] = []
    for index, card in enumerate(all_cards):
        if not isinstance(card, Mapping):
            reasons.append(f"turn1_card_not_mapping:{index}")
            continue
        card_id = _norm(card.get("id"))
        ids.append(card_id)
        if card_id not in _SUPPORTED_STARTER_CARDS_V1:
            reasons.append(f"turn1_nonstarter_card:{card_id or 'MISSING'}")
    if Counter(ids) != _EXPECTED_STARTER_COUNTS_V1:
        reasons.append("turn1_starter_composition_mismatch")

    powers = _sequence(state.get("powers"))
    if powers is None:
        reasons.append("turn1_powers_not_sequence")
    elif len(powers) != 0:
        reasons.append("turn1_requires_no_player_powers")

    energy = state.get("energy")
    block = state.get("block")
    if isinstance(energy, bool) or not isinstance(energy, int) or not 0 <= energy <= 3:
        reasons.append(f"turn1_energy_unreachable:{energy}")
        return reasons
    if isinstance(block, bool) or not isinstance(block, int) or block < 0 or block % 5 != 0:
        reasons.append(f"turn1_block_unreachable:{block}")
        return reasons

    played = 5 - len(hand)
    skills = block // 5
    attacks = played - skills
    spent = 3 - energy
    if skills < 0 or attacks < 0:
        reasons.append(f"turn1_counter_mix_unreachable:played={played}:skills={skills}")
        return reasons
    if spent not in {played, played + 1} or spent > 3:
        reasons.append(f"turn1_energy_spend_unreachable:played={played}:spent={spent}")
        return reasons

    discard_ids = Counter(
        _norm(card.get("id"))
        for card in discard
        if isinstance(card, Mapping)
    )
    if skills > discard_ids["DEFEND_RED"]:
        reasons.append("turn1_skill_history_not_publicly_reachable")
    if spent == played + 1:
        # The one extra energy can only be the unique Bash.  There must be at
        # least one current attack and Bash must be outside the current hand.
        if attacks < 1 or discard_ids["BASH"] < 1 or attacks - 1 > discard_ids["STRIKE_RED"]:
            reasons.append("turn1_bash_history_not_publicly_reachable")
    elif attacks > discard_ids["STRIKE_RED"]:
        # No extra energy means every current-turn attack must be a 1-cost Strike.
        reasons.append("turn1_attack_history_not_publicly_reachable")

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
        reasons.extend(_jaw_worm_turn_one_reasons(state))
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
