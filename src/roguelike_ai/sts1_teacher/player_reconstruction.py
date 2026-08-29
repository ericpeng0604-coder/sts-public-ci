"""Narrow V1 admission for public player combat state.

The pinned sts_lightspeed BattleContext is zero-based: ``turn == 0`` is the
first player turn. V1 admits that opening surface directly. The established
turn-1 starter-only slice remains unchanged when no reconstruction auxiliary
trace is supplied.

Draw-card expansions use a bounded CommunicationMod command trace only for
turn-local counters that cannot be recovered from one snapshot. The trace is
reconstruction metadata only; it is never merged into policy state or decision
identity. Audited draw slices are deliberately tiny: exactly one normal Pommel
Strike or exactly one normal Shrug It Off on Jaw Worm turn 1. Any other richer
midturn still fails closed.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_SUPPORTED_PLAYER_POWERS_V1 = frozenset({"STRENGTH", "DEXTERITY", "FOCUS", "ARTIFACT"})
_SUPPORTED_STARTER_CARDS_V1 = frozenset({"STRIKE_RED", "DEFEND_RED", "BASH"})
_EXPECTED_STARTER_COUNTS_V1 = Counter({"STRIKE_RED": 5, "DEFEND_RED": 4, "BASH": 1})
_EXPECTED_POMMEL_COUNTS_V1 = Counter({"STRIKE_RED": 5, "DEFEND_RED": 4, "BASH": 1, "POMMEL_STRIKE": 1})
_EXPECTED_SHRUG_COUNTS_V1 = Counter({"STRIKE_RED": 5, "DEFEND_RED": 4, "BASH": 1, "SHRUG_IT_OFF": 1})
_CARD_ID_ALIASES = {"STRIKE_R": "STRIKE_RED", "DEFEND_R": "DEFEND_RED"}
_AUX_SCHEMA = "sts1-public-reconstruction-aux-v1"
_AUX_SOURCE = "communicationmod_command_trace_v1"


def _norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def _card_id(value: Any) -> str:
    normalized = _norm(value)
    return _CARD_ID_ALIASES.get(normalized, normalized)


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return None


def _jaw_worm_turn_one_reasons(state: Mapping[str, Any]) -> list[str]:
    """Prove the established starter-only turn-1 surface from public state."""

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
        card_id = _card_id(card.get("id"))
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
        _card_id(card.get("id"))
        for card in discard
        if isinstance(card, Mapping)
    )
    if skills > discard_ids["DEFEND_RED"]:
        reasons.append("turn1_skill_history_not_publicly_reachable")
    if spent == played + 1:
        if attacks < 1 or discard_ids["BASH"] < 1 or attacks - 1 > discard_ids["STRIKE_RED"]:
            reasons.append("turn1_bash_history_not_publicly_reachable")
    elif attacks > discard_ids["STRIKE_RED"]:
        reasons.append("turn1_attack_history_not_publicly_reachable")

    return reasons


def _validate_aux(
    aux: Mapping[str, Any],
    *,
    prefix: str,
    attacks: int,
    skills: int,
) -> list[str]:
    reasons: list[str] = []
    if aux.get("schema_version") != _AUX_SCHEMA:
        reasons.append(f"{prefix}_aux_schema_mismatch")
    if aux.get("source") != _AUX_SOURCE:
        reasons.append(f"{prefix}_aux_source_mismatch")
    if aux.get("complete") is not True:
        reasons.append(f"{prefix}_aux_incomplete")

    expected_ints = {
        "turn": 1,
        "cards_played_this_turn": 1,
        "attacks_played_this_turn": attacks,
        "skills_played_this_turn": skills,
        "cards_discarded_this_turn": 0,
    }
    for key, expected in expected_ints.items():
        value = aux.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            reasons.append(f"{prefix}_aux_{key}_mismatch:{value!r}!={expected}")
    return reasons


def _draw_slice_common_reasons(
    state: Mapping[str, Any],
    *,
    prefix: str,
    expected_counts: Counter[str],
    played_card_id: str,
    expected_block: int,
) -> list[str]:
    reasons: list[str] = []
    enemies = _sequence(state.get("enemies"))
    if enemies is None or len(enemies) != 1 or not isinstance(enemies[0], Mapping):
        reasons.append(f"{prefix}_turn1_requires_single_enemy")
    elif _norm(enemies[0].get("name")) != "JAW_WORM":
        reasons.append(f"{prefix}_turn1_requires_jaw_worm")

    hand = _sequence(state.get("hand"))
    draw = _sequence(state.get("draw_pile"))
    discard = _sequence(state.get("discard_pile"))
    exhaust = _sequence(state.get("exhaust_pile"))
    if hand is None or draw is None or discard is None or exhaust is None:
        reasons.append(f"{prefix}_turn1_card_piles_not_sequences")
        return reasons

    if len(hand) != 5 or len(draw) != 0 or len(discard) != 6 or len(exhaust) != 0:
        reasons.append(
            f"{prefix}_turn1_pile_shape:hand={len(hand)}:draw={len(draw)}:discard={len(discard)}:exhaust={len(exhaust)}"
        )

    ids: list[str] = []
    for index, card in enumerate([*hand, *draw, *discard, *exhaust]):
        if not isinstance(card, Mapping):
            reasons.append(f"{prefix}_turn1_card_not_mapping:{index}")
            continue
        ids.append(_card_id(card.get("id")))
        upgrades = card.get("upgrades")
        if upgrades != 0 or isinstance(upgrades, bool):
            reasons.append(f"{prefix}_turn1_requires_unupgraded_cards:{index}:{upgrades!r}")
    if Counter(ids) != expected_counts:
        reasons.append(f"{prefix}_turn1_deck_composition_mismatch")
    if sum(1 for card in discard if isinstance(card, Mapping) and _card_id(card.get("id")) == played_card_id) != 1:
        reasons.append(f"{prefix}_turn1_played_card_not_in_discard")

    if state.get("energy") != 2 or isinstance(state.get("energy"), bool):
        reasons.append(f"{prefix}_turn1_energy_mismatch:{state.get('energy')!r}")
    if state.get("block") != expected_block or isinstance(state.get("block"), bool):
        reasons.append(f"{prefix}_turn1_block_mismatch:{state.get('block')!r}")

    powers = _sequence(state.get("powers"))
    if powers is None:
        reasons.append(f"{prefix}_turn1_powers_not_sequence")
    elif len(powers) != 0:
        reasons.append(f"{prefix}_turn1_requires_no_player_powers")
    return reasons


def _pommel_turn_one_reasons(state: Mapping[str, Any], aux: Mapping[str, Any]) -> list[str]:
    """Prove exactly one normal Pommel Strike play using public state + bounded trace."""

    reasons = _validate_aux(aux, prefix="pommel", attacks=1, skills=0)
    common = _draw_slice_common_reasons(
        state,
        prefix="pommel",
        expected_counts=_EXPECTED_POMMEL_COUNTS_V1,
        played_card_id="POMMEL_STRIKE",
        expected_block=0,
    )
    # Preserve the established externally-visible reason for this check.
    common = [
        "pommel_turn1_pommel_not_in_discard" if item == "pommel_turn1_played_card_not_in_discard" else item
        for item in common
    ]
    reasons.extend(common)
    return reasons


def _shrug_turn_one_reasons(state: Mapping[str, Any], aux: Mapping[str, Any]) -> list[str]:
    """Prove exactly one normal Shrug It Off play using public state + bounded trace."""

    reasons = _validate_aux(aux, prefix="shrug", attacks=0, skills=1)
    reasons.extend(
        _draw_slice_common_reasons(
            state,
            prefix="shrug",
            expected_counts=_EXPECTED_SHRUG_COUNTS_V1,
            played_card_id="SHRUG_IT_OFF",
            expected_block=8,
        )
    )
    return reasons


@dataclass(frozen=True)
class PublicPlayerAdmission:
    allowed: bool
    reasons: tuple[str, ...]


def assess_public_player(
    state: Mapping[str, Any],
    *,
    reconstruction_aux: Mapping[str, Any] | None = None,
) -> PublicPlayerAdmission:
    reasons: list[str] = []
    for field in ("hp", "max_hp", "block", "energy", "gold", "turn", "floor", "ascension_level"):
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            reasons.append(f"invalid_player_scalar:{field}")

    turn = state.get("turn")
    if turn == 1:
        if reconstruction_aux is None:
            reasons.extend(_jaw_worm_turn_one_reasons(state))
        elif not isinstance(reconstruction_aux, MappingABC):
            reasons.append("reconstruction_aux_not_mapping")
        else:
            attacks = reconstruction_aux.get("attacks_played_this_turn")
            skills = reconstruction_aux.get("skills_played_this_turn")
            if type(attacks) is int and type(skills) is int and attacks == 1 and skills == 0:
                reasons.extend(_pommel_turn_one_reasons(state, reconstruction_aux))
            elif type(attacks) is int and type(skills) is int and attacks == 0 and skills == 1:
                reasons.extend(_shrug_turn_one_reasons(state, reconstruction_aux))
            else:
                reasons.append(f"draw_aux_slice_unsupported:attacks={attacks!r}:skills={skills!r}")
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
