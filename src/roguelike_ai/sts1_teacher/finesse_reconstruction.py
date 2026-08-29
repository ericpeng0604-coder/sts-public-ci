"""Narrow public-state admission for exactly one normal Finesse on Jaw Worm turn 1.

Finesse is deliberately handled as an additive slice rather than widening the
existing Shrug path.  The bounded CommunicationMod auxiliary trace proves only
turn-local counters; card identity, upgrade count, energy, block and pile shape
must all be visible in the public snapshot.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Sequence

from .player_reconstruction import PublicPlayerAdmission, assess_public_player

_AUX_SCHEMA = "sts1-public-reconstruction-aux-v1"
_AUX_SOURCE = "communicationmod_command_trace_v1"
_CARD_ID_ALIASES = {"STRIKE_R": "STRIKE_RED", "DEFEND_R": "DEFEND_RED"}
_EXPECTED_COUNTS = Counter({"STRIKE_RED": 5, "DEFEND_RED": 4, "BASH": 1, "FINESSE": 1})


def _norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def _card_id(value: Any) -> str:
    normalized = _norm(value)
    return _CARD_ID_ALIASES.get(normalized, normalized)


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return None


def is_finesse_slice_candidate(state: Mapping[str, Any], aux: Mapping[str, Any] | None) -> bool:
    if not isinstance(aux, Mapping) or state.get("turn") != 1:
        return False
    if aux.get("attacks_played_this_turn") != 0 or aux.get("skills_played_this_turn") != 1:
        return False
    for pile_name in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        pile = _sequence(state.get(pile_name))
        if pile is None:
            return False
        for card in pile:
            if isinstance(card, Mapping) and _card_id(card.get("id")) == "FINESSE":
                return True
    return False


def assess_public_finesse_player(
    state: Mapping[str, Any],
    *,
    reconstruction_aux: Mapping[str, Any],
) -> PublicPlayerAdmission:
    reasons: list[str] = []

    expected_aux = {
        "schema_version": _AUX_SCHEMA,
        "source": _AUX_SOURCE,
        "turn": 1,
        "complete": True,
        "cards_played_this_turn": 1,
        "attacks_played_this_turn": 0,
        "skills_played_this_turn": 1,
        "cards_discarded_this_turn": 0,
    }
    for key, expected in expected_aux.items():
        value = reconstruction_aux.get(key)
        if type(expected) is int:
            if type(value) is not int or value != expected:
                reasons.append(f"finesse_aux_{key}_mismatch:{value!r}!={expected}")
        elif value != expected:
            reasons.append(f"finesse_aux_{key}_mismatch:{value!r}!={expected!r}")

    hand = _sequence(state.get("hand"))
    draw = _sequence(state.get("draw_pile"))
    discard = _sequence(state.get("discard_pile"))
    exhaust = _sequence(state.get("exhaust_pile"))
    if hand is None or draw is None or discard is None or exhaust is None:
        reasons.append("finesse_turn1_card_piles_not_sequences")
        return PublicPlayerAdmission(False, tuple(sorted(set(reasons))))

    if len(hand) != 5 or len(draw) != 0 or len(discard) != 6 or len(exhaust) != 0:
        reasons.append(
            f"finesse_turn1_pile_shape:hand={len(hand)}:draw={len(draw)}:discard={len(discard)}:exhaust={len(exhaust)}"
        )

    ids: list[str] = []
    finesse_cards: list[Mapping[str, Any]] = []
    for index, card in enumerate([*hand, *draw, *discard, *exhaust]):
        if not isinstance(card, Mapping):
            reasons.append(f"finesse_turn1_card_not_mapping:{index}")
            continue
        card_id = _card_id(card.get("id"))
        ids.append(card_id)
        upgrades = card.get("upgrades")
        if type(upgrades) is not int or upgrades != 0:
            reasons.append(f"finesse_turn1_requires_unupgraded_cards:{index}:{upgrades!r}")
        if card_id == "FINESSE":
            finesse_cards.append(card)
            if card.get("cost") != 0 or isinstance(card.get("cost"), bool):
                reasons.append(f"finesse_turn1_cost_mismatch:{card.get('cost')!r}")

    if Counter(ids) != _EXPECTED_COUNTS:
        reasons.append("finesse_turn1_deck_composition_mismatch")
    if len(finesse_cards) != 1:
        reasons.append(f"finesse_turn1_identity_count:{len(finesse_cards)}")
    if sum(1 for card in discard if isinstance(card, Mapping) and _card_id(card.get("id")) == "FINESSE") != 1:
        reasons.append("finesse_turn1_finesse_not_in_discard")

    if state.get("energy") != 3 or isinstance(state.get("energy"), bool):
        reasons.append(f"finesse_turn1_energy_mismatch:{state.get('energy')!r}")
    if state.get("block") != 2 or isinstance(state.get("block"), bool):
        reasons.append(f"finesse_turn1_block_mismatch:{state.get('block')!r}")

    if reasons:
        return PublicPlayerAdmission(False, tuple(sorted(set(reasons))))

    # Reuse the already-audited Shrug structural gate as a shadow proof for
    # global player fields, enemy shape, powers and auxiliary semantics.  Only
    # public data is transformed, and the shadow never enters policy state.
    shadow = deepcopy(dict(state))
    for pile_name in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        for card in shadow[pile_name]:
            if _card_id(card.get("id")) == "FINESSE":
                card["id"] = "Shrug It Off"
                card["name"] = "Shrug It Off"
                card["type"] = "SKILL"
                card["cost"] = 1
                card["upgrades"] = 0
    shadow["energy"] = 2
    shadow["block"] = 8
    base = assess_public_player(shadow, reconstruction_aux=reconstruction_aux)
    if not base.allowed:
        return PublicPlayerAdmission(
            False,
            tuple(sorted(f"finesse_base:{reason}" for reason in base.reasons)),
        )

    return PublicPlayerAdmission(True, ())


__all__ = ["assess_public_finesse_player", "is_finesse_slice_candidate"]
