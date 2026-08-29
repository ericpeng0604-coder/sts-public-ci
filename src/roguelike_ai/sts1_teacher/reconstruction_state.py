"""Attach reconstruction capability evidence without changing policy identity."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .anger_reconstruction import assess_public_anger_midturn
from .card_reconstruction import assess_public_cards
from .composition_reconstruction import assess_public_anger_pommel_composition
from .contract import DecisionContext
from .enemy_reconstruction import assess_public_enemies
from .player_reconstruction import assess_public_player
from .reconstruction import PUBLIC_RECONSTRUCTION_SCHEMA
from .run_reconstruction import assess_public_run_state


def attach_reconstruction_capabilities(
    public_state: Mapping[str, Any],
    *,
    run_state: Mapping[str, Any] | None = None,
    reconstruction_aux: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = deepcopy(dict(public_state))
    source_marker = (run_state or {}).get("reconstruction", {})
    source = dict(source_marker) if isinstance(source_marker, Mapping) else {}

    player_admission = assess_public_player(result, reconstruction_aux=reconstruction_aux)
    card_admission = assess_public_cards(result)
    anger_admission = assess_public_anger_midturn(result, reconstruction_aux)
    anger_pommel_admission = assess_public_anger_pommel_composition(result, reconstruction_aux)
    enemy_admission = assess_public_enemies(result)
    run_admission = assess_public_run_state(result)

    # Rich generated-card slices stay deliberately isolated from the established
    # V1 player/card admission paths. Their proof comes from current public
    # piles plus bounded turn-local command counters. This keeps every frozen
    # V1/V2 slice unchanged while allowing only explicitly audited rich states.
    anger_override = anger_admission.allowed
    anger_pommel_override = anger_pommel_admission.allowed
    rich_override = anger_override or anger_pommel_override
    player_complete = player_admission.allowed or rich_override
    card_complete = card_admission.allowed or rich_override

    marker: dict[str, Any] = {
        "schema_version": PUBLIC_RECONSTRUCTION_SCHEMA,
        "public_player_state_complete": player_complete,
        "public_card_instance_state_complete": card_complete,
        "public_relic_state_complete": bool(source.get("public_relic_state_complete")) and run_admission.relics_allowed,
        "public_potion_state_complete": bool(source.get("public_potion_state_complete")) and run_admission.potions_allowed,
        "public_enemy_state_complete": enemy_admission.allowed,
        "player_admission_reasons": [] if rich_override else list(player_admission.reasons),
        "card_admission_reasons": [] if rich_override else list(card_admission.reasons),
        "anger_midturn_complete": anger_override,
        "anger_admission_reasons": list(anger_admission.reasons),
        "anger_pommel_composition_complete": anger_pommel_override,
        "anger_pommel_composition_admission_reasons": list(anger_pommel_admission.reasons),
        "card_count": card_admission.card_count,
        "enemy_admission_reasons": list(enemy_admission.reasons),
        "enemy_count": enemy_admission.enemy_count,
        "run_admission_reasons": list(run_admission.reasons),
    }
    result["reconstruction"] = marker

    before = DecisionContext.from_public_state(public_state)
    after = DecisionContext.from_public_state(result)
    if before.decision_signature != after.decision_signature:
        raise RuntimeError("reconstruction_metadata_changed_policy_identity")
    return result


__all__ = ["attach_reconstruction_capabilities"]
