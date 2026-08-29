"""Attach reconstruction capability evidence without changing policy identity."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .card_reconstruction import assess_public_cards
from .contract import DecisionContext
from .enemy_reconstruction import assess_public_enemies
from .finesse_reconstruction import assess_public_finesse_player, is_finesse_slice_candidate
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
    if (
        not player_admission.allowed
        and isinstance(reconstruction_aux, Mapping)
        and is_finesse_slice_candidate(result, reconstruction_aux)
    ):
        player_admission = assess_public_finesse_player(
            result,
            reconstruction_aux=reconstruction_aux,
        )
    card_admission = assess_public_cards(result)
    enemy_admission = assess_public_enemies(result)
    run_admission = assess_public_run_state(result)

    marker: dict[str, Any] = {
        "schema_version": PUBLIC_RECONSTRUCTION_SCHEMA,
        "public_player_state_complete": player_admission.allowed,
        "public_card_instance_state_complete": card_admission.allowed,
        "public_relic_state_complete": bool(source.get("public_relic_state_complete")) and run_admission.relics_allowed,
        "public_potion_state_complete": bool(source.get("public_potion_state_complete")) and run_admission.potions_allowed,
        "public_enemy_state_complete": enemy_admission.allowed,
        "player_admission_reasons": list(player_admission.reasons),
        "card_admission_reasons": list(card_admission.reasons),
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
