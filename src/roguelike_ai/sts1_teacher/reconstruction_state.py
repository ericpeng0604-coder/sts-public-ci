"""Attach reconstruction capability evidence without changing policy identity."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .card_reconstruction import assess_public_cards
from .contract import DecisionContext
from .reconstruction import PUBLIC_RECONSTRUCTION_SCHEMA


def attach_reconstruction_capabilities(
    public_state: Mapping[str, Any],
    *,
    run_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a public state with fail-closed reconstruction metadata.

    The top-level ``reconstruction`` field is intentionally outside the frozen
    policy field set.  ``DecisionContext`` still validates it for forbidden
    hidden keys, but does not include it in policy features or the decision
    signature.  This lets reconstruction get stricter without silently changing
    Teacher observations.
    """

    result = deepcopy(dict(public_state))
    source_marker = (run_state or {}).get("reconstruction", {})
    marker = dict(source_marker) if isinstance(source_marker, Mapping) else {}
    marker["schema_version"] = PUBLIC_RECONSTRUCTION_SCHEMA

    card_admission = assess_public_cards(result)
    marker["public_card_instance_state_complete"] = card_admission.allowed
    marker.setdefault("public_relic_state_complete", False)
    marker.setdefault("public_potion_state_complete", False)
    marker.setdefault("public_enemy_state_complete", False)
    marker["card_admission_reasons"] = list(card_admission.reasons)
    marker["card_count"] = card_admission.card_count
    result["reconstruction"] = marker

    # Prove reconstruction metadata is policy-identity neutral.
    before = DecisionContext.from_public_state(public_state)
    after = DecisionContext.from_public_state(result)
    if before.decision_signature != after.decision_signature:
        raise RuntimeError("reconstruction_metadata_changed_policy_identity")
    return result


__all__ = ["attach_reconstruction_capabilities"]
