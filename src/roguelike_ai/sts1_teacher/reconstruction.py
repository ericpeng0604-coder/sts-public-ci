"""Fail-closed admission gate for STS1 public-state rollout reconstruction."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contract import DecisionContext, PublicStateContractError

PUBLIC_RECONSTRUCTION_SCHEMA = "sts1-public-reconstruction-v1"
_REQUIRED_CAPABILITIES = (
    "public_player_state_complete",
    "public_card_instance_state_complete",
    "public_relic_state_complete",
    "public_potion_state_complete",
    "public_enemy_state_complete",
)
_REQUIRED_FIELDS = (
    "hp", "max_hp", "block", "energy",
    "hand", "draw_pile", "discard_pile", "exhaust_pile",
    "powers", "enemies", "turn", "combat_active",
    "relics", "potions", "gold", "floor", "act", "character",
    "ascension_level", "room", "legal_actions",
)


@dataclass(frozen=True)
class ReconstructionAdmission:
    allowed: bool
    reasons: tuple[str, ...]
    decision_signature: str | None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def assess_public_reconstruction(state: Mapping[str, Any]) -> ReconstructionAdmission:
    reasons: list[str] = []
    for field in _REQUIRED_FIELDS:
        if field not in state:
            reasons.append(f"missing_public_field:{field}")

    marker = state.get("reconstruction")
    if not isinstance(marker, Mapping):
        reasons.append("missing_reconstruction_capability_marker")
    else:
        if marker.get("schema_version") != PUBLIC_RECONSTRUCTION_SCHEMA:
            reasons.append("unsupported_reconstruction_schema")
        for capability in _REQUIRED_CAPABILITIES:
            if marker.get(capability) is not True:
                reasons.append(f"capability_not_proven:{capability}")

    if state.get("combat_active") is not True:
        reasons.append("combat_not_active")
    if str(state.get("room", "")).upper() != "COMBAT":
        reasons.append("room_not_combat")

    for field in ("hand", "draw_pile", "discard_pile", "exhaust_pile", "powers", "enemies", "relics", "potions", "legal_actions"):
        if field in state and not _is_sequence(state[field]):
            reasons.append(f"public_field_not_sequence:{field}")

    signature: str | None = None
    try:
        context = DecisionContext.from_public_state(state)
        signature = context.decision_signature
    except PublicStateContractError as exc:
        reasons.append(f"public_state_contract:{exc}")

    return ReconstructionAdmission(
        allowed=not reasons,
        reasons=tuple(sorted(set(reasons))),
        decision_signature=signature,
    )


def require_public_reconstruction(state: Mapping[str, Any]) -> DecisionContext:
    admission = assess_public_reconstruction(state)
    if not admission.allowed:
        raise PublicStateContractError("reconstruction_not_admitted:" + ",".join(admission.reasons))
    return DecisionContext.from_public_state(state)


__all__ = [
    "PUBLIC_RECONSTRUCTION_SCHEMA",
    "ReconstructionAdmission",
    "assess_public_reconstruction",
    "require_public_reconstruction",
]
