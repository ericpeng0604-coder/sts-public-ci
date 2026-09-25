"""Registered STS1 Phase-3 reference policies.

A reference policy is a reproducible decision system used for evaluation and
Teacher data collection. It is not automatically a Champion and does not need
to be a neural-network checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .frozen_student import ACTION_SCHEMA_VERSION, PUBLIC_STATE_SCHEMA_VERSION
from .protocol import FROZEN_SIMULATOR_SHA


REFERENCE_POLICY_SCHEMA_VERSION = "sts1-reference-policy-v1"
ARMG_MCTS_2000_REFERENCE_ID = "armg+mcts2000-reference-v1"


class ReferencePolicyError(ValueError):
    """Reference-policy identity or configuration is invalid."""


@dataclass(frozen=True)
class ReferencePolicySpec:
    policy_id: str
    role: str
    character: str
    ascension: int
    combat_policy: str
    noncombat_policy: str
    mcts_sims: int
    simulator_sha: str
    public_state_schema: str
    action_schema: str
    armg_repository: str
    armg_upstream_commit: str
    armg_weight_path: str
    armg_weight_git_blob_sha: str

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ReferencePolicyError("reference policy id must be non-empty")
        if self.role != "teacher_reference":
            raise ReferencePolicyError("reference policy role must be teacher_reference")
        if self.character != "IRONCLAD" or self.ascension != 0:
            raise ReferencePolicyError("current registered reference policy is A0 Ironclad only")
        if self.mcts_sims < 1:
            raise ReferencePolicyError("MCTS simulations must be positive")
        if self.simulator_sha != FROZEN_SIMULATOR_SHA:
            raise ReferencePolicyError("reference policy simulator identity drift")
        if self.public_state_schema != PUBLIC_STATE_SCHEMA_VERSION:
            raise ReferencePolicyError("reference policy public-state schema drift")
        if self.action_schema != ACTION_SCHEMA_VERSION:
            raise ReferencePolicyError("reference policy action schema drift")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": REFERENCE_POLICY_SCHEMA_VERSION,
            **asdict(self),
            "promotion_semantics": "REFERENCE_ONLY_NOT_CHAMPION",
        }


ARMG_MCTS_2000_REFERENCE_V1 = ReferencePolicySpec(
    policy_id=ARMG_MCTS_2000_REFERENCE_ID,
    role="teacher_reference",
    character="IRONCLAD",
    ascension=0,
    combat_policy="mcts",
    noncombat_policy="armg",
    mcts_sims=2000,
    simulator_sha=FROZEN_SIMULATOR_SHA,
    public_state_schema=PUBLIC_STATE_SCHEMA_VERSION,
    action_schema=ACTION_SCHEMA_VERSION,
    armg_repository="https://github.com/Jialeiv/sts-rl-agent",
    armg_upstream_commit="b20eb2cac2f52b22fbb6c79900c309b51ea0a1db",
    armg_weight_path="weights/armG_model_G128x128_15k.pt",
    armg_weight_git_blob_sha="6666fb690fcceb592a1aa8dc74fa27bfee4575f0",
)

REFERENCE_POLICIES = {
    ARMG_MCTS_2000_REFERENCE_ID: ARMG_MCTS_2000_REFERENCE_V1,
}


def get_reference_policy(policy_id: str) -> ReferencePolicySpec:
    try:
        return REFERENCE_POLICIES[policy_id]
    except KeyError as exc:
        raise ReferencePolicyError(f"unknown STS1 reference policy: {policy_id}") from exc


__all__ = [
    "ARMG_MCTS_2000_REFERENCE_ID",
    "ARMG_MCTS_2000_REFERENCE_V1",
    "REFERENCE_POLICIES",
    "REFERENCE_POLICY_SCHEMA_VERSION",
    "ReferencePolicyError",
    "ReferencePolicySpec",
    "get_reference_policy",
]
