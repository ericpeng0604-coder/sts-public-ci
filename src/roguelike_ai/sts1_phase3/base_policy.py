"""Canonical STS1 project base policy.

The project base is the strongest reproducible decision system chosen as the
foundation for future Student learning. It may be a composite policy rather
than a single neural-network checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .reference_policy import (
    ARMG_MCTS_2000_REFERENCE_ID,
    ARMG_MCTS_2000_REFERENCE_V1,
)


BASE_POLICY_SCHEMA_VERSION = "sts1-base-policy-v1"
STS1_BASE_POLICY_ID = "armg+mcts2000-base-v1"


class BasePolicyError(ValueError):
    """Base-policy identity or configuration is invalid."""


@dataclass(frozen=True)
class BasePolicySpec:
    policy_id: str
    source_reference_policy_id: str
    role: str
    combat_policy: str
    noncombat_policy: str
    mcts_sims: int
    character: str
    ascension: int

    def __post_init__(self) -> None:
        if self.role != "project_base_policy":
            raise BasePolicyError("base policy role must be project_base_policy")
        if self.source_reference_policy_id != ARMG_MCTS_2000_REFERENCE_ID:
            raise BasePolicyError("base policy source reference identity drift")
        if self.combat_policy != "mcts" or self.mcts_sims != 2000:
            raise BasePolicyError("base combat policy must be MCTS 2000")
        if self.noncombat_policy != "armg":
            raise BasePolicyError("base noncombat policy must be ArmG")
        if self.character != "IRONCLAD" or self.ascension != 0:
            raise BasePolicyError("current STS1 base policy is A0 Ironclad only")

    def manifest(self) -> dict[str, Any]:
        reference = ARMG_MCTS_2000_REFERENCE_V1.manifest()
        return {
            "schema_version": BASE_POLICY_SCHEMA_VERSION,
            **asdict(self),
            "reference_policy": reference,
            "foundation_semantics": (
                "PRIMARY_PROJECT_BASE; frozen Student v0 remains a numerical "
                "network anchor and reproducibility baseline only"
            ),
        }


STS1_BASE_POLICY = BasePolicySpec(
    policy_id=STS1_BASE_POLICY_ID,
    source_reference_policy_id=ARMG_MCTS_2000_REFERENCE_ID,
    role="project_base_policy",
    combat_policy="mcts",
    noncombat_policy="armg",
    mcts_sims=2000,
    character="IRONCLAD",
    ascension=0,
)


def get_base_policy() -> BasePolicySpec:
    return STS1_BASE_POLICY


__all__ = [
    "BASE_POLICY_SCHEMA_VERSION",
    "BasePolicyError",
    "BasePolicySpec",
    "STS1_BASE_POLICY",
    "STS1_BASE_POLICY_ID",
    "get_base_policy",
]
