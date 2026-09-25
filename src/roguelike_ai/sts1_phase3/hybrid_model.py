"""Canonical STS1 Hybrid Model bundle.

A deployable/evaluable STS1 model is not the Student checkpoint alone.
It is a bundle of:
- one Student checkpoint,
- MCTS runtime budgets,
- Student-on-MCTS-tie fusion,
- ArmG non-combat policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .base_policy import STS1_BASE_POLICY_ID


HYBRID_MODEL_SCHEMA_VERSION = "sts1-hybrid-model-v1"
HYBRID_RUNTIME_ID = "student+mcts1000+2000+armg-v1"
DEFAULT_HYBRID_MCTS_BUDGETS = (1000, 2000)
HYBRID_FUSION_RULE = "mcts_majority_student_breaks_mcts_ties"


class HybridModelError(ValueError):
    """Hybrid model bundle identity is incomplete or invalid."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HybridModelSpec:
    runtime_id: str
    student_checkpoint_sha256: str
    mcts_budgets: tuple[int, ...]
    fusion_rule: str
    noncombat_policy: str
    base_policy_id: str
    loop_round: int | None = None

    def __post_init__(self) -> None:
        if self.runtime_id != HYBRID_RUNTIME_ID:
            raise HybridModelError("hybrid runtime id drift")
        if len(self.student_checkpoint_sha256) != 64:
            raise HybridModelError("Student checkpoint SHA-256 is invalid")
        if tuple(self.mcts_budgets) != DEFAULT_HYBRID_MCTS_BUDGETS:
            raise HybridModelError("hybrid MCTS budgets drift")
        if self.fusion_rule != HYBRID_FUSION_RULE:
            raise HybridModelError("hybrid fusion rule drift")
        if self.noncombat_policy != "armg":
            raise HybridModelError("hybrid non-combat policy must be ArmG")
        if self.base_policy_id != STS1_BASE_POLICY_ID:
            raise HybridModelError("hybrid base policy identity drift")
        if self.loop_round is not None and self.loop_round < 0:
            raise HybridModelError("loop_round must be non-negative")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": HYBRID_MODEL_SCHEMA_VERSION,
            **asdict(self),
            "mcts_budgets": list(self.mcts_budgets),
            "model_semantics": "STUDENT_CHECKPOINT_PLUS_MCTS_PLUS_ARMG",
        }


def build_hybrid_model_spec(
    student_checkpoint: Path,
    *,
    loop_round: int | None = None,
) -> HybridModelSpec:
    if not student_checkpoint.is_file():
        raise HybridModelError(f"Student checkpoint missing: {student_checkpoint}")
    return HybridModelSpec(
        runtime_id=HYBRID_RUNTIME_ID,
        student_checkpoint_sha256=_file_sha256(student_checkpoint),
        mcts_budgets=DEFAULT_HYBRID_MCTS_BUDGETS,
        fusion_rule=HYBRID_FUSION_RULE,
        noncombat_policy="armg",
        base_policy_id=STS1_BASE_POLICY_ID,
        loop_round=loop_round,
    )


def write_hybrid_model_manifest(
    student_checkpoint: Path,
    manifest_path: Path,
    *,
    loop_round: int | None = None,
) -> dict[str, Any]:
    spec = build_hybrid_model_spec(student_checkpoint, loop_round=loop_round)
    payload = spec.manifest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "DEFAULT_HYBRID_MCTS_BUDGETS",
    "HYBRID_FUSION_RULE",
    "HYBRID_MODEL_SCHEMA_VERSION",
    "HYBRID_RUNTIME_ID",
    "HybridModelError",
    "HybridModelSpec",
    "build_hybrid_model_spec",
    "write_hybrid_model_manifest",
]
