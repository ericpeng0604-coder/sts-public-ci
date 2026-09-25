"""Fail-closed bridge from frozen STS1 Phase-1 Search evidence to Phase-2 records.

This importer never consumes diagnostic oracle scores, simple/random baselines,
or hidden simulator state. It re-derives the formal DecisionContext identity from
the public state, requires a fully resolved Search result, and preserves the
entire Search tie set. A deterministic canonical action-id tie-break exists only
because the Phase-2 record schema requires one selected action; downstream
training must keep the tie set available rather than pretending the tie is unique.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .dataset import (
    ProducerProvenance,
    STS1DatasetError,
    decision_signature_for_public_state,
    legal_action_ids_for_public_state,
    make_decision_record,
    sha256_json,
)


FORMAL_TEACHER_REPOSITORY = "ericpeng0604-coder/sts-public-ci"
FORMAL_TEACHER_SHA = "cc4d2c2dc13f8f06cbea086de4c4e0f359ea1b1a"
FORMAL_SIMULATOR_SHA = "7476a81954020087da31d41d16fddf475746ec2d"
FORMAL_SEARCH_CONFIG = {
    "samples_per_semantic_action": 8,
    "rollout_budget": 256,
    "node_budget": 16384,
    "max_depth": 30,
    "timeout_ms": 60_000,
    "tie_tolerance": 1e-9,
    "sampling_seed": 20260830,
}
TIE_SELECTION_POLICY = "canonical-min-search-tie-action-id-v1"
FORMAL_CAPTURE_CONFIG = {
    "teacher_repository": FORMAL_TEACHER_REPOSITORY,
    "teacher_sha": FORMAL_TEACHER_SHA,
    "search_config": FORMAL_SEARCH_CONFIG,
    "selected_action_policy": TIE_SELECTION_POLICY,
    "requires_resolved_search": True,
    "oracle_fields_are_policy_inputs": False,
}
FORMAL_TEACHER_CONFIG_HASH = sha256_json(FORMAL_CAPTURE_CONFIG)


class TeacherCaptureError(STS1DatasetError):
    """Raised when frozen Teacher evidence cannot become a trusted Phase-2 label."""


def formal_teacher_provenance(
    *,
    teacher_id: str,
    capture_code_sha: str,
    worker_generation: int,
) -> ProducerProvenance:
    """Build the exact producer identity for Phase-2 frozen-Teacher capture."""

    return ProducerProvenance(
        teacher_id=teacher_id,
        teacher_sha=FORMAL_TEACHER_SHA,
        teacher_config_hash=FORMAL_TEACHER_CONFIG_HASH,
        simulator_sha=FORMAL_SIMULATOR_SHA,
        capture_code_sha=capture_code_sha,
        worker_generation=worker_generation,
    ).validated()


def capture_frozen_search_record(
    *,
    public_state: Mapping[str, Any],
    teacher_decision: Mapping[str, Any],
    seed: str | int,
    run_id: str,
    decision_index: int,
    provenance: ProducerProvenance,
    worker_id: str,
    floor: int | None = None,
    combat_id: str | None = None,
    terminal_outcome: str | None = None,
) -> dict[str, Any]:
    """Convert one resolved formal Search row to a Phase-2 decision record."""

    if provenance.teacher_sha != FORMAL_TEACHER_SHA:
        raise TeacherCaptureError("teacher provenance is not pinned to final Phase-1 formal SHA")
    if provenance.simulator_sha != FORMAL_SIMULATOR_SHA:
        raise TeacherCaptureError("simulator provenance is not pinned to frozen sts_lightspeed SHA")
    if provenance.teacher_config_hash != FORMAL_TEACHER_CONFIG_HASH:
        raise TeacherCaptureError("teacher config provenance does not match frozen capture contract")

    if not isinstance(public_state, Mapping):
        raise TeacherCaptureError("public_state must be an object")
    if not isinstance(teacher_decision, Mapping):
        raise TeacherCaptureError("teacher_decision must be an object")

    for key in ("search_timed_out", "timed_out"):
        if key in teacher_decision and teacher_decision.get(key) is not False:
            raise TeacherCaptureError("Teacher Search evidence reports a timeout")
    for key in ("search_resolved", "resolved"):
        if key in teacher_decision and teacher_decision.get(key) is not True:
            raise TeacherCaptureError("Teacher Search evidence is not resolved")
    for key in ("search_unresolved_action_ids", "unresolved_action_ids"):
        if key not in teacher_decision:
            continue
        unresolved_ids = teacher_decision.get(key)
        if not isinstance(unresolved_ids, Sequence) or isinstance(unresolved_ids, str | bytes | bytearray):
            raise TeacherCaptureError("Teacher unresolved_action_ids must be a sequence")
        if unresolved_ids:
            raise TeacherCaptureError("Teacher Search evidence contains unresolved actions")

    signature = decision_signature_for_public_state(public_state)
    state_signature = public_state.get("decision_signature")
    if state_signature is not None and state_signature != signature:
        raise TeacherCaptureError("public_state embedded decision_signature drift")
    if teacher_decision.get("decision_signature") != signature:
        raise TeacherCaptureError("Teacher decision signature does not match public state")

    legal_ids = legal_action_ids_for_public_state(public_state)
    teacher_legal = teacher_decision.get("legal_action_ids")
    if not isinstance(teacher_legal, Sequence) or isinstance(teacher_legal, str | bytes | bytearray):
        raise TeacherCaptureError("Teacher legal_action_ids must be a sequence")
    teacher_legal_ids = tuple(sorted(str(value) for value in teacher_legal))
    if len(teacher_legal_ids) != len(set(teacher_legal_ids)) or teacher_legal_ids != legal_ids:
        raise TeacherCaptureError("Teacher legal_action_ids disagree with formal public action identities")

    candidate_rows = teacher_decision.get("search_candidate_scores")
    if not isinstance(candidate_rows, Sequence) or isinstance(candidate_rows, str | bytes | bytearray):
        raise TeacherCaptureError("Teacher search_candidate_scores must be a sequence")
    scores: dict[str, float] = {}
    for row in candidate_rows:
        if not isinstance(row, Mapping):
            raise TeacherCaptureError("Teacher candidate score row must be an object")
        action_id = row.get("action_id")
        raw_score = row.get("score")
        unresolved = row.get("unresolved")
        samples = row.get("samples")
        if not isinstance(action_id, str) or action_id not in legal_ids or action_id in scores:
            raise TeacherCaptureError("Teacher candidate action identity is invalid or duplicated")
        if unresolved is not False:
            raise TeacherCaptureError(f"Teacher candidate is unresolved: {action_id}")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise TeacherCaptureError(f"Teacher candidate has invalid sample count: {action_id}")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise TeacherCaptureError(f"Teacher candidate score is not numeric: {action_id}")
        score = float(raw_score)
        if not math.isfinite(score):
            raise TeacherCaptureError(f"Teacher candidate score is not finite: {action_id}")
        scores[action_id] = score
    if tuple(sorted(scores)) != legal_ids:
        raise TeacherCaptureError("Teacher candidate scores do not cover legal actions exactly")

    tie_values = teacher_decision.get("search_tie_ids")
    if not isinstance(tie_values, Sequence) or isinstance(tie_values, str | bytes | bytearray) or not tie_values:
        raise TeacherCaptureError("Teacher search tie set must be non-empty")
    tie_ids = tuple(sorted(str(value) for value in tie_values))
    if len(tie_ids) != len(set(tie_ids)) or not set(tie_ids).issubset(legal_ids):
        raise TeacherCaptureError("Teacher search tie set contains invalid actions")

    best = max(scores.values())
    expected_ties = tuple(
        sorted(
            action_id
            for action_id, score in scores.items()
            if abs(score - best) <= float(FORMAL_SEARCH_CONFIG["tie_tolerance"])
        )
    )
    if tie_ids != expected_ties:
        raise TeacherCaptureError("Teacher search tie set disagrees with frozen candidate scores")

    unique_best = teacher_decision.get("search_unique_best_action_id")
    expected_unique = tie_ids[0] if len(tie_ids) == 1 else None
    if unique_best != expected_unique:
        raise TeacherCaptureError("Teacher unique-best field disagrees with tie set")

    rollout_count = teacher_decision.get("search_rollout_count")
    if isinstance(rollout_count, bool) or not isinstance(rollout_count, int) or rollout_count <= 0:
        raise TeacherCaptureError("Teacher search_rollout_count must be positive")
    evidence_hash = teacher_decision.get("search_evidence_hash")
    if not isinstance(evidence_hash, str) or len(evidence_hash) != 64:
        raise TeacherCaptureError("Teacher search_evidence_hash must be a SHA-256 hex identity")
    try:
        int(evidence_hash, 16)
    except ValueError as exc:
        raise TeacherCaptureError("Teacher search_evidence_hash must be hexadecimal") from exc

    selected_action_id = tie_ids[0]
    raw_legal_actions = public_state.get("legal_actions")
    if not isinstance(raw_legal_actions, Sequence) or isinstance(raw_legal_actions, str | bytes | bytearray):
        raise TeacherCaptureError("public_state legal_actions must be a sequence")

    return make_decision_record(
        observation=public_state,
        legal_actions=raw_legal_actions,
        selected_action_id=selected_action_id,
        teacher_scores=scores,
        teacher_tie_action_ids=tie_ids,
        decision_signature=signature,
        seed=seed,
        run_id=run_id,
        decision_index=decision_index,
        provenance=provenance,
        worker_id=worker_id,
        floor=floor,
        combat_id=combat_id,
        terminal_outcome=terminal_outcome,
    )


__all__ = [
    "FORMAL_CAPTURE_CONFIG",
    "FORMAL_SEARCH_CONFIG",
    "FORMAL_SIMULATOR_SHA",
    "FORMAL_TEACHER_CONFIG_HASH",
    "FORMAL_TEACHER_REPOSITORY",
    "FORMAL_TEACHER_SHA",
    "TIE_SELECTION_POLICY",
    "TeacherCaptureError",
    "capture_frozen_search_record",
    "formal_teacher_provenance",
]
