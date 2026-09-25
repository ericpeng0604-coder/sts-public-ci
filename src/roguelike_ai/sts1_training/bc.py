"""Behavior-cloning input boundary for STS1 Phase 2A.

The loader intentionally exposes only policy inputs and labels.  Seed/run,
worker, simulator, and Teacher provenance stay in dataset artifacts and are
not returned as model features.  Frozen holdout access is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from .dataset import DATASET_SCHEMA_VERSION, RECORD_SCHEMA_VERSION, STS1DatasetError, load_manifest


SplitName = Literal["train", "validation", "holdout"]


class HoldoutAccessError(STS1DatasetError):
    """Raised when training code tries to read the frozen holdout implicitly."""


@dataclass(frozen=True)
class BCExample:
    """One provenance-free Student imitation example."""

    decision_signature: str
    observation: dict[str, Any]
    action_ids: tuple[str, ...]
    action_payloads: tuple[dict[str, Any], ...]
    legal_mask: tuple[bool, ...]
    selected_index: int
    tie_indices: tuple[int, ...]


def load_bc_examples(
    dataset_dir: str | Path,
    split: SplitName,
    *,
    allow_holdout: bool = False,
) -> tuple[BCExample, ...]:
    """Load checksum-verified examples without exposing provenance as features."""

    if split not in ("train", "validation", "holdout"):
        raise STS1DatasetError(f"unsupported split: {split}")
    if split == "holdout" and not allow_holdout:
        raise HoldoutAccessError("holdout is frozen and cannot be opened by the default BC training path")

    root = Path(dataset_dir)
    manifest = load_manifest(root)
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise STS1DatasetError("incompatible STS1 dataset schema")
    if split == "holdout" and manifest.get("holdout_training_allowed") is not False:
        raise STS1DatasetError("dataset manifest does not explicitly freeze holdout training access")

    examples: list[BCExample] = []
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise STS1DatasetError("dataset manifest shards must be an array")
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("split") != split:
            continue
        relative = shard.get("path")
        expected_sha = shard.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise STS1DatasetError("dataset manifest contains invalid shard metadata")
        path = root / relative
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise STS1DatasetError(f"dataset shard checksum mismatch: {relative}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                continue
            payload = json.loads(line)
            try:
                examples.append(_to_example(payload, expected_split=split))
            except Exception as exc:
                raise STS1DatasetError(f"invalid BC record {relative}:{line_number}: {exc}") from exc
    return tuple(examples)


def _to_example(payload: Any, *, expected_split: SplitName) -> BCExample:
    if not isinstance(payload, dict) or payload.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise STS1DatasetError("record schema mismatch")
    if payload.get("split") != expected_split:
        raise STS1DatasetError("record split disagrees with shard split")
    observation = payload.get("observation")
    legal_actions = payload.get("legal_actions")
    teacher = payload.get("teacher")
    signature = payload.get("decision_signature")
    if not isinstance(observation, dict) or not isinstance(legal_actions, list) or not legal_actions:
        raise STS1DatasetError("record is missing observation/legal actions")
    if not isinstance(teacher, dict) or not isinstance(signature, str) or not signature:
        raise STS1DatasetError("record is missing Teacher label/signature")

    action_ids: list[str] = []
    action_payloads: list[dict[str, Any]] = []
    for action in legal_actions:
        if not isinstance(action, dict):
            raise STS1DatasetError("legal action row must be an object")
        action_id = action.get("action_id")
        action_payload = action.get("payload")
        if not isinstance(action_id, str) or not action_id or not isinstance(action_payload, dict):
            raise STS1DatasetError("legal action row is malformed")
        action_ids.append(action_id)
        action_payloads.append(action_payload)
    if len(action_ids) != len(set(action_ids)):
        raise STS1DatasetError("BC example contains duplicate legal action ids")

    selected_action_id = teacher.get("selected_action_id")
    tie_action_ids = teacher.get("tie_action_ids")
    if not isinstance(selected_action_id, str) or selected_action_id not in action_ids:
        raise STS1DatasetError("Teacher label is not legal")
    if not isinstance(tie_action_ids, list) or not tie_action_ids:
        raise STS1DatasetError("Teacher tie set is missing")
    try:
        tie_indices = tuple(sorted(action_ids.index(str(action_id)) for action_id in tie_action_ids))
    except ValueError as exc:
        raise STS1DatasetError("Teacher tie set contains illegal action") from exc

    return BCExample(
        decision_signature=signature,
        observation=observation,
        action_ids=tuple(action_ids),
        action_payloads=tuple(action_payloads),
        legal_mask=tuple(True for _ in action_ids),
        selected_index=action_ids.index(selected_action_id),
        tie_indices=tie_indices,
    )


__all__ = ["BCExample", "HoldoutAccessError", "load_bc_examples"]
