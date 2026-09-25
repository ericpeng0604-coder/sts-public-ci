"""Fail-closed STS1 decision-dataset primitives for Phase 2A.

This module deliberately stops before model training.  It turns already-public
Teacher decisions into deterministic, provenance-rich shards while keeping
seed/run metadata outside the policy observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Literal, Mapping, Sequence


PUBLIC_STATE_SCHEMA_VERSION = "sts1-public-state-v1"
ACTION_SCHEMA_VERSION = "sts1-public-action-v1"
RECORD_SCHEMA_VERSION = "sts1-decision-record-v1"
DATASET_SCHEMA_VERSION = "sts1-decision-dataset-v1"
SPLIT_SCHEMA_VERSION = "sts1-seed-split-v1"

SplitName = Literal["train", "validation", "holdout"]

# Exact policy fields used by the frozen Phase-1 DecisionContext. Extra
# reconstruction/provenance metadata is validated but never exposed to Student.
POLICY_OBSERVATION_FIELDS = (
    "hp",
    "max_hp",
    "block",
    "energy",
    "hand",
    "draw_pile",
    "discard_pile",
    "exhaust_pile",
    "powers",
    "enemies",
    "turn",
    "combat_active",
    "relics",
    "potions",
    "gold",
    "floor",
    "act",
    "character",
    "ascension_level",
    "room",
    "screen_type",
    "screen_choices",
    "rewards",
    "map_choices",
)
_POLICY_LIST_FIELDS = frozenset(
    {
        "hand",
        "draw_pile",
        "discard_pile",
        "exhaust_pile",
        "powers",
        "enemies",
        "relics",
        "potions",
        "screen_choices",
        "rewards",
        "map_choices",
    }
)
_COMPOSITION_FIELDS = frozenset({"draw_pile", "discard_pile", "exhaust_pile"})
_TRANSPORT_ACTION_KEYS = frozenset({"command", "transport", "raw_action", "source_action"})

_FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {
        "seed",
        "game_seed",
        "run_seed",
        "rng",
        "rng_state",
        "rng_counter",
        "random_state",
        "uuid",
        "card_uuid",
        "move_id",
        "last_move_id",
        "second_last_move_id",
        "move_history",
        "counter_history",
        "misc_info",
        "battle_context",
        "monster_ai_state",
        "hidden_state",
        "draw_order",
        "future_draw_order",
        "future_encounter",
        "future_event",
        "future_reward",
        "future_potion",
        "future_relic",
        "terminal_outcome",
    }
)


class STS1DatasetError(ValueError):
    """Raised when a Phase 2A dataset invariant is violated."""


class StaleWorkerError(STS1DatasetError):
    """Raised when worker output was produced by a stale/incompatible producer."""


class DuplicateDecisionConflictError(STS1DatasetError):
    """Raised when one decision identity has contradictory labels or payloads."""


@dataclass(frozen=True)
class ProducerProvenance:
    """Exact identity a capture worker must pin before producing decisions."""

    teacher_id: str
    teacher_sha: str
    teacher_config_hash: str
    simulator_sha: str
    capture_code_sha: str
    worker_generation: int
    public_state_schema: str = PUBLIC_STATE_SCHEMA_VERSION
    action_schema: str = ACTION_SCHEMA_VERSION

    def validated(self) -> "ProducerProvenance":
        for name in (
            "teacher_id",
            "teacher_sha",
            "teacher_config_hash",
            "simulator_sha",
            "capture_code_sha",
            "public_state_schema",
            "action_schema",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise STS1DatasetError(f"producer provenance field {name} must be non-empty")
        if isinstance(self.worker_generation, bool) or not isinstance(self.worker_generation, int) or self.worker_generation < 0:
            raise STS1DatasetError("worker_generation must be a non-negative integer")
        if self.public_state_schema != PUBLIC_STATE_SCHEMA_VERSION:
            raise STS1DatasetError(
                f"public-state schema must be {PUBLIC_STATE_SCHEMA_VERSION}, got {self.public_state_schema}"
            )
        if self.action_schema != ACTION_SCHEMA_VERSION:
            raise STS1DatasetError(
                f"action schema must be {ACTION_SCHEMA_VERSION}, got {self.action_schema}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validated()
        return asdict(self)

    @property
    def identity_hash(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class DatasetBuildConfig:
    """Deterministic whole-seed split and shard contract."""

    train_permyriad: int = 8000
    validation_permyriad: int = 1000
    holdout_permyriad: int = 1000
    split_salt: str = "sts1-phase2a-v1"
    shard_size: int = 5000
    require_all_splits: bool = True

    def validated(self) -> "DatasetBuildConfig":
        values = (self.train_permyriad, self.validation_permyriad, self.holdout_permyriad)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise STS1DatasetError("split weights must be non-negative integers")
        if sum(values) != 10_000:
            raise STS1DatasetError("train/validation/holdout split weights must sum to 10000")
        if not self.train_permyriad or not self.validation_permyriad or not self.holdout_permyriad:
            raise STS1DatasetError("train, validation, and holdout must all have non-zero allocation")
        if not isinstance(self.split_salt, str) or not self.split_salt:
            raise STS1DatasetError("split_salt must be non-empty")
        if isinstance(self.shard_size, bool) or not isinstance(self.shard_size, int) or self.shard_size <= 0:
            raise STS1DatasetError("shard_size must be a positive integer")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validated()
        return asdict(self)


def canonical_json(value: Any) -> str:
    """Canonical JSON used for identities and deterministic artifact output."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_decision_record(
    *,
    observation: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
    selected_action_id: str,
    teacher_scores: Mapping[str, float],
    teacher_tie_action_ids: Sequence[str],
    decision_signature: str,
    seed: str | int,
    run_id: str,
    decision_index: int,
    provenance: ProducerProvenance,
    worker_id: str,
    floor: int | None = None,
    combat_id: str | None = None,
    terminal_outcome: str | None = None,
) -> dict[str, Any]:
    """Create one canonical record while separating policy state from provenance."""

    provenance.validated()
    raw_observation = _json_clone(observation)
    if not isinstance(raw_observation, dict):
        raise STS1DatasetError("observation must be an object")
    _validate_public_observation(raw_observation)
    schema = raw_observation.get("schema_version")
    if schema is not None and schema != PUBLIC_STATE_SCHEMA_VERSION:
        raise STS1DatasetError(
            f"observation schema must be {PUBLIC_STATE_SCHEMA_VERSION} when present, got {schema}"
        )
    public_observation = project_policy_observation(raw_observation)

    normalized_actions = _normalize_legal_actions(legal_actions)
    legal_ids = [row["action_id"] for row in normalized_actions]
    legal_set = set(legal_ids)
    if selected_action_id not in legal_set:
        raise STS1DatasetError("Teacher selected action is not in legal_actions")

    score_map = _normalize_scores(teacher_scores, legal_set)
    tie_ids = _normalize_tie_set(teacher_tie_action_ids, legal_set)
    if selected_action_id not in tie_ids:
        raise STS1DatasetError("Teacher selected action must belong to the Teacher tie set")

    if not isinstance(decision_signature, str) or not decision_signature:
        raise STS1DatasetError("decision_signature must be non-empty")
    expected_signature = decision_signature_for_policy_input(public_observation, normalized_actions)
    if decision_signature != expected_signature:
        raise STS1DatasetError(
            f"decision_signature disagrees with frozen public policy projection: {decision_signature}!={expected_signature}"
        )
    seed_text = str(seed)
    if not seed_text:
        raise STS1DatasetError("seed provenance must be non-empty")
    if not isinstance(run_id, str) or not run_id:
        raise STS1DatasetError("run_id must be non-empty")
    if isinstance(decision_index, bool) or not isinstance(decision_index, int) or decision_index < 0:
        raise STS1DatasetError("decision_index must be a non-negative integer")
    if not isinstance(worker_id, str) or not worker_id:
        raise STS1DatasetError("worker_id must be non-empty")
    if floor is not None and (isinstance(floor, bool) or not isinstance(floor, int) or floor < 0):
        raise STS1DatasetError("floor must be a non-negative integer when present")
    if combat_id is not None and (not isinstance(combat_id, str) or not combat_id):
        raise STS1DatasetError("combat_id must be a non-empty string when present")
    if terminal_outcome is not None and not isinstance(terminal_outcome, str):
        raise STS1DatasetError("terminal_outcome must be a string when present")

    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "decision_signature": decision_signature,
        "observation": public_observation,
        "legal_actions": normalized_actions,
        "teacher": {
            "selected_action_id": selected_action_id,
            "tie_action_ids": tie_ids,
            "scores": score_map,
        },
        "provenance": {
            "seed": seed_text,
            "run_id": run_id,
            "decision_index": decision_index,
            "floor": floor,
            "combat_id": combat_id,
            "terminal_outcome": terminal_outcome,
            "worker_id": worker_id,
            "producer": provenance.to_dict(),
        },
    }


def validate_decision_record(
    record: Mapping[str, Any],
    *,
    expected_provenance: ProducerProvenance,
) -> dict[str, Any]:
    """Validate one record and fail closed on stale producer identity."""

    expected = expected_provenance.validated().to_dict()
    payload = _json_clone(record)
    if not isinstance(payload, dict):
        raise STS1DatasetError("decision record must be an object")
    if payload.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise STS1DatasetError(f"decision record schema must be {RECORD_SCHEMA_VERSION}")

    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise STS1DatasetError("decision record observation must be an object")
    _validate_public_observation(observation)
    schema = observation.get("schema_version")
    if schema is not None and schema != PUBLIC_STATE_SCHEMA_VERSION:
        raise STS1DatasetError("decision record observation has incompatible public-state schema")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise STS1DatasetError("decision record provenance must be an object")
    producer = provenance.get("producer")
    if not isinstance(producer, dict):
        raise STS1DatasetError("decision record producer provenance must be an object")
    mismatches = [key for key, value in expected.items() if producer.get(key) != value]
    extra_identity_keys = sorted(set(producer).difference(expected))
    if mismatches or extra_identity_keys:
        details = ",".join(mismatches + [f"extra:{key}" for key in extra_identity_keys])
        raise StaleWorkerError(f"stale/incompatible worker provenance: {details}")

    seed = provenance.get("seed")
    run_id = provenance.get("run_id")
    decision_index = provenance.get("decision_index")
    worker_id = provenance.get("worker_id")
    if not isinstance(seed, str) or not seed:
        raise STS1DatasetError("decision record seed provenance must be non-empty")
    if not isinstance(run_id, str) or not run_id:
        raise STS1DatasetError("decision record run_id must be non-empty")
    if isinstance(decision_index, bool) or not isinstance(decision_index, int) or decision_index < 0:
        raise STS1DatasetError("decision record decision_index must be non-negative")
    if not isinstance(worker_id, str) or not worker_id:
        raise STS1DatasetError("decision record worker_id must be non-empty")

    legal_actions = payload.get("legal_actions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise STS1DatasetError("decision record legal_actions must be a non-empty array")
    legal_ids: list[str] = []
    for item in legal_actions:
        if not isinstance(item, dict):
            raise STS1DatasetError("legal action rows must be objects")
        action_id = item.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise STS1DatasetError("legal action row has invalid action_id")
        legal_ids.append(action_id)
    if len(legal_ids) != len(set(legal_ids)):
        raise STS1DatasetError("legal action ids must be unique")
    legal_set = set(legal_ids)

    teacher = payload.get("teacher")
    if not isinstance(teacher, dict):
        raise STS1DatasetError("decision record teacher block must be an object")
    selected = teacher.get("selected_action_id")
    if not isinstance(selected, str) or selected not in legal_set:
        raise STS1DatasetError("Teacher selected action is not legal")
    _normalize_scores(teacher.get("scores"), legal_set)
    tie_ids = _normalize_tie_set(teacher.get("tie_action_ids"), legal_set)
    if selected not in tie_ids:
        raise STS1DatasetError("Teacher selected action is absent from tie set")

    signature = payload.get("decision_signature")
    if not isinstance(signature, str) or not signature:
        raise STS1DatasetError("decision_signature must be non-empty")
    expected_signature = decision_signature_for_policy_input(observation, legal_actions)
    if signature != expected_signature:
        raise STS1DatasetError("stored decision_signature disagrees with policy observation/legal actions")
    return payload


def project_policy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project raw public state to the exact frozen Phase-1 policy fields."""

    raw = _json_clone(observation)
    if not isinstance(raw, dict):
        raise STS1DatasetError("observation must be an object")
    _validate_public_observation(raw)
    schema = raw.get("schema_version")
    if schema is not None and schema != PUBLIC_STATE_SCHEMA_VERSION:
        raise STS1DatasetError(
            f"observation schema must be {PUBLIC_STATE_SCHEMA_VERSION} when present, got {schema}"
        )

    projected: dict[str, Any] = {}
    for field in POLICY_OBSERVATION_FIELDS:
        if field in raw:
            value = raw[field]
        else:
            value = [] if field in _POLICY_LIST_FIELDS else None
        if field in _COMPOSITION_FIELDS:
            if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
                projected[field] = []
            else:
                projected[field] = sorted((_json_clone(item) for item in value), key=canonical_json)
        elif field in _POLICY_LIST_FIELDS:
            if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
                projected[field] = []
            else:
                projected[field] = [_json_clone(item) for item in value]
        else:
            projected[field] = _json_clone(value)
    return projected


def decision_signature_for_policy_input(
    projected_observation: Mapping[str, Any],
    normalized_legal_actions: Sequence[Mapping[str, Any]],
) -> str:
    """Recompute the frozen Phase-1 DecisionContext signature."""

    state = _json_clone(projected_observation)
    if not isinstance(state, dict):
        raise STS1DatasetError("projected observation must be an object")
    policy_actions: list[dict[str, Any]] = []
    for action in normalized_legal_actions:
        if not isinstance(action, Mapping) or not isinstance(action.get("payload"), Mapping):
            raise STS1DatasetError("normalized legal action is malformed")
        policy_actions.append(_json_clone(action["payload"]))
    policy_actions.sort(key=canonical_json)
    signature_payload = dict(state)
    signature_payload["legal_actions"] = policy_actions
    return sha256_json(signature_payload)


def legal_action_ids_for_public_state(public_state: Mapping[str, Any]) -> tuple[str, ...]:
    """Recompute exact frozen Phase-1 executable action identities.

    Formal capture must never trust an externally supplied ``action_id``. The
    Phase-1 contract derives the id from the policy-visible action payload after
    stripping transport-only keys, so this helper deliberately does the same.
    """

    legal_actions = public_state.get("legal_actions")
    if not isinstance(legal_actions, Sequence) or isinstance(legal_actions, str | bytes | bytearray):
        raise STS1DatasetError("public_state legal_actions must be a sequence")

    action_ids: list[str] = []
    for action in legal_actions:
        payload = _json_clone(action)
        if not isinstance(payload, dict):
            raise STS1DatasetError("legal action must be an object")
        payload.pop("action_id", None)
        for key in _TRANSPORT_ACTION_KEYS:
            payload.pop(key, None)
        action_ids.append(sha256_json(payload))
    if len(action_ids) != len(set(action_ids)):
        raise STS1DatasetError("duplicate legal action identity")
    return tuple(sorted(action_ids))


def decision_signature_for_public_state(public_state: Mapping[str, Any]) -> str:
    """Recompute the exact formal signature directly from one raw public state."""

    legal_actions = public_state.get("legal_actions")
    if not isinstance(legal_actions, Sequence) or isinstance(legal_actions, str | bytes | bytearray):
        raise STS1DatasetError("public_state legal_actions must be a sequence")
    projected = project_policy_observation(public_state)
    normalized = _normalize_legal_actions(legal_actions)
    return decision_signature_for_policy_input(projected, normalized)


def split_for_seed(seed: str | int, config: DatasetBuildConfig = DatasetBuildConfig()) -> SplitName:
    """Assign every occurrence of one seed to exactly one stable split."""

    config.validated()
    seed_text = str(seed)
    digest = hashlib.sha256(f"{config.split_salt}\0{seed_text}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < config.train_permyriad:
        return "train"
    if bucket < config.train_permyriad + config.validation_permyriad:
        return "validation"
    return "holdout"


def build_dataset(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    expected_provenance: ProducerProvenance,
    config: DatasetBuildConfig = DatasetBuildConfig(),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build deterministic whole-seed shards and a checksum manifest.

    Any stale worker, illegal Teacher label, hidden-state key, contradictory
    duplicate, or malformed provenance aborts the build before publication.
    """

    config.validated()
    expected_provenance.validated()
    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise FileExistsError(f"dataset output already exists: {output}")

    unique: dict[tuple[str, str, int, str], tuple[str, dict[str, Any]]] = {}
    duplicates_dropped = 0
    for record in records:
        normalized = validate_decision_record(record, expected_provenance=expected_provenance)
        provenance = normalized["provenance"]
        key = (
            str(provenance["seed"]),
            str(provenance["run_id"]),
            int(provenance["decision_index"]),
            str(normalized["decision_signature"]),
        )
        serialized = canonical_json(normalized)
        previous = unique.get(key)
        if previous is not None:
            if previous[0] != serialized:
                raise DuplicateDecisionConflictError(
                    "same seed/run/decision/signature has contradictory record content"
                )
            duplicates_dropped += 1
            continue
        unique[key] = (serialized, normalized)

    if not unique:
        raise STS1DatasetError("no valid decision records supplied")

    by_split: dict[SplitName, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "holdout": [],
    }
    seed_splits: dict[str, SplitName] = {}
    for _, normalized in unique.values():
        seed = str(normalized["provenance"]["seed"])
        split = split_for_seed(seed, config)
        prior = seed_splits.setdefault(seed, split)
        if prior != split:
            raise STS1DatasetError(f"seed leakage detected for seed {seed}")
        with_split = dict(normalized)
        with_split["split"] = split
        by_split[split].append(with_split)

    split_counts = {name: len(rows) for name, rows in by_split.items()}
    if config.require_all_splits and any(count == 0 for count in split_counts.values()):
        raise STS1DatasetError(f"dataset must contain train/validation/holdout records: {split_counts}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent)))
    try:
        shard_root = staging / "shards"
        shard_root.mkdir()
        shard_manifest: list[dict[str, Any]] = []
        for split in ("train", "validation", "holdout"):
            rows = sorted(
                by_split[split],
                key=lambda row: (
                    str(row["provenance"]["seed"]),
                    str(row["provenance"]["run_id"]),
                    int(row["provenance"]["decision_index"]),
                    str(row["decision_signature"]),
                ),
            )
            for shard_index, start in enumerate(range(0, len(rows), config.shard_size)):
                chunk = rows[start : start + config.shard_size]
                relative = Path("shards") / f"{split}-{shard_index:05d}.jsonl"
                path = staging / relative
                text = "".join(canonical_json(row) + "\n" for row in chunk)
                path.write_text(text, encoding="utf-8", newline="\n")
                shard_manifest.append(
                    {
                        "path": relative.as_posix(),
                        "split": split,
                        "record_count": len(chunk),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )

        split_seed_hashes = {
            split: sorted(
                hashlib.sha256(seed.encode("utf-8")).hexdigest()
                for seed, assigned in seed_splits.items()
                if assigned == split
            )
            for split in ("train", "validation", "holdout")
        }
        dataset_identity = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "producer_identity_hash": expected_provenance.identity_hash,
            "split_contract": {"schema_version": SPLIT_SCHEMA_VERSION, **config.to_dict()},
            "shards": shard_manifest,
        }
        manifest = {
            **dataset_identity,
            "dataset_hash": sha256_json(dataset_identity),
            "record_count": sum(split_counts.values()),
            "duplicates_dropped": duplicates_dropped,
            "split_counts": split_counts,
            "unique_seed_counts": {
                split: len(split_seed_hashes[split])
                for split in ("train", "validation", "holdout")
            },
            "split_seed_hashes": split_seed_hashes,
            "producer_provenance": expected_provenance.to_dict(),
            "policy_observation_contains_seed": False,
            "provenance_is_not_policy_input": True,
            "holdout_training_allowed": False,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _replace_directory(staging, output, overwrite=overwrite)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise STS1DatasetError("invalid STS1 decision dataset manifest")
    identity = {
        "schema_version": payload.get("schema_version"),
        "producer_identity_hash": payload.get("producer_identity_hash"),
        "split_contract": payload.get("split_contract"),
        "shards": payload.get("shards"),
    }
    if payload.get("dataset_hash") != sha256_json(identity):
        raise STS1DatasetError("STS1 decision dataset manifest identity hash mismatch")
    producer = payload.get("producer_provenance")
    if not isinstance(producer, dict) or payload.get("producer_identity_hash") != sha256_json(producer):
        raise STS1DatasetError("STS1 decision dataset producer identity hash mismatch")
    return payload


def _normalize_legal_actions(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(actions, Sequence) or isinstance(actions, str | bytes | bytearray) or not actions:
        raise STS1DatasetError("legal_actions must be a non-empty sequence")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in actions:
        cloned = _json_clone(action)
        if not isinstance(cloned, dict):
            raise STS1DatasetError("legal action must be an object")
        explicit = cloned.pop("action_id", None)
        for key in _TRANSPORT_ACTION_KEYS:
            cloned.pop(key, None)
        action_id = explicit if isinstance(explicit, str) and explicit else sha256_json(cloned)
        if action_id in seen:
            raise STS1DatasetError("duplicate legal action identity")
        seen.add(action_id)
        result.append({"action_id": action_id, "payload": cloned})
    return result


def _normalize_scores(scores: Any, legal_ids: set[str]) -> dict[str, float]:
    if not isinstance(scores, Mapping):
        raise STS1DatasetError("Teacher scores must be an object keyed by legal action id")
    if set(scores) != legal_ids:
        missing = sorted(legal_ids.difference(scores))
        extra = sorted(set(scores).difference(legal_ids))
        raise STS1DatasetError(f"Teacher scores must cover legal actions exactly; missing={missing}, extra={extra}")
    normalized: dict[str, float] = {}
    for key in sorted(legal_ids):
        value = scores[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise STS1DatasetError(f"Teacher score for {key} must be finite")
        normalized[key] = float(value)
    return normalized


def _normalize_tie_set(values: Any, legal_ids: set[str]) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes | bytearray) or not values:
        raise STS1DatasetError("Teacher tie set must be a non-empty sequence")
    normalized = [str(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise STS1DatasetError("Teacher tie set must not contain duplicates")
    if not set(normalized).issubset(legal_ids):
        raise STS1DatasetError("Teacher tie set contains an illegal action")
    return sorted(normalized)


def _validate_public_observation(value: Any, *, path: str = "observation") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            compact = "".join(character for character in lowered if character.isalnum())
            if (
                lowered in _FORBIDDEN_OBSERVATION_KEYS
                or lowered.startswith("future_")
                or lowered.startswith("rng_")
                or lowered.endswith("_rng_state")
                or lowered.endswith("_seed")
                or compact.endswith(
                    (
                        "movehistory",
                        "counterhistory",
                        "miscinfo",
                        "battlecontext",
                        "monsteraistate",
                        "hiddenstate",
                        "draworder",
                    )
                )
            ):
                raise STS1DatasetError(f"forbidden hidden/provenance key in policy observation: {path}.{key}")
            _validate_public_observation(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _validate_public_observation(child, path=f"{path}[{index}]")


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise STS1DatasetError(f"value is not canonical JSON: {exc}") from exc


def _replace_directory(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"dataset output already exists: {output}")
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}") if output.exists() else None
    try:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
        if backup is not None:
            output.rename(backup)
        staging.rename(output)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if output.exists() and output != staging:
            shutil.rmtree(output, ignore_errors=True)
        if backup is not None and backup.exists() and not output.exists():
            backup.rename(output)
        raise


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "DatasetBuildConfig",
    "DuplicateDecisionConflictError",
    "ProducerProvenance",
    "POLICY_OBSERVATION_FIELDS",
    "PUBLIC_STATE_SCHEMA_VERSION",
    "RECORD_SCHEMA_VERSION",
    "SPLIT_SCHEMA_VERSION",
    "STS1DatasetError",
    "StaleWorkerError",
    "build_dataset",
    "canonical_json",
    "decision_signature_for_policy_input",
    "decision_signature_for_public_state",
    "legal_action_ids_for_public_state",
    "load_manifest",
    "make_decision_record",
    "project_policy_observation",
    "sha256_json",
    "split_for_seed",
    "validate_decision_record",
]
