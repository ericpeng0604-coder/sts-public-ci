"""Fail-closed inference for the exact accepted STS1 Student v0.

Phase 3 is evaluation-only. This module does not train or tune a model. It
loads an already-materialized weight artifact and accepts it only when the
frozen Phase-2 source/config/model/data identities match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PREDECESSOR_HEAD = "046854ecfbb497f2fc5e4379990300e46be0c248"
EXPECTED_SOURCE_SHA256 = "06e206b7f04e6e7b77ad1fc8c35ba1bcdca1c978596fbb176933323e3940aed3"
EXPECTED_CONFIG_HASH = "906afcaec38a2050f48e28f2351745408bbc0c72607704dbce2e134cdff4192c"
EXPECTED_MODEL_SHA256 = "e15604b95247615a6d424f834e8b8a1e9fd680af4fe9e22a6754372027e89513"
EXPECTED_DATASET_HASH = "4509f9c48206606638f24f264c0dbc471b1cc46c60f2633b3659f72cf7c6ccea"
EXPECTED_FEATURE_COUNT = 3028
STUDENT_SCHEMA_VERSION = "sts1-student-v0-linear-v1"
ARTIFACT_SCHEMA_VERSION = "sts1-phase3-frozen-student-v0-artifact-v1"
PUBLIC_STATE_SCHEMA_VERSION = "sts1-public-state-v1"
ACTION_SCHEMA_VERSION = "sts1-public-action-v1"

POLICY_OBSERVATION_FIELDS = (
    "hp", "max_hp", "block", "energy", "hand", "draw_pile", "discard_pile",
    "exhaust_pile", "powers", "enemies", "turn", "combat_active", "relics",
    "potions", "gold", "floor", "act", "character", "ascension_level", "room",
    "screen_type", "screen_choices", "rewards", "map_choices",
)
_POLICY_LIST_FIELDS = frozenset({
    "hand", "draw_pile", "discard_pile", "exhaust_pile", "powers", "enemies",
    "relics", "potions", "screen_choices", "rewards", "map_choices",
})
_COMPOSITION_FIELDS = frozenset({"draw_pile", "discard_pile", "exhaust_pile"})
_TRANSPORT_ACTION_KEYS = frozenset({"command", "transport", "raw_action", "source_action"})
_FORBIDDEN_OBSERVATION_KEYS = frozenset({
    "seed", "game_seed", "run_seed", "rng", "rng_state", "rng_counter", "random_state",
    "uuid", "card_uuid", "move_id", "last_move_id", "second_last_move_id", "move_history",
    "counter_history", "misc_info", "battle_context", "monster_ai_state", "hidden_state",
    "draw_order", "future_draw_order", "future_encounter", "future_event", "future_reward",
    "future_potion", "future_relic", "terminal_outcome",
})


class FrozenStudentError(ValueError):
    """Frozen Student identity or policy-input contract was violated."""


@dataclass(frozen=True)
class StudentDecision:
    action_index: int
    action_id: str
    command: str | None
    score: float
    legal_action_count: int


@dataclass(frozen=True)
class FrozenStudentV0:
    weights: Mapping[str, float]
    artifact_sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> "FrozenStudentV0":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise FrozenStudentError("frozen Student artifact must be a JSON object")
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenStudentV0":
        expected = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "predecessor_head": PREDECESSOR_HEAD,
            "student_source_sha256": EXPECTED_SOURCE_SHA256,
            "student_config_hash": EXPECTED_CONFIG_HASH,
            "student_model_sha256": EXPECTED_MODEL_SHA256,
            "dataset_hash": EXPECTED_DATASET_HASH,
            "feature_count": EXPECTED_FEATURE_COUNT,
            "student_schema_version": STUDENT_SCHEMA_VERSION,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise FrozenStudentError(f"frozen Student identity drift for {key}: {payload.get(key)!r}")

        raw_weights = payload.get("weights")
        if not isinstance(raw_weights, Mapping) or len(raw_weights) != EXPECTED_FEATURE_COUNT:
            raise FrozenStudentError("frozen Student weights are missing or feature count drifted")
        weights: dict[str, float] = {}
        for raw_key, raw_value in raw_weights.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise FrozenStudentError("frozen Student weight keys must be non-empty strings")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise FrozenStudentError(f"frozen Student weight is not numeric: {raw_key}")
            value = float(raw_value)
            if not math.isfinite(value):
                raise FrozenStudentError(f"frozen Student weight is not finite: {raw_key}")
            weights[raw_key] = value

        model_payload = {
            "schema_version": STUDENT_SCHEMA_VERSION,
            "config_hash": EXPECTED_CONFIG_HASH,
            "weights": {key: weights[key] for key in sorted(weights)},
        }
        actual_model_sha = sha256_json(model_payload)
        if actual_model_sha != EXPECTED_MODEL_SHA256:
            raise FrozenStudentError(f"frozen Student model hash drift: {actual_model_sha}")

        envelope = dict(payload)
        envelope["weights"] = {key: weights[key] for key in sorted(weights)}
        return cls(weights=weights, artifact_sha256=sha256_json(envelope))

    def select_action(
        self,
        public_state: Mapping[str, Any],
        *,
        require_command: bool = True,
    ) -> StudentDecision:
        legal = public_state.get("legal_actions")
        if not isinstance(legal, Sequence) or isinstance(legal, str | bytes | bytearray) or not legal:
            raise FrozenStudentError("Student requires a non-empty legal_actions array")

        observation = project_policy_observation(public_state)
        action_rows: list[tuple[str, dict[str, Any], str | None]] = []
        seen_ids: set[str] = set()
        for raw_action in legal:
            if not isinstance(raw_action, Mapping):
                raise FrozenStudentError("legal action must be an object")
            payload = normalize_action_payload(raw_action)
            action_id = sha256_json(payload)
            if action_id in seen_ids:
                raise FrozenStudentError("duplicate legal action identity")
            seen_ids.add(action_id)
            raw_command = raw_action.get("command")
            command = raw_command.strip() if isinstance(raw_command, str) and raw_command.strip() else None
            if require_command and command is None:
                raise FrozenStudentError("real-game legal action is missing an executable command")
            action_rows.append((action_id, payload, command))

        state_features: dict[str, float] = {}
        _flatten(observation, "state", state_features)
        scores: list[float] = []
        for action_id, payload, _ in action_rows:
            action_features: dict[str, float] = {f"action.id={action_id}": 1.0}
            _flatten(payload, "action", action_features)
            row = dict(action_features)
            for state_key, state_value in state_features.items():
                for action_key, action_value in action_features.items():
                    product = state_value * action_value
                    if product != 0.0:
                        row[f"cross::{state_key}::{action_key}"] = product
            scores.append(sum(float(self.weights.get(key, 0.0)) * value for key, value in row.items()))

        best_index = max(range(len(scores)), key=scores.__getitem__)
        action_id, _, command = action_rows[best_index]
        return StudentDecision(
            action_index=best_index,
            action_id=action_id,
            command=command,
            score=scores[best_index],
            legal_action_count=len(action_rows),
        )


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_action_payload(action: Mapping[str, Any]) -> dict[str, Any]:
    cloned = json.loads(canonical_json(action))
    if not isinstance(cloned, dict):
        raise FrozenStudentError("legal action must serialize to an object")
    cloned.pop("action_id", None)
    for key in _TRANSPORT_ACTION_KEYS:
        cloned.pop(key, None)
    return cloned


def project_policy_observation(public_state: Mapping[str, Any]) -> dict[str, Any]:
    _validate_public_observation(public_state)
    projected: dict[str, Any] = {}
    for field in POLICY_OBSERVATION_FIELDS:
        if field in public_state:
            value = json.loads(canonical_json(public_state[field]))
        else:
            value = [] if field in _POLICY_LIST_FIELDS else None
        if field in _COMPOSITION_FIELDS:
            if not isinstance(value, list):
                value = []
            else:
                value = sorted(value, key=canonical_json)
        elif field in _POLICY_LIST_FIELDS and not isinstance(value, list):
            value = []
        projected[field] = value
    return projected


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
                or compact.endswith((
                    "movehistory", "counterhistory", "miscinfo", "battlecontext",
                    "monsteraistate", "hiddenstate", "draworder",
                ))
            ):
                raise FrozenStudentError(f"forbidden hidden/provenance key in policy observation: {path}.{key}")
            _validate_public_observation(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _validate_public_observation(child, path=f"{path}[{index}]")


def _flatten(value: Any, prefix: str, output: dict[str, float]) -> None:
    if value is None:
        output[f"{prefix}=null"] = 1.0
        return
    if isinstance(value, bool):
        output[f"{prefix}={str(value).lower()}"] = 1.0
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise FrozenStudentError(f"non-finite numeric policy feature at {prefix}")
        output[prefix] = numeric / (1.0 + abs(numeric))
        return
    if isinstance(value, str):
        output[f"{prefix}={value}"] = 1.0
        return
    if isinstance(value, Mapping):
        output[f"{prefix}.len"] = float(len(value)) / (1.0 + float(len(value)))
        for key in sorted(value):
            _flatten(value[key], f"{prefix}.{key}", output)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        output[f"{prefix}.len"] = float(len(value)) / (1.0 + float(len(value)))
        for index, item in enumerate(value):
            _flatten(item, f"{prefix}[{index}]", output)
        return
    raise FrozenStudentError(f"unsupported policy feature type at {prefix}: {type(value).__name__}")


__all__ = [
    "ACTION_SCHEMA_VERSION", "ARTIFACT_SCHEMA_VERSION", "EXPECTED_CONFIG_HASH",
    "EXPECTED_DATASET_HASH", "EXPECTED_FEATURE_COUNT", "EXPECTED_MODEL_SHA256",
    "EXPECTED_SOURCE_SHA256", "FrozenStudentError", "FrozenStudentV0", "PREDECESSOR_HEAD",
    "PUBLIC_STATE_SCHEMA_VERSION", "STUDENT_SCHEMA_VERSION", "StudentDecision", "canonical_json",
    "normalize_action_payload", "project_policy_observation", "sha256_json",
]
