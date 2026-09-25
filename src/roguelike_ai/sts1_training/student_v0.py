"""Small deterministic STS1 behavior-cloning pilot.

This module intentionally accepts already-sanitized ``BCExample`` objects only.
It cannot open the frozen holdout or provenance-bearing dataset records itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from roguelike_ai.behavior_cloning import masked_softmax

from .bc import BCExample


STUDENT_V0_SCHEMA_VERSION = "sts1-student-v0-linear-v1"


class StudentV0Error(ValueError):
    """Raised when the bounded Student v0 pilot input is invalid."""


@dataclass(frozen=True)
class StudentV0Config:
    epochs: int = 40
    learning_rate: float = 0.05
    weight_decay: float = 1e-4

    def validate(self) -> None:
        if self.epochs <= 0:
            raise StudentV0Error("epochs must be positive")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise StudentV0Error("learning_rate must be finite and positive")
        if self.weight_decay < 0.0 or not math.isfinite(self.weight_decay):
            raise StudentV0Error("weight_decay must be finite and non-negative")


@dataclass(frozen=True)
class StudentV0Result:
    schema_version: str
    config: dict[str, object]
    config_hash: str
    model_sha256: str
    feature_count: int
    train: dict[str, object]
    validation: dict[str, object]
    holdout_opened: bool
    promotion_decision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": dict(self.config),
            "config_hash": self.config_hash,
            "model_sha256": self.model_sha256,
            "feature_count": self.feature_count,
            "train": dict(self.train),
            "validation": dict(self.validation),
            "holdout_opened": self.holdout_opened,
            "promotion_decision": self.promotion_decision,
        }


def train_student_v0(
    train_examples: Sequence[BCExample],
    validation_examples: Sequence[BCExample],
    *,
    config: StudentV0Config | None = None,
) -> StudentV0Result:
    """Train the smallest deterministic legal-candidate linear BC baseline."""

    config = config or StudentV0Config()
    config.validate()
    train = _ordered_examples(train_examples, name="train")
    validation = _ordered_examples(validation_examples, name="validation")
    if not validation:
        raise StudentV0Error("validation examples are required for the bounded pilot")

    weights: dict[str, float] = {}
    for _ in range(config.epochs):
        gradient: dict[str, float] = {}
        for example in train:
            rows = _feature_rows(example)
            logits = [_dot(weights, row) for row in rows]
            probabilities = masked_softmax(logits, example.legal_mask)
            for index, row in enumerate(rows):
                delta = probabilities[index] - float(index == example.selected_index)
                for key, value in row.items():
                    gradient[key] = gradient.get(key, 0.0) + delta * value
        scale = config.learning_rate / float(len(train))
        for key in set(weights) | set(gradient):
            current = weights.get(key, 0.0)
            updated = current * (1.0 - config.learning_rate * config.weight_decay) - scale * gradient.get(key, 0.0)
            if not math.isfinite(updated):
                raise StudentV0Error("training produced a non-finite weight")
            if updated == 0.0:
                weights.pop(key, None)
            else:
                weights[key] = updated

    config_payload = asdict(config)
    config_hash = _sha256_json(config_payload)
    model_payload = {
        "schema_version": STUDENT_V0_SCHEMA_VERSION,
        "config_hash": config_hash,
        "weights": {key: weights[key] for key in sorted(weights)},
    }
    return StudentV0Result(
        schema_version=STUDENT_V0_SCHEMA_VERSION,
        config=config_payload,
        config_hash=config_hash,
        model_sha256=_sha256_json(model_payload),
        feature_count=len(weights),
        train=evaluate_student_v0(train, weights),
        validation=evaluate_student_v0(validation, weights),
        holdout_opened=False,
        promotion_decision="NOT_VERIFIED",
    )


def evaluate_student_v0(examples: Sequence[BCExample], weights: Mapping[str, float]) -> dict[str, object]:
    ordered = _ordered_examples(examples, name="evaluation", allow_empty=True)
    if not ordered:
        return {
            "decisions": 0,
            "loss": None,
            "top1_accuracy": None,
            "tie_aware_accuracy": None,
            "mean_confidence": None,
            "mean_margin": None,
            "mean_teacher_tie_probability_mass": None,
            "legal_action_violations": 0,
        }

    losses: list[float] = []
    confidences: list[float] = []
    margins: list[float] = []
    tie_probability_mass: list[float] = []
    top1 = 0
    tie_aware = 0
    for example in ordered:
        rows = _feature_rows(example)
        logits = [_dot(weights, row) for row in rows]
        probabilities = masked_softmax(logits, example.legal_mask)
        selected_probability = probabilities[example.selected_index]
        if selected_probability <= 0.0:
            raise StudentV0Error("selected legal action received zero probability")
        losses.append(-math.log(selected_probability))
        ordering = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
        predicted = ordering[0]
        top1 += int(predicted == example.selected_index)
        tie_aware += int(predicted in example.tie_indices)
        confidences.append(probabilities[predicted])
        second = probabilities[ordering[1]] if len(ordering) > 1 else 0.0
        margins.append(probabilities[predicted] - second)
        tie_probability_mass.append(sum(probabilities[index] for index in example.tie_indices))

    total = len(ordered)
    return {
        "decisions": total,
        "loss": sum(losses) / total,
        "top1_accuracy": top1 / total,
        "tie_aware_accuracy": tie_aware / total,
        "mean_confidence": sum(confidences) / total,
        "mean_margin": sum(margins) / total,
        "mean_teacher_tie_probability_mass": sum(tie_probability_mass) / total,
        "legal_action_violations": 0,
    }


def _ordered_examples(
    examples: Sequence[BCExample],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[BCExample, ...]:
    ordered = tuple(sorted(examples, key=lambda example: example.decision_signature))
    if not ordered and not allow_empty:
        raise StudentV0Error(f"{name} examples are required")
    signatures = [example.decision_signature for example in ordered]
    if len(signatures) != len(set(signatures)):
        raise StudentV0Error(f"{name} examples contain duplicate decision signatures")
    for example in ordered:
        if len(example.action_ids) != len(example.legal_mask) or not example.action_ids:
            raise StudentV0Error(f"{name} example has an invalid legal-action mask")
        if example.selected_index < 0 or example.selected_index >= len(example.action_ids):
            raise StudentV0Error(f"{name} example selected index is out of range")
        if not example.legal_mask[example.selected_index]:
            raise StudentV0Error(f"{name} example selected action is not legal")
        if not example.tie_indices or example.selected_index not in example.tie_indices:
            raise StudentV0Error(f"{name} example selected action is outside Teacher tie set")
    return ordered


def _feature_rows(example: BCExample) -> list[dict[str, float]]:
    state_features: dict[str, float] = {}
    _flatten(example.observation, "state", state_features)
    rows: list[dict[str, float]] = []
    for action_id, payload in zip(example.action_ids, example.action_payloads):
        action_features: dict[str, float] = {f"action.id={action_id}": 1.0}
        _flatten(payload, "action", action_features)
        row = dict(action_features)
        for state_key, state_value in state_features.items():
            for action_key, action_value in action_features.items():
                product = state_value * action_value
                if product != 0.0:
                    row[f"cross::{state_key}::{action_key}"] = product
        rows.append(row)
    return rows


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
            raise StudentV0Error(f"non-finite numeric policy feature at {prefix}")
        output[prefix] = numeric / (1.0 + abs(numeric))
        return
    if isinstance(value, str):
        output[f"{prefix}={value}"] = 1.0
        return
    if isinstance(value, dict):
        output[f"{prefix}.len"] = float(len(value)) / (1.0 + float(len(value)))
        for key in sorted(value):
            _flatten(value[key], f"{prefix}.{key}", output)
        return
    if isinstance(value, (list, tuple)):
        output[f"{prefix}.len"] = float(len(value)) / (1.0 + float(len(value)))
        for index, item in enumerate(value):
            _flatten(item, f"{prefix}[{index}]", output)
        return
    raise StudentV0Error(f"unsupported policy feature type at {prefix}: {type(value).__name__}")


def _dot(weights: Mapping[str, float], features: Mapping[str, float]) -> float:
    return sum(float(weights.get(key, 0.0)) * value for key, value in features.items())


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "STUDENT_V0_SCHEMA_VERSION",
    "StudentV0Config",
    "StudentV0Error",
    "StudentV0Result",
    "evaluate_student_v0",
    "train_student_v0",
]
