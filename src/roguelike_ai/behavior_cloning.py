"""Legal-action-masked behavior-cloning primitives.

The mask is part of the model contract: logits for actions that are not in
the observed legal-action set never receive probability mass or gradient.
This module is dependency-light so the artifact can be used by the existing
feature-based HybridPolicy runtime without introducing a second framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Sequence

import polars as pl

try:
    from .agents.hybrid_policy import FEATURE_SCHEMA_VERSION, make_artifact
    from .decision_training import (
        DECISION_DATASET_SCHEMA,
        DecisionTrainingConfig,
        _assemble_examples,
        _dataset_hash,
        _read_candidates,
        _read_decisions,
        _training_context,
    )
except ModuleNotFoundError:
    # The frozen STS1 v0 rebuild only needs masked_softmax. Keep that
    # dependency-light path importable without publishing unrelated legacy
    # HybridPolicy/decision-training modules into the public compute repo.
    FEATURE_SCHEMA_VERSION = None
    DECISION_DATASET_SCHEMA = None
    DecisionTrainingConfig = None
    make_artifact = None
    _assemble_examples = None
    _dataset_hash = None
    _read_candidates = None
    _read_decisions = None
    _training_context = None


class MaskedBehaviorCloningError(ValueError):
    """Raised when a behavior-cloning example violates its legality mask."""


def masked_softmax(logits: Sequence[float], legal_mask: Sequence[bool]) -> list[float]:
    """Return a numerically stable softmax with zero mass on illegal actions."""

    if len(logits) != len(legal_mask):
        raise MaskedBehaviorCloningError("logits and legal_mask must have equal length")
    if not logits:
        raise MaskedBehaviorCloningError("masked softmax requires at least one candidate")
    if not any(legal_mask):
        raise MaskedBehaviorCloningError("legal-action mask must contain a legal candidate")
    for value in logits:
        if not math.isfinite(float(value)):
            raise MaskedBehaviorCloningError("logits must be finite")

    legal_values = [float(value) for value, legal in zip(logits, legal_mask) if legal]
    maximum = max(legal_values)
    exponentials = [
        math.exp(float(value) - maximum) if legal else 0.0
        for value, legal in zip(logits, legal_mask)
    ]
    denominator = sum(exponentials)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise MaskedBehaviorCloningError("masked softmax denominator is not finite and positive")
    return [value / denominator for value in exponentials]


def masked_cross_entropy(
    logits: Sequence[float],
    *,
    selected_index: int,
    legal_mask: Sequence[bool],
) -> float:
    """Return NLL for one selected legal action under a masked softmax."""

    if selected_index < 0 or selected_index >= len(logits):
        raise MaskedBehaviorCloningError("selected action index is outside candidate set")
    if selected_index >= len(legal_mask) or not legal_mask[selected_index]:
        raise MaskedBehaviorCloningError("selected action is not legal in the observed mask")
    probabilities = masked_softmax(logits, legal_mask)
    probability = probabilities[selected_index]
    if probability <= 0.0 or not math.isfinite(probability):
        raise MaskedBehaviorCloningError("selected legal action received no probability")
    return -math.log(probability)


@dataclass(frozen=True)
class MaskedBehaviorCloningConfig:
    """Deterministic compact-dataset configuration for masked BC."""

    dataset_dir: Path
    model_dir: Path
    epochs: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 128
    weight_decay: float = 1e-4
    seed: int = 0
    agent_name: str | None = None
    character: str | None = None
    outcomes: tuple[str, ...] = ()
    max_decisions: int = 0
    overwrite: bool = False


@dataclass(frozen=True)
class MaskedBehaviorCloningResult:
    model_dir: str
    metrics: dict[str, object]


def train_masked_behavior_cloning(config: MaskedBehaviorCloningConfig) -> MaskedBehaviorCloningResult:
    """Train a legal-candidate scorer with masked cross-entropy.

    The compact candidate table is the legal-action mask: every candidate
    row is an observed legal candidate and exactly one row must be selected.
    Invalid or ambiguous decisions are excluded with reason codes by the
    shared decision-dataset assembler; no selected action is added to the
    mask during training.
    """

    if DecisionTrainingConfig is None or make_artifact is None:
        raise ModuleNotFoundError(
            "legacy HybridPolicy decision-training modules are required for "
            "train_masked_behavior_cloning but not for frozen Student v0 rebuild"
        )

    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config.learning_rate <= 0.0 or not math.isfinite(config.learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if config.weight_decay < 0.0 or not math.isfinite(config.weight_decay):
        raise ValueError("weight_decay must be finite and non-negative")
    if config.max_decisions < 0:
        raise ValueError("max_decisions must be non-negative")

    dataset_dir = Path(config.dataset_dir)
    model_dir = Path(config.model_dir)
    if model_dir.exists() and any(model_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"model directory is not empty: {model_dir}")

    assembly_config = DecisionTrainingConfig(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        agent_name=config.agent_name,
        character=config.character,
        outcomes=config.outcomes,
        max_decisions=config.max_decisions,
    )
    decisions = _read_decisions(dataset_dir / "decisions.parquet")
    candidates = _read_candidates(dataset_dir / "candidates.parquet")
    examples, skipped = _assemble_examples(decisions, candidates, assembly_config)
    if not examples:
        raise MaskedBehaviorCloningError("no usable decisions remain after dataset validation")

    train_examples = tuple(example for example in examples if example.split == "train")
    validation_examples = tuple(example for example in examples if example.split == "validation")
    test_examples = tuple(example for example in examples if example.split == "test")
    if not train_examples:
        raise MaskedBehaviorCloningError("masked BC dataset has no train split")

    weights: dict[str, float] = {}
    ordered_train = tuple(sorted(train_examples, key=lambda item: item.decision_id))
    for epoch in range(config.epochs):
        # A deterministic permutation avoids Parquet row-order dependence
        # while still giving each epoch a different bounded batch order.
        order = list(ordered_train)
        random.Random(config.seed + epoch).shuffle(order)
        gradient: dict[str, float] = {}
        batch_count = 0
        for example in order:
            selected_index = next(index for index, candidate in enumerate(example.candidates) if candidate[2])
            feature_rows = [_merged_features(example.state_features, candidate[1]) for candidate in example.candidates]
            logits = [_dot(weights, row) for row in feature_rows]
            legal_mask = [True] * len(logits)
            probabilities = masked_softmax(logits, legal_mask)
            for index, row in enumerate(feature_rows):
                delta = probabilities[index] - float(index == selected_index)
                for key, value in row.items():
                    gradient[key] = gradient.get(key, 0.0) + delta * value
            batch_count += 1
            if batch_count >= config.batch_size:
                _apply_gradient(weights, gradient, batch_count, config.learning_rate, config.weight_decay)
                gradient = {}
                batch_count = 0
        if batch_count:
            _apply_gradient(weights, gradient, batch_count, config.learning_rate, config.weight_decay)

    dataset_hash = _dataset_hash(dataset_dir)
    train_metrics = _evaluate_masked(train_examples, weights)
    validation_metrics = _evaluate_masked(validation_examples, weights)
    test_metrics = _evaluate_masked(test_examples, weights)
    source_context = _training_context(dataset_dir, assembly_config)
    metrics: dict[str, object] = {
        "algorithm": "behavior-cloning-legal-action-masked-softmax",
        "dataset_schema": DECISION_DATASET_SCHEMA,
        "dataset_hash": dataset_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "legal_action_mask": "observed_candidate_rows_only",
        "legal_action_violations": 0,
        "episodes_seen": len({str(row.get("episode_id")) for row in decisions if row.get("episode_id") is not None}),
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "skipped_decisions": sum(skipped.values()),
        "skipped_reasons": dict(sorted(skipped.items())),
        "promotion_decision": "NOT_VERIFIED",
        "training_context": source_context,
    }
    artifact = make_artifact(
        weights,
        planner_objective="combined",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_context=source_context,
    )
    artifact.update(
        {
            "algorithm": metrics["algorithm"],
            "schema_version": 1,
            "decision_dataset_schema": DECISION_DATASET_SCHEMA,
            "decision_dataset_hash": dataset_hash,
            "legal_action_mask": True,
        }
    )
    _write_bc_outputs(model_dir, artifact, metrics, dataset_dir)
    return MaskedBehaviorCloningResult(model_dir=str(model_dir), metrics=metrics)


def _merged_features(state: dict[str, float] | Sequence[tuple[str, float]], action: dict[str, float]) -> dict[str, float]:
    if isinstance(state, dict):
        state_items = state.items()
    else:
        state_items = state
    merged = {f"state::{key}": float(value) for key, value in state_items}
    merged.update({f"action::{key}": float(value) for key, value in action.items()})
    return merged


def _dot(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(key, 0.0) * value for key, value in features.items())


def _apply_gradient(weights: dict[str, float], gradient: dict[str, float], count: int, learning_rate: float, weight_decay: float) -> None:
    scale = learning_rate / float(count)
    for key in set(weights) | set(gradient):
        value = weights.get(key, 0.0)
        updated = value * (1.0 - learning_rate * weight_decay) - scale * gradient.get(key, 0.0)
        if math.isfinite(updated) and updated != 0.0:
            weights[key] = updated
        else:
            weights.pop(key, None)


def _evaluate_masked(examples: Sequence[object], weights: dict[str, float]) -> dict[str, object]:
    if not examples:
        return {"decisions": 0, "loss": None, "top1_accuracy": None, "top3_accuracy": None, "legal_action_violations": 0}
    losses: list[float] = []
    top1 = 0
    top3 = 0
    for example in examples:
        selected_index = next(index for index, candidate in enumerate(example.candidates) if candidate[2])
        rows = [_merged_features(example.state_features, candidate[1]) for candidate in example.candidates]
        logits = [_dot(weights, row) for row in rows]
        mask = [True] * len(logits)
        losses.append(masked_cross_entropy(logits, selected_index=selected_index, legal_mask=mask))
        ordering = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
        top1 += int(ordering[0] == selected_index)
        top3 += int(selected_index in ordering[:3])
    total = len(examples)
    return {
        "decisions": total,
        "loss": sum(losses) / total,
        "top1_accuracy": top1 / total,
        "top3_accuracy": top3 / total,
        "legal_action_violations": 0,
    }


def _write_bc_outputs(model_dir: Path, artifact: dict[str, object], metrics: dict[str, object], dataset_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = dataset_dir / "decision-manifest.json"
    source_manifest: object = {}
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "behavior-cloning-dataset-manifest-v1",
        "source_dataset": source_manifest,
        "dataset_hash": metrics["dataset_hash"],
        "legal_action_mask": "observed_candidate_rows_only",
        "promotion_decision": "NOT_VERIFIED",
    }
    config = {
        "algorithm": metrics["algorithm"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_schema": DECISION_DATASET_SCHEMA,
        "legal_action_mask": True,
        "training_context": metrics["training_context"],
    }
    card = """# Legal-action masked behavior cloning\n\nThis artifact scores only candidates present in the observed legal-action mask.\nOffline metrics do not establish strict Bridge victory; promotion remains NOT_VERIFIED.\n"""
    payloads = {
        "model.json": artifact,
        "config.json": config,
        "metrics.json": metrics,
        "dataset-manifest.json": manifest,
        "model-card.md": card,
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{model_dir.name}.tmp-", dir=str(model_dir.parent)))
    try:
        for name, payload in payloads.items():
            if isinstance(payload, str):
                (staging / name).write_text(payload, encoding="utf-8")
            else:
                (staging / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        for name in payloads:
            (model_dir / name).write_bytes((staging / name).read_bytes())
        checksums = {}
        for name in payloads:
            digest = hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
            checksums[name] = digest
        (model_dir / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()


__all__ = [
    "MaskedBehaviorCloningConfig",
    "MaskedBehaviorCloningError",
    "MaskedBehaviorCloningResult",
    "masked_cross_entropy",
    "masked_softmax",
    "train_masked_behavior_cloning",
]
