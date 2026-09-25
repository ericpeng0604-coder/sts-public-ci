"""Strict evaluation-only frozen-holdout comparison for the fixed STS1 Student v0.

The sequence is deliberately fail-closed:
1. verify the exact frozen Student source identity;
2. rebuild training/validation data from train-eligible artifacts only;
3. reproduce the exact fixed Student candidate and require its known model SHA;
4. only then open external holdout artifacts and compare Student, simple, random,
   and Teacher actions without changing any model/config/feature/threshold.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence

from roguelike_ai.behavior_cloning import masked_softmax
from roguelike_ai.sts1_training import DatasetBuildConfig, build_dataset, formal_teacher_provenance, load_bc_examples
from roguelike_ai.sts1_training import bc as bc_module
from roguelike_ai.sts1_training import student_v0 as student_module
from roguelike_ai.sts1_training.teacher_capture import capture_frozen_search_record


EXPECTED_CAPTURE_SHA = "775dd9baa3b1c8a4841af490b7fdfeb3f384fc47"
EXPECTED_DATASET_HASH = "4509f9c48206606638f24f264c0dbc471b1cc46c60f2633b3659f72cf7c6ccea"
EXPECTED_STUDENT_SOURCE_SHA256 = "06e206b7f04e6e7b77ad1fc8c35ba1bcdca1c978596fbb176933323e3940aed3"
EXPECTED_CONFIG_HASH = "906afcaec38a2050f48e28f2351745408bbc0c72607704dbce2e134cdff4192c"
EXPECTED_MODEL_SHA256 = "e15604b95247615a6d424f834e8b8a1e9fd680af4fe9e22a6754372027e89513"
EXPECTED_TRAIN_ELIGIBLE = 96
EXPECTED_EXTERNAL_HOLDOUT = 96
EXPECTED_SPLIT_COUNTS = {"train": 83, "validation": 7, "holdout": 6}
SIGN_TEST_ALPHA = 0.05


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_dataset_pilot(repo_root: Path) -> ModuleType:
    path = repo_root / "scripts" / "sts1" / "sts1_phase2_dataset_factory_pilot.py"
    spec = importlib.util.spec_from_file_location("sts1_phase2_dataset_factory_pilot_for_holdout_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Dataset Factory pilot")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture_specs(
    pilot: ModuleType,
    specs: Sequence[Any],
    *,
    producer: Any,
    worker_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for artifact in specs:
        rows, case_map, summary = pilot._load_artifact(artifact)
        summaries.append({"artifact": artifact.name, "summary": summary})
        for teacher_decision in rows:
            case_id = teacher_decision.get("case_id")
            if not isinstance(case_id, str) or case_id not in case_map:
                raise RuntimeError(f"artifact {artifact.artifact_id} decision has no matching suite case")
            case = case_map[case_id]
            public_state = case.get("public_state")
            if not isinstance(public_state, dict):
                raise RuntimeError(f"artifact {artifact.artifact_id} case {case_id} missing public_state")
            if case.get("decision_signature") != teacher_decision.get("decision_signature"):
                raise RuntimeError(f"artifact {artifact.artifact_id} case {case_id} signature mismatch")
            if case.get("source_seed_provenance_only") != teacher_decision.get("source_seed_provenance_only"):
                raise RuntimeError(f"artifact {artifact.artifact_id} case {case_id} seed mismatch")
            record = capture_frozen_search_record(
                public_state=public_state,
                teacher_decision=teacher_decision,
                seed=teacher_decision["source_seed_provenance_only"],
                run_id=case_id,
                decision_index=0,
                provenance=producer,
                worker_id=worker_id,
                floor=public_state.get("floor") if isinstance(public_state.get("floor"), int) else None,
                combat_id=f"{case_id}:combat:1",
            )
            records.append(record)
            diagnostics.append(
                {
                    "artifact": artifact.name,
                    "case_id": case_id,
                    "decision_signature": record["decision_signature"],
                    "simple_action_id": teacher_decision.get("simple_action_id"),
                    "random_action_id": teacher_decision.get("random_action_id"),
                    "search_unique_best_action_id": teacher_decision.get("search_unique_best_action_id"),
                    "search_tie_ids": teacher_decision.get("search_tie_ids"),
                }
            )
    return records, diagnostics, summaries


def _reconstruct_frozen_weights(
    train_examples: Sequence[Any],
    config: student_module.StudentV0Config,
) -> dict[str, float]:
    """Mirror only the frozen optimizer so evaluation can access the exact weights.

    The harness verifies the reconstructed model payload SHA against the already
    frozen candidate SHA before any holdout is opened.
    """

    train = student_module._ordered_examples(train_examples, name="train")
    weights: dict[str, float] = {}
    for _ in range(config.epochs):
        gradient: dict[str, float] = {}
        for example in train:
            rows = student_module._feature_rows(example)
            logits = [student_module._dot(weights, row) for row in rows]
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
                raise RuntimeError("frozen Student reconstruction produced non-finite weight")
            if updated == 0.0:
                weights.pop(key, None)
            else:
                weights[key] = updated
    return weights


def _model_sha(weights: Mapping[str, float], config: student_module.StudentV0Config) -> str:
    config_payload = asdict(config)
    config_hash = student_module._sha256_json(config_payload)
    payload = {
        "schema_version": student_module.STUDENT_V0_SCHEMA_VERSION,
        "config_hash": config_hash,
        "weights": {key: weights[key] for key in sorted(weights)},
    }
    return student_module._sha256_json(payload)


def _freeze_candidate(repo_root: Path, pilot: ModuleType, dataset_root: Path) -> tuple[Any, dict[str, float], Any, dict[str, Any]]:
    student_path = repo_root / "src" / "roguelike_ai" / "sts1_training" / "student_v0.py"
    source_sha = _sha256_file(student_path)
    if source_sha != EXPECTED_STUDENT_SOURCE_SHA256:
        raise RuntimeError(f"Student source identity drift: {source_sha}")

    train_specs = tuple(spec for spec in pilot.ARTIFACTS if spec.train_eligible)
    if not train_specs or any(not spec.train_eligible for spec in train_specs):
        raise RuntimeError("train-only artifact selection failed closed")
    producer = formal_teacher_provenance(
        teacher_id=pilot.FORMAL_TEACHER_ID,
        capture_code_sha=EXPECTED_CAPTURE_SHA,
        worker_generation=1,
    )
    train_records, _, _ = _capture_specs(
        pilot,
        train_specs,
        producer=producer,
        worker_id="sts1-phase2-dataset-factory-pilot",
    )
    if len(train_records) != EXPECTED_TRAIN_ELIGIBLE:
        raise RuntimeError(f"train-eligible source count drift: {len(train_records)}")

    config = DatasetBuildConfig(
        split_salt=pilot.PILOT_SPLIT_SALT,
        shard_size=pilot.PILOT_SHARD_SIZE,
        require_all_splits=True,
    )
    manifest = build_dataset(train_records, dataset_root, expected_provenance=producer, config=config)
    if manifest.get("dataset_hash") != EXPECTED_DATASET_HASH:
        raise RuntimeError(f"training dataset hash drift: {manifest.get('dataset_hash')}")
    if manifest.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"training split count drift: {manifest.get('split_counts')}")
    if manifest.get("holdout_training_allowed") is not False:
        raise RuntimeError("training manifest unexpectedly permits holdout access")

    train_examples = load_bc_examples(dataset_root, "train")
    validation_examples = load_bc_examples(dataset_root, "validation")
    fixed_config = student_module.StudentV0Config(epochs=40, learning_rate=0.05, weight_decay=1e-4)
    result = student_module.train_student_v0(train_examples, validation_examples, config=fixed_config)
    if result.config_hash != EXPECTED_CONFIG_HASH:
        raise RuntimeError(f"Student config hash drift: {result.config_hash}")
    if result.model_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"Student model hash drift: {result.model_sha256}")
    if result.holdout_opened:
        raise RuntimeError("Student training path opened holdout")

    weights = _reconstruct_frozen_weights(train_examples, fixed_config)
    reconstructed_sha = _model_sha(weights, fixed_config)
    if reconstructed_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"reconstructed Student model hash drift: {reconstructed_sha}")
    if student_module.evaluate_student_v0(train_examples, weights) != result.train:
        raise RuntimeError("reconstructed Student train metrics disagree with frozen candidate")
    if student_module.evaluate_student_v0(validation_examples, weights) != result.validation:
        raise RuntimeError("reconstructed Student validation metrics disagree with frozen candidate")
    frozen = {
        "student_source_sha256": source_sha,
        "student_config_hash": result.config_hash,
        "student_model_sha256": result.model_sha256,
        "feature_count": result.feature_count,
        "train_metrics": result.train,
        "validation_metrics": result.validation,
        "dataset_hash": manifest["dataset_hash"],
        "split_counts": manifest["split_counts"],
        "holdout_opened_before_freeze": False,
    }
    return result, weights, producer, frozen


def _example_from_record(record: Mapping[str, Any]) -> Any:
    payload = dict(record)
    payload["split"] = "holdout"
    return bc_module._to_example(payload, expected_split="holdout")


def _predict_student(example: Any, weights: Mapping[str, float]) -> str:
    rows = student_module._feature_rows(example)
    logits = [student_module._dot(weights, row) for row in rows]
    probabilities = masked_softmax(logits, example.legal_mask)
    ordering = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    return example.action_ids[ordering[0]]


def _baseline_metrics(rows: Sequence[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        raise RuntimeError("empty holdout evaluation")
    exact = 0
    tie_aware = 0
    illegal = 0
    unique_total = 0
    unique_correct = 0
    for row in rows:
        predicted = row[prediction_key]
        legal_ids = row["legal_action_ids"]
        teacher_selected = row["teacher_selected_action_id"]
        teacher_ties = row["teacher_tie_action_ids"]
        if predicted not in legal_ids:
            illegal += 1
            continue
        exact += int(predicted == teacher_selected)
        tie_aware += int(predicted in teacher_ties)
        if len(teacher_ties) == 1:
            unique_total += 1
            unique_correct += int(predicted == teacher_ties[0])
    return {
        "decisions": total,
        "teacher_agreement": exact / total,
        "top1_accuracy": exact / total,
        "tie_aware_accuracy": tie_aware / total,
        "unique_best_decisions": unique_total,
        "unique_best_accuracy": unique_correct / unique_total if unique_total else None,
        "illegal_action_count": illegal,
        "illegal_action_rate": illegal / total,
        "teacher_agreement_definition": "exact match to canonical frozen Teacher selected action",
    }


def _paired_correctness(rows: Sequence[dict[str, Any]], left_key: str, right_key: str, *, tie_aware: bool) -> dict[str, Any]:
    left_wins = 0
    right_wins = 0
    both_correct = 0
    both_wrong = 0
    for row in rows:
        teacher_targets = set(row["teacher_tie_action_ids"]) if tie_aware else {row["teacher_selected_action_id"]}
        left = row[left_key] in teacher_targets
        right = row[right_key] in teacher_targets
        if left and not right:
            left_wins += 1
        elif right and not left:
            right_wins += 1
        elif left:
            both_correct += 1
        else:
            both_wrong += 1
    discordant = left_wins + right_wins
    if discordant == 0 or left_wins <= right_wins:
        one_sided_p = 1.0
    else:
        numerator = sum(math.comb(discordant, k) for k in range(left_wins, discordant + 1))
        one_sided_p = numerator / float(2**discordant)
    return {
        "left_wins": left_wins,
        "right_wins": right_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "one_sided_exact_sign_p": one_sided_p,
    }


def _summary_outcome_evidence(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tokens = ("outcome", "win", "floor", "hp", "quality", "determin", "case_count", "simple", "random", "search")
    artifact_evidence: list[dict[str, Any]] = []
    available = False
    for entry in summaries:
        summary = entry["summary"]
        selected: dict[str, Any] = {}
        for key, value in summary.items():
            lowered = str(key).lower()
            if any(token in lowered for token in tokens) and isinstance(value, (str, int, float, bool, type(None))):
                selected[str(key)] = value
        if any(token in key.lower() for key in selected for token in ("outcome", "win", "floor", "hp")):
            available = True
        artifact_evidence.append({"artifact": entry["artifact"], "summary": selected})
    return {
        "gameplay_outcome_fields_available": available,
        "note": (
            "Preserved source summaries include gameplay/outcome-like scalar fields."
            if available
            else "These frozen artifacts are decision-quality holdouts; no separate gameplay outcome field was available in their summaries."
        ),
        "artifacts": artifact_evidence,
    }


def _evaluate_holdout(pilot: ModuleType, producer: Any, weights: Mapping[str, float]) -> dict[str, Any]:
    holdout_specs = tuple(spec for spec in pilot.ARTIFACTS if not spec.train_eligible)
    records, diagnostics, summaries = _capture_specs(
        pilot,
        holdout_specs,
        producer=producer,
        worker_id="sts1-phase2-student-v0-holdout-eval",
    )
    if len(records) != EXPECTED_EXTERNAL_HOLDOUT or len(diagnostics) != EXPECTED_EXTERNAL_HOLDOUT:
        raise RuntimeError(f"external holdout count drift: {len(records)}")

    diagnostic_by_signature = {row["decision_signature"]: row for row in diagnostics}
    if len(diagnostic_by_signature) != len(diagnostics):
        raise RuntimeError("external holdout contains duplicate decision signatures")

    rows: list[dict[str, Any]] = []
    for record in records:
        example = _example_from_record(record)
        diagnostic = diagnostic_by_signature[example.decision_signature]
        simple_action = diagnostic.get("simple_action_id")
        random_action = diagnostic.get("random_action_id")
        if not isinstance(simple_action, str) or not isinstance(random_action, str):
            raise RuntimeError("external holdout is missing frozen simple/random baseline actions")
        legal_ids = tuple(example.action_ids)
        if simple_action not in legal_ids or random_action not in legal_ids:
            raise RuntimeError("frozen simple/random baseline emitted an illegal holdout action")
        teacher_selected = legal_ids[example.selected_index]
        teacher_ties = tuple(legal_ids[index] for index in example.tie_indices)
        rows.append(
            {
                "decision_signature": example.decision_signature,
                "legal_action_ids": legal_ids,
                "teacher_selected_action_id": teacher_selected,
                "teacher_tie_action_ids": teacher_ties,
                "student_action_id": _predict_student(example, weights),
                "simple_action_id": simple_action,
                "random_action_id": random_action,
                "teacher_action_id": teacher_selected,
            }
        )

    student = _baseline_metrics(rows, "student_action_id")
    simple = _baseline_metrics(rows, "simple_action_id")
    random = _baseline_metrics(rows, "random_action_id")
    teacher = _baseline_metrics(rows, "teacher_action_id")
    paired_top1 = _paired_correctness(rows, "student_action_id", "simple_action_id", tie_aware=False)
    paired_tie = _paired_correctness(rows, "student_action_id", "simple_action_id", tie_aware=True)
    top1_delta_pp = 100.0 * (student["top1_accuracy"] - simple["top1_accuracy"])
    tie_delta_pp = 100.0 * (student["tie_aware_accuracy"] - simple["tie_aware_accuracy"])
    clearly_better = (
        student["illegal_action_count"] == 0
        and top1_delta_pp > 0.0
        and tie_delta_pp > 0.0
        and paired_tie["left_wins"] > paired_tie["right_wins"]
        and paired_tie["one_sided_exact_sign_p"] < SIGN_TEST_ALPHA
    )
    return {
        "external_holdout_decisions": len(rows),
        "comparators": {
            "student_v0": student,
            "simple_heuristic": simple,
            "random_legal": random,
            "teacher": teacher,
        },
        "student_vs_simple": {
            "top1_delta_percentage_points": top1_delta_pp,
            "tie_aware_delta_percentage_points": tie_delta_pp,
            "paired_top1": paired_top1,
            "paired_tie_aware": paired_tie,
            "clear_better_rule_frozen_before_holdout": {
                "alpha": SIGN_TEST_ALPHA,
                "requires_positive_top1_delta": True,
                "requires_positive_tie_aware_delta": True,
                "requires_zero_illegal_actions": True,
                "requires_paired_tie_aware_one_sided_exact_sign_p_below_alpha": True,
            },
            "clearly_better": clearly_better,
        },
        "outcome_evidence": _summary_outcome_evidence(summaries),
        "usefulness_decision": "GO_FIXED_STUDENT_BEATS_SIMPLE" if clearly_better else "NO_GO_INSUFFICIENT_HOLDOUT_SUPERIORITY",
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    if student_module._sha256_json(asdict(student_module.StudentV0Config(epochs=40, learning_rate=0.05, weight_decay=1e-4))) != EXPECTED_CONFIG_HASH:
        raise RuntimeError("frozen config identity drift before artifact access")
    pilot = _load_dataset_pilot(repo_root)
    with tempfile.TemporaryDirectory(prefix="sts1-student-v0-holdout-eval-") as directory:
        dataset_root = Path(directory) / "train-dataset"
        candidate, weights, producer, frozen = _freeze_candidate(repo_root, pilot, dataset_root)
        if candidate.model_sha256 != EXPECTED_MODEL_SHA256:
            raise RuntimeError("candidate was not frozen before holdout access")
        holdout = _evaluate_holdout(pilot, producer, weights)

    result = {
        "result": "STS1_STUDENT_V0_FROZEN_HOLDOUT_EVALUATION_PASS",
        "candidate_frozen_before_holdout": True,
        "candidate": frozen,
        "holdout": holdout,
        "no_tuning_after_holdout": True,
        "ppo": "LOCKED",
        "promotion_decision": "NOT_VERIFIED" if holdout["usefulness_decision"].startswith("GO_") else "NO_GO",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
