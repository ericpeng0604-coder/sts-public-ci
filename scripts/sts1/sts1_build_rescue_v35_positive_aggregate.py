#!/usr/bin/env python3
"""Validate v3.5 Phase-B mining outputs and aggregate only proven full-win teachers.

Missing or malformed per-seed outputs are reported as isolated failures. They
never become training data and do not block aggregation of other valid seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

SIM_SCHEMA = "sts1-phase3-a0-simulator-v1"
SIM_PROTOCOL = "sts1-phase3-a0-full-run-v1"
TEACHER_SCHEMA = "sts1-armg-strategy-branch-dataset-v1"
TEACHER_SOURCE = "sts1-build-rescue-new-win-miner-v24"
FULL_WIN = {"NEW_FULL_WIN_ONE_STEP", "NEW_FULL_WIN_TWO_STEP"}
KNOWN_STATUSES = FULL_WIN | {
    "BASE_REPLAY_FAILURE",
    "BASE_50K_FAILURE",
    "BASE_UNEXPECTED_WIN",
    "COMBAT_ONLY_RESCUED_50K",
    "NO_REPLAYABLE_BUILD_STATE",
    "NO_NEW_WIN_RESCUE",
    "NEW_BOSS_RESCUE_ONE_STEP",
    "NEW_BOSS_RESCUE_TWO_STEP",
    "SEED_SEARCH_INCOMPLETE",
    "SEED_SEARCH_FAILURE",
}
SAFETY_COUNTERS = (
    "illegal_action_count",
    "crash_count",
    "timeout_count",
    "remote_error_count",
    "fallback_count",
)


def _finite_numeric(values: list[Any]) -> bool:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            if not math.isfinite(float(value)):
                return False
        except (OverflowError, TypeError, ValueError):
            return False
    return True


def _seed_list(path: Path, expected_count: int) -> tuple[list[int], set[int]]:
    values = [int(x) for x in path.read_text(encoding="utf-8").split()]
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} unique seeds")
    return values, set(values)


def _safe_sim(result: Any, seed: int) -> bool:
    if not isinstance(result, dict):
        return False
    try:
        return (
            result.get("schema_version") == SIM_SCHEMA
            and result.get("phase_protocol") == SIM_PROTOCOL
            and result.get("result") == "PASS_SIMULATOR_COMPLETE_RUN"
            and result.get("error") is None
            and str(result.get("seed")) == str(seed)
            and int(result.get("simulator_seed_long", -1)) == seed
            and result.get("seed_contract") == "heldout_internal"
            and result.get("combat_policy") == "mcts_2000"
            and result.get("noncombat_policy") == "armg"
            and str(result.get("outcome", "")).lower() in {"victory", "defeat"}
            and all(int(result.get(k, 0) or 0) == 0 for k in SAFETY_COUNTERS)
        )
    except (TypeError, ValueError):
        return False


def _passes_target(result: dict[str, Any], target: int) -> bool:
    try:
        return (
            str(result.get("outcome", "")).lower() == "victory"
            or int(result.get("final_floor") or result.get("max_floor") or 0) > target
        )
    except (TypeError, ValueError):
        return False


def _teacher_error(row: Any, seed: int, status: str) -> str | None:
    if not isinstance(row, dict):
        return "teacher row is not an object"
    if row.get("schema_version") != TEACHER_SCHEMA or row.get("source") != TEACHER_SOURCE:
        return "teacher schema/source mismatch"
    if row.get("type") != "v24_new_win_full_victory":
        return "teacher is not a verified full-victory label"
    try:
        row_seed = int(row.get("seed", -1))
    except (TypeError, ValueError):
        return "teacher seed is malformed"
    if row_seed != seed:
        return "teacher seed mismatch"
    obs = row.get("obs")
    descs = row.get("descs")
    if not isinstance(obs, list) or len(obs) != 412:
        return "teacher observation dimension mismatch"
    if not isinstance(descs, list) or len(descs) < 2 or any(
        not isinstance(desc, list) or len(desc) != 368 for desc in descs
    ):
        return "teacher candidate descriptor dimension mismatch"
    vectors = [obs, *descs]
    if any(not _finite_numeric(vector) for vector in vectors):
        return "teacher features contain a non-numeric or non-finite value"
    try:
        current = int(row["current_armg_index"])
        target = int(row["teacher_best_index"])
        if not isinstance(row["target_probs"], list):
            return "teacher target distribution is malformed"
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in row["target_probs"]):
            return "teacher target distribution is non-numeric"
        probs = [float(x) for x in row["target_probs"]]
    except (KeyError, OverflowError, TypeError, ValueError):
        return "teacher action/target fields are malformed"
    if not (0 <= current < len(descs) and 0 <= target < len(descs)) or target == current:
        return "teacher did not change the selected action"
    if any(not math.isfinite(p) for p in probs) or len(probs) != len(descs) or abs(sum(probs) - 1.0) > 1e-6:
        return "teacher target distribution is malformed"
    if any(abs(p - (1.0 if i == target else 0.0)) > 1e-6 for i, p in enumerate(probs)):
        return "teacher target is not one-hot on the verified action"
    if row.get("original_choice") == row.get("teacher_choice"):
        return "teacher semantic action did not change"
    if row.get("combat_policy") != "mcts_2000":
        return "teacher combat-policy mismatch"
    if row.get("confirmation_policy") != "boss_mcts_10000_and_50000":
        return "teacher confirmation-policy mismatch"
    try:
        target_floor = int(row["target_boss_floor"])
        floor = int(row["floor"])
        intervention_step = int(row["intervention_step"])
    except (KeyError, TypeError, ValueError):
        return "teacher floor/intervention fields are malformed"
    if floor < 0:
        return "teacher floor is invalid"
    if not _safe_sim(row.get("boss_10k"), seed) or not _passes_target(row["boss_10k"], target_floor):
        return "teacher 10k counterfactual was not a safe target rescue"
    if not _safe_sim(row.get("boss_50k"), seed) or str(row["boss_50k"].get("outcome", "")).lower() != "victory":
        return "teacher 50k counterfactual was not a safe full win"
    required_step = 2 if status == "NEW_FULL_WIN_TWO_STEP" else 1
    if intervention_step not in ({1, 2} if required_step == 2 else {1}):
        return "teacher intervention-step mismatch"
    return None


def _load_teacher_file(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{number}: teacher row is not an object")
        rows.append(value)
    return rows


def aggregate(results_dir: Path, screen_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = json.loads((screen_dir / "seed-ledger.json").read_text(encoding="utf-8"))
    if ledger.get("schema") != "sts1-v35-disjoint-seed-ledger-v1":
        raise ValueError("unsupported v3.5 Phase-B seed ledger schema")
    expected_counts = {"train300": 300, "phaseb_gate50": 50, "phaseb_fresh100": 100, "phaseb_fresh500": 500}
    sets: dict[str, tuple[list[int], set[int]]] = {}
    for name, count in expected_counts.items():
        path = screen_dir / f"{name}.txt"
        raw = path.read_bytes()
        values, ids = _seed_list(path, count)
        digest = hashlib.sha256(raw).hexdigest()
        expected = ledger.get("sets", {}).get(name, {})
        if expected.get("size") != count or expected.get("sha256") != digest:
            raise ValueError(f"{name} does not match the frozen ledger")
        sets[name] = (values, ids)

    priority_values, priority = _seed_list(screen_dir / "priority80.txt", 80)
    training = sets["train300"][1]
    gate50 = sets["phaseb_gate50"][1]
    fresh100 = sets["phaseb_fresh100"][1]
    fresh500 = sets["phaseb_fresh500"][1]
    if not priority.issubset(training):
        raise ValueError("priority80 contains a seed outside train160")
    evaluation_sets = [gate50, fresh100, fresh500]
    if any(training & heldout for heldout in evaluation_sets):
        raise ValueError("training seeds overlap formal evaluation seeds")
    if gate50 & fresh100 or gate50 & fresh500 or fresh100 & fresh500:
        raise ValueError("formal evaluation sets overlap")
    if int(ledger.get("overlap_with_protected", -1)) != 0 or int(ledger.get("pairwise_overlap", -1)) != 0:
        raise ValueError("seed ledger reports an overlap")

    expected = set(priority)
    files = sorted(results_dir.rglob("result-*.json"))
    by_seed: dict[int, list[Path]] = {}
    unassigned: list[str] = []
    for path in files:
        try:
            seed = int(path.stem.removeprefix("result-"))
        except ValueError:
            unassigned.append(str(path))
            continue
        by_seed.setdefault(seed, []).append(path)

    failed: dict[int, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    trial_failure_kinds: Counter[str] = Counter()
    verified_rows: list[dict[str, Any]] = []
    safe_base_result_count = 0
    for seed in sorted(expected):
        paths = by_seed.get(seed, [])
        if len(paths) != 1:
            failed[seed] = {
                "status": "MISSING_RESULT" if not paths else "DUPLICATE_RESULT",
                "paths": [str(p) for p in paths],
            }
            continue
        path = paths[0]
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failed[seed] = {"status": "INVALID_RESULT_JSON", "error": str(exc)}
            continue
        if not isinstance(result, dict):
            failed[seed] = {"status": "RESULT_SEED_MISMATCH"}
            continue
        try:
            result_seed = int(result.get("seed", -1))
        except (TypeError, ValueError):
            failed[seed] = {"status": "RESULT_SEED_MISMATCH"}
            continue
        if result_seed != seed:
            failed[seed] = {"status": "RESULT_SEED_MISMATCH"}
            continue
        status = str(result.get("status", ""))
        if status not in KNOWN_STATUSES:
            failed[seed] = {"status": "UNKNOWN_RESULT_STATUS", "observed": status}
            continue
        status_counts[status] += 1
        for failure in result.get("failures") or []:
            kind = failure.get("kind", "unknown") if isinstance(failure, dict) else "malformed_failure_row"
            trial_failure_kinds[str(kind)] += 1

        teacher_paths = list(path.parent.glob(f"teacher-{seed}.jsonl"))
        if len(teacher_paths) != 1:
            failed[seed] = {
                "status": "MISSING_OR_DUPLICATE_TEACHER_FILE",
                "teacher_files": [str(p) for p in teacher_paths],
            }
            continue
        try:
            teacher_file_rows = _load_teacher_file(teacher_paths[0])
        except (OSError, ValueError) as exc:
            failed[seed] = {"status": "INVALID_TEACHER_JSONL", "error": str(exc)}
            continue
        result_rows = result.get("teachers") or []
        if json.dumps(teacher_file_rows, sort_keys=True) != json.dumps(result_rows, sort_keys=True):
            failed[seed] = {"status": "TEACHER_FILE_RESULT_MISMATCH"}
            continue

        if status == "BASE_REPLAY_FAILURE":
            if result_rows:
                failed[seed] = {"status": "TEACHER_ATTACHED_TO_FAILED_BASE"}
            else:
                failed[seed] = {"status": status, "failures": result.get("failures") or []}
            continue

        base = result.get("base")
        if not _safe_sim(base, seed):
            failed[seed] = {"status": "UNSAFE_OR_INCOMPLETE_BASE"}
            continue
        safe_base_result_count += 1

        if status in {"BASE_50K_FAILURE", "SEED_SEARCH_INCOMPLETE", "SEED_SEARCH_FAILURE"}:
            failed[seed] = {"status": status, "failures": result.get("failures") or []}
            continue

        if status in FULL_WIN:
            baseline = result.get("baseline_50k")
            rescue = result.get("rescue") or {}
            target = int(result.get("target_boss_floor", 0) or 0)
            if not _safe_sim(baseline, seed):
                failed[seed] = {"status": "UNSAFE_OR_INCOMPLETE_BASELINE_50K"}
                continue
            if _passes_target(baseline, target):
                failed[seed] = {"status": "BASELINE_ALREADY_RESCUED_TARGET"}
                continue
            if not _isinstance_dict(rescue.get("boss_10k")) or not _isinstance_dict(rescue.get("boss_50k")):
                failed[seed] = {"status": "MISSING_VERIFICATION_RUNS"}
                continue
            if not _safe_sim(rescue["boss_10k"], seed) or not _passes_target(rescue["boss_10k"], target):
                failed[seed] = {"status": "UNSAFE_OR_FAILED_BOSS_10K"}
                continue
            if not _safe_sim(rescue["boss_50k"], seed) or str(rescue["boss_50k"].get("outcome", "")).lower() != "victory":
                failed[seed] = {"status": "UNSAFE_OR_FAILED_BOSS_50K"}
                continue
            required = 2 if status == "NEW_FULL_WIN_TWO_STEP" else 1
            if len(result_rows) != required:
                failed[seed] = {"status": "TEACHER_COUNT_MISMATCH", "expected": required, "actual": len(result_rows)}
                continue
            errors = [_teacher_error(row, seed, status) for row in result_rows]
            errors = [error for error in errors if error]
            if errors:
                failed[seed] = {"status": "INVALID_TEACHER_LABEL", "errors": errors}
                continue
            if status == "NEW_FULL_WIN_TWO_STEP" and sorted(int(r["intervention_step"]) for r in result_rows) != [1, 2]:
                failed[seed] = {"status": "TWO_STEP_TEACHER_ORDER_INVALID"}
                continue
            verified_rows.extend(result_rows)
        else:
            if result_rows:
                failed[seed] = {"status": "TEACHER_ATTACHED_TO_NON_FULL_WIN", "observed": status}
                continue
            if status not in {"BASE_UNEXPECTED_WIN", "BASE_50K_FAILURE", "SEED_SEARCH_INCOMPLETE", "SEED_SEARCH_FAILURE"}:
                if not _safe_sim(result.get("baseline_50k"), seed):
                    failed[seed] = {"status": "UNSAFE_OR_INCOMPLETE_BASELINE_50K"}
                    continue

    unexpected = sorted(set(by_seed) - expected)
    for seed in unexpected:
        failed[seed] = {"status": "UNEXPECTED_SEED_RESULT", "paths": [str(p) for p in by_seed[seed]]}
    if unassigned:
        failed[-1] = {"status": "UNASSIGNED_RESULT_FILES", "paths": unassigned}

    # Exact JSON identity avoids collapsing two different interventions that
    # happen to share seed/floor/action indices.
    seen: set[str] = set()
    unique_rows = []
    for row in verified_rows:
        key = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    unique_rows.sort(key=lambda r: (int(r["seed"]), int(r["floor"]), int(r.get("intervention_step", 1))))
    teacher_seeds = sorted({int(row["seed"]) for row in unique_rows})
    if set(teacher_seeds) & (gate50 | fresh100 | fresh500):
        raise ValueError("formal evaluation seed leaked into Teachers")
    if not set(teacher_seeds).issubset(expected & training):
        raise ValueError("Teacher seed is outside priority80/train160")

    teacher_file = output_dir / "new-win-teachers.jsonl"
    teacher_file.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in unique_rows), encoding="utf-8")
    (output_dir / "teacher-seeds.txt").write_text("".join(f"{seed}\n" for seed in teacher_seeds), encoding="utf-8")
    failure_rows = [
        {"seed": seed, **detail}
        for seed, detail in sorted(failed.items(), key=lambda x: x[0])
    ]
    (output_dir / "failed-seeds.json").write_text(json.dumps(failure_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "schema_version": "sts1-v35-phaseb-isolated-teacher-pool-v1",
        "expected_mining_seeds": len(expected),
        "safe_base_results": safe_base_result_count,
        "failed_or_missing_seed_count": len(failed),
        "failed_or_missing_seed_ids": sorted(seed for seed in failed if seed >= 0),
        "teacher_examples": len(unique_rows),
        "teacher_seeds": len(teacher_seeds),
        "teacher_seed_ids": teacher_seeds,
        "status_counts": dict(sorted(status_counts.items())),
        "teacher_kind_counts": dict(sorted(Counter(str(row["kind"]) for row in unique_rows).items())),
        "teacher_type_counts": dict(sorted(Counter(str(row["type"]) for row in unique_rows).items())),
        "counterfactual_failure_kind_counts": dict(sorted(trial_failure_kinds.items())),
        "seed_sets_verified": True,
        "training_eval_overlap": 0,
        "teacher_data_source": "verified NEW_FULL_WIN_* results only",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _isinstance_dict(value: Any) -> bool:
    return isinstance(value, dict)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.results_dir, args.screen_dir, args.output_dir)
    print("V35_POSITIVE_AGGREGATE", json.dumps(report, sort_keys=True), flush=True)
    print(f"teacher_count={report['teacher_examples']}")
    print(f"teacher_seed_count={report['teacher_seeds']}")
    print(f"failed_seed_count={report['failed_or_missing_seed_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
