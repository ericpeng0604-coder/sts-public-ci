"""Audit and reaggregate a fully executed H3 held-out run without rerunning seeds."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sts1"))

import sts1_g7_h3_emergency_potion_eval as h3  # noqa: E402
from sts1_g7_h3_stage_eval import (  # noqa: E402
    FROZEN_CANDIDATE_COMMIT,
    FROZEN_SIMULATOR_SOURCE_SHA256,
    STAGES,
    StageEvaluationError,
    _paired_summary,
    _validate_pool,
)
from sts1_g7_seed_ledger import sha256_json  # noqa: E402


RUN_COMMIT = "4d171f5828ba56d2ec43434a2da1ccd269158943"
STAGE_EVALUATOR_COMMIT = "48f60b04868714cbd026b90b12e7d68ef0a35c7e"
STAGE_EVALUATOR_SOURCE_SHA256 = "46b6272d76f9c4a7cbd88a5608347993a7498ebd60068724f5c78040eabdda02"
MCTS_SIMS = 2000


class ReaggregationError(RuntimeError):
    """Stored held-out evidence failed independent post-run validation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_one_summary(path: Path, expected_seed: int) -> dict[str, Any]:
    events = h3.shared._read_jsonl(path)
    summaries = [event for event in events if event.get("type") == "summary"]
    if len(summaries) != 1:
        raise ReaggregationError("each episode evidence file must contain one summary")
    summary = summaries[0]
    if (
        summary.get("seed_contract") != "heldout_internal"
        or summary.get("simulator_seed_long") != expected_seed
        or str(summary.get("seed")) != str(expected_seed)
        or summary.get("result") != "PASS_SIMULATOR_COMPLETE_RUN"
        or summary.get("error") is not None
        or summary.get("outcome") not in {"victory", "defeat"}
        or any(int(summary.get(key, -1)) != 0 for key in (
            "illegal_action_count", "crash_count", "timeout_count"
        ))
    ):
        raise ReaggregationError("episode summary failed held-out, completion, outcome, or safety checks")
    return summary


def _private_use_history(usage_ledger: Path, pool_id: str) -> list[dict[str, Any]]:
    if not usage_ledger.is_file():
        raise ReaggregationError("private usage ledger is missing")
    events = [event for event in h3.shared._read_jsonl(usage_ledger) if event.get("pool_id") == pool_id]
    if (
        len(events) != 2
        or events[0].get("status") != "RUNNING"
        or events[1].get("status") != "NOT_VERIFIED"
        or events[1].get("error_type") != "EvaluationIntegrityError"
        or events[1].get("episode_count") != 61
        or events[0].get("candidate_policy_commit") != FROZEN_CANDIDATE_COMMIT
        or any(event.get("status") == "COMPLETE" for event in events)
    ):
        raise ReaggregationError("usage history does not match the single stopped Dev30 aggregation attempt")
    return events


def _expected_file_names(pair_count: int) -> set[str]:
    names = {"paired-runs.ndjson"}
    for index in range(pair_count):
        for arm in ("parent", "candidate"):
            names.add(f"pair-{index:02d}-{arm}.ndjson")
            names.add(f"pair-{index:02d}-{arm}.trace.ndjson")
    names.add("pair-00-candidate-trace-off.ndjson")
    return names


def _summarize_artifact_inventory(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def reaggregate_dev30(
    *,
    pool_path: Path,
    pools_dir: Path,
    inventory_path: Path,
    evidence_dir: Path,
    usage_ledger: Path,
) -> dict[str, Any]:
    spec = STAGES["dev"]
    pool, seeds, pool_audit = _validate_pool("dev", pool_path, pools_dir, inventory_path)
    if len(seeds) != 30:
        raise ReaggregationError("Dev30 manifest does not contain exactly 30 held-out seeds")
    _private_use_history(usage_ledger, spec["pool_id"])
    if not evidence_dir.is_dir() or REPO_ROOT.resolve() in evidence_dir.resolve().parents:
        raise ReaggregationError("private Dev30 evidence directory is missing or inside the repository")

    evidence_paths = sorted(evidence_dir.glob("*.ndjson"), key=lambda path: path.name)
    expected_names = _expected_file_names(len(seeds))
    if {path.name for path in evidence_paths} != expected_names:
        raise ReaggregationError("private evidence file inventory is incomplete or contains unexpected files")

    parent_outcomes: list[str] = []
    candidate_outcomes: list[str] = []
    rows = list(h3.shared._read_jsonl(evidence_dir / "paired-runs.ndjson"))
    if len(rows) != len(seeds):
        raise ReaggregationError("paired-runs ledger does not contain exactly 30 pair records")

    terminal_count = 0
    combat_decisions = 0
    mcts_budgets: Counter[str] = Counter()
    emergency_opportunities = 0
    emergency_overrides = 0
    for index, seed in enumerate(seeds):
        parent_path = evidence_dir / f"pair-{index:02d}-parent.ndjson"
        candidate_path = evidence_dir / f"pair-{index:02d}-candidate.ndjson"
        parent = _read_one_summary(parent_path, seed)
        candidate = _read_one_summary(candidate_path, seed)
        parent_outcomes.append(str(parent["outcome"]))
        candidate_outcomes.append(str(candidate["outcome"]))

        parent_trace_path = evidence_dir / f"pair-{index:02d}-parent.trace.ndjson"
        candidate_trace_path = evidence_dir / f"pair-{index:02d}-candidate.trace.ndjson"
        parent_trace = h3._check_trace(
            parent_trace_path, candidate=False,
            expected_overrides=int(parent.get("emergency_potion_override_count", 0)),
        )
        candidate_trace = h3._check_trace(
            candidate_trace_path, candidate=True,
            expected_overrides=int(candidate.get("emergency_potion_override_count", 0)),
        )
        terminal_count += 2
        combat_decisions += parent_trace["combat_decisions"] + candidate_trace["combat_decisions"]
        emergency_opportunities += candidate_trace["low_hp_potion_opportunities"]
        emergency_overrides += candidate_trace["overrides"]
        for trace_path in (parent_trace_path, candidate_trace_path):
            for event in h3.shared._read_jsonl(trace_path):
                if event.get("type") == "combat_decision_trace_v1":
                    mcts_budgets.update([str(event.get("mcts_sims"))])

        row = rows[index]
        if (
            row.get("parent") != h3._summary_view(parent)
            or row.get("candidate") != h3._summary_view(candidate)
            or row.get("parent_trace") != parent_trace
            or row.get("candidate_trace") != candidate_trace
        ):
            raise ReaggregationError("private pair ledger differs from its validated episode/trace files")

    trace_off_path = evidence_dir / "pair-00-candidate-trace-off.ndjson"
    trace_off = _read_one_summary(trace_off_path, seeds[0])
    candidate_0 = _read_one_summary(evidence_dir / "pair-00-candidate.ndjson", seeds[0])
    if (
        trace_off.get("outcome") != candidate_0.get("outcome")
        or h3._summary_view(trace_off) != h3._summary_view(candidate_0)
        or h3._action_signature(trace_off_path)
        != h3._action_signature(evidence_dir / "pair-00-candidate.ndjson")
    ):
        raise ReaggregationError("stored trace-on/off invariance comparison failed")
    if terminal_count != 60 or mcts_budgets != Counter({str(MCTS_SIMS): sum(mcts_budgets.values())}):
        raise ReaggregationError("trace terminal count or MCTS-2000 budget inventory is invalid")

    paired = _paired_summary(parent_outcomes, candidate_outcomes)
    artifact_inventory_sha = _summarize_artifact_inventory(evidence_paths)
    final: dict[str, Any] = {
        "schema_version": "sts1-g7-h3-dev30-reaggregated-v1",
        "status": "COMPLETE_REAGGREGATED",
        "original_runner_status": "NOT_VERIFIED",
        "original_runner_failure": "paired summary helper was train10-only and rejected 30 pairs after all episodes completed",
        "stage": "dev",
        "pool_id": spec["pool_id"],
        "pool_manifest_sha256": spec["manifest_sha256"],
        "candidate_policy_commit": FROZEN_CANDIDATE_COMMIT,
        "candidate_policy_source_sha256": FROZEN_SIMULATOR_SOURCE_SHA256,
        "run_commit": RUN_COMMIT,
        "stage_evaluator_commit": STAGE_EVALUATOR_COMMIT,
        "stage_evaluator_source_sha256": STAGE_EVALUATOR_SOURCE_SHA256,
        "seed_contract": "heldout_internal",
        "paired_seed_count": len(seeds),
        "episode_count": len(seeds) * 2 + 1,
        "trace_invariance": "PASS",
        "trace_enabled_terminal_records_complete_and_legal": terminal_count,
        "combat_decisions": combat_decisions,
        "combat_mcts_budgets": dict(mcts_budgets),
        "candidate_low_hp_potion_opportunities": emergency_opportunities,
        "candidate_emergency_potion_overrides": emergency_overrides,
        "illegal_actions": 0,
        "crashes": 0,
        "timeouts": 0,
        "communication_errors": "N/A_LOCAL_SIMULATOR",
        "paired": paired,
        "pool_audit": pool_audit,
        "private_evidence_file_count": len(evidence_paths),
        "private_evidence_inventory_sha256": artifact_inventory_sha,
    }
    final["reaggregated_summary_sha256"] = sha256_json(final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--pools-dir", type=Path, required=True)
    parser.add_argument("--exclusion-inventory", type=Path, required=True)
    parser.add_argument("--private-evidence-dir", type=Path, required=True)
    parser.add_argument("--private-usage-ledger", type=Path, required=True)
    parser.add_argument("--write-private-summary", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        final = reaggregate_dev30(
            pool_path=args.pool_file.resolve(), pools_dir=args.pools_dir.resolve(),
            inventory_path=args.exclusion_inventory.resolve(),
            evidence_dir=args.private_evidence_dir.resolve(), usage_ledger=args.private_usage_ledger.resolve(),
        )
        if args.write_private_summary:
            output_path = args.private_evidence_dir.resolve() / "paired-summary-reaggregated.json"
            if output_path.exists():
                raise ReaggregationError("reaggregated summary already exists; preserve it and do not overwrite")
            output_path.write_text(json.dumps(final, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            h3.shared._append_jsonl(args.private_usage_ledger.resolve(), {
                "pool_id": final["pool_id"], "manifest_sha256": final["pool_manifest_sha256"],
                "stage": "dev", "status": "COMPLETE_REAGGREGATED",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "episode_count": final["episode_count"], "aggregate_sha256": final["reaggregated_summary_sha256"],
                "original_runner_status": "NOT_VERIFIED", "candidate_policy_commit": FROZEN_CANDIDATE_COMMIT,
            })
        print(json.dumps(final, ensure_ascii=False, sort_keys=True))
        return 0
    except (ReaggregationError, StageEvaluationError, h3.EvaluationIntegrityError, OSError, ValueError) as exc:
        print(json.dumps({"status": "NOT_VERIFIED", "error_type": type(exc).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
