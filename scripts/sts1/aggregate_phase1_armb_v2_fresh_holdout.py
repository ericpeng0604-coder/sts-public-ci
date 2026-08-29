#!/usr/bin/env python3
"""Aggregate the predeclared Arm B v2 fresh holdout.

Arm B v2 is a reference-baseline candidate only. It never claims the full
Phase-1 Teacher gate. The policy is frozen as: unique Arm B top action -> Arm B;
otherwise -> deterministic simple-public fallback.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roguelike_ai.sts1_teacher.armb_v2 import ARMB_V2_POLICY_ID, armb_v2_action_id
from roguelike_ai.sts1_teacher.benchmark import (
    EXPECTED_HELDOUT_SEEDS,
    ORACLE_MCTS_SIMS,
    canonical_decision_digest,
    load_heldout_seeds,
    sha256_file,
    write_json,
    write_ndjson,
)

SCHEMA_VERSION = "sts1-phase1-armb-v2-fresh-holdout-v1"
FRESH_SEEDS = ROOT / "tests" / "data" / "sts1_phase1_armb_v2_fresh_seeds_50.txt"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected_json_object:{path}")
    return value


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"expected_ndjson_object:{path}")
        rows.append(value)
    return rows


def _collect(root: Path, repeat: int) -> tuple[list[dict[str, Any]], list[int], dict[str, int], list[dict[str, Any]]]:
    directories = sorted(path for path in root.glob(f"repeat-{repeat}-shard-*") if path.is_dir())
    if not directories:
        raise RuntimeError(f"no_shards_for_repeat:{repeat}")

    rows: list[dict[str, Any]] = []
    seeds: list[int] = []
    counters = {"unresolved": 0, "illegal": 0, "leakage": 0}
    provenance: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    shard_count: int | None = None

    for directory in directories:
        summary = _read_json(directory / "benchmark_summary.json")
        prov = _read_json(directory / "provenance.json")
        if int(prov.get("repeat", -1)) != repeat:
            raise RuntimeError(f"repeat_mismatch:{directory}")
        index = int(prov.get("shard_index", -1))
        count = int(prov.get("shard_count", -1))
        if index in seen_shards:
            raise RuntimeError(f"duplicate_shard:{repeat}:{index}")
        seen_shards.add(index)
        shard_count = count if shard_count is None else shard_count
        if shard_count != count:
            raise RuntimeError("inconsistent_shard_count")
        if int(summary.get("oracle_mcts_sims", -1)) != ORACLE_MCTS_SIMS:
            raise RuntimeError("oracle_mcts_budget_drift")

        rows.extend(_read_ndjson(directory / "benchmark_decisions.ndjson"))
        seeds.extend(int(seed) for seed in summary.get("seeds", []))
        for key in counters:
            counters[key] += int(summary.get(key, 0))
        provenance.append(prov)

    if shard_count is None or seen_shards != set(range(shard_count)):
        raise RuntimeError(f"missing_shards:{repeat}:{sorted(seen_shards)}:{shard_count}")
    if len(seeds) != EXPECTED_HELDOUT_SEEDS or len(set(seeds)) != EXPECTED_HELDOUT_SEEDS:
        raise RuntimeError(f"seed_coverage_not_50_unique:{repeat}:{len(seeds)}:{len(set(seeds))}")

    expected_seeds = set(load_heldout_seeds(FRESH_SEEDS))
    if set(seeds) != expected_seeds:
        raise RuntimeError(f"fresh_seed_set_mismatch:{repeat}")

    rows.sort(key=lambda row: (
        int(row["seed"]),
        int(row["combat_index"]),
        int(row["decision_index"]),
        str(row["decision_signature"]),
    ))
    return rows, seeds, counters, provenance


def _candidate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    source_counts = Counter()
    decisions: list[dict[str, Any]] = []

    for row in rows:
        oracle = set(str(value) for value in row.get("oracle_tie_ids", []))
        simple = str(row.get("simple_action_id")) if row.get("simple_action_id") is not None else None
        decision = armb_v2_action_id(tuple(row.get("armb_tie_ids", [])), simple)
        if decision.action_id is None:
            counts["candidate_unresolved"] += 1
            candidate_agrees = False
        else:
            candidate_agrees = decision.action_id in oracle
        counts["candidate"] += int(candidate_agrees)
        counts["simple"] += int(bool(row.get("agreement", {}).get("simple")))
        counts["random"] += int(bool(row.get("agreement", {}).get("random")))
        counts["armb_v1_conservative"] += int(bool(row.get("agreement", {}).get("armb")))
        source_counts[decision.source] += 1
        decisions.append({
            "seed": int(row["seed"]),
            "combat_index": int(row["combat_index"]),
            "decision_index": int(row["decision_index"]),
            "decision_signature": str(row["decision_signature"]),
            "candidate_action_id": decision.action_id,
            "candidate_source": decision.source,
            "candidate_agrees": candidate_agrees,
        })

    total = len(rows)
    return {
        "decision_count": total,
        "agreement_counts": {
            "candidate": counts["candidate"],
            "simple": counts["simple"],
            "random": counts["random"],
            "armb_v1_conservative": counts["armb_v1_conservative"],
        },
        "agreement_rates": {
            name: (counts[name] / total if total else 0.0)
            for name in ("candidate", "simple", "random", "armb_v1_conservative")
        },
        "candidate_unresolved": counts["candidate_unresolved"],
        "source_counts": dict(source_counts),
        "candidate_decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run1_rows, run1_seeds, run1_counts, run1_prov = _collect(args.input_root, 1)
    run2_rows, run2_seeds, run2_counts, run2_prov = _collect(args.input_root, 2)
    digest1 = canonical_decision_digest(run1_rows)
    digest2 = canonical_decision_digest(run2_rows)
    deterministic = (
        run1_rows == run2_rows
        and digest1 == digest2
        and sorted(run1_seeds) == sorted(run2_seeds)
        and run1_counts == run2_counts
    )

    metrics = _candidate_metrics(run1_rows)
    rates = metrics["agreement_rates"]
    reasons: list[str] = []
    if not deterministic:
        reasons.append("fresh_holdout_not_deterministic")
    for key, value in run1_counts.items():
        if int(value) != 0:
            reasons.append(f"{key}_must_be_zero")
    if int(metrics["candidate_unresolved"]) != 0:
        reasons.append("candidate_unresolved_must_be_zero")
    if int(metrics["decision_count"]) <= 0:
        reasons.append("no_benchmark_decisions")
    if not float(rates["candidate"]) > float(rates["random"]):
        reasons.append("candidate_not_strictly_better_than_random")
    if not float(rates["candidate"]) > float(rates["simple"]):
        reasons.append("candidate_not_strictly_better_than_simple")
    gate_pass = not reasons

    all_prov = run1_prov + run2_prov
    seed_hashes = {str(item.get("heldout_seed_file_sha256")) for item in all_prov}
    agent_shas = {str(item.get("agent_sha")) for item in all_prov}
    sim_shas = {str(item.get("simulator_sha")) for item in all_prov}
    if len(seed_hashes) != 1 or len(agent_shas) != 1 or len(sim_shas) != 1:
        raise RuntimeError("provenance_drift_across_repeats")
    expected_seed_hash = sha256_file(FRESH_SEEDS)
    if next(iter(seed_hashes)) != expected_seed_hash:
        raise RuntimeError("fresh_seed_file_hash_mismatch")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if gate_pass else "FAIL",
        "policy_id": ARMB_V2_POLICY_ID,
        "seed_count": len(run1_seeds),
        "fresh_seed_file_sha256": expected_seed_hash,
        "oracle_mcts_sims": ORACLE_MCTS_SIMS,
        "decision_count": metrics["decision_count"],
        "agreement_counts": metrics["agreement_counts"],
        "agreement_rates": metrics["agreement_rates"],
        "source_counts": metrics["source_counts"],
        "candidate_unresolved": metrics["candidate_unresolved"],
        "unresolved": run1_counts["unresolved"],
        "illegal": run1_counts["illegal"],
        "leakage": run1_counts["leakage"],
        "gate_reasons": reasons,
        "phase1_gate_claimed": False,
    }
    determinism = {
        "schema_version": SCHEMA_VERSION,
        "pass": deterministic,
        "full_repeats": 2,
        "repeat_1_decision_digest": digest1,
        "repeat_2_decision_digest": digest2,
        "exact_rows_equal": run1_rows == run2_rows,
        "repeat_1_decisions": len(run1_rows),
        "repeat_2_decisions": len(run2_rows),
        "repeat_1_counters": run1_counts,
        "repeat_2_counters": run2_counts,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": ARMB_V2_POLICY_ID,
        "candidate_role": "reference_baseline_only",
        "candidate_rule": "unique Arm B top action; otherwise deterministic simple-public fallback",
        "agent_sha": next(iter(agent_shas)),
        "simulator_sha": next(iter(sim_shas)),
        "fresh_seed_file_sha256": expected_seed_hash,
        "fresh_seed_count": EXPECTED_HELDOUT_SEEDS,
        "original_50_reused_as_holdout": False,
        "oracle_role": "benchmark_reference_only",
        "oracle_uses_hidden_rng": True,
        "formal_teacher_uses_oracle": False,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if gate_pass else "FAIL",
        "phase1_gate_claimed": False,
        "note": "Arm B v2 is a fresh reference-baseline experiment only. PASS does not complete #345.",
        "summary_file": "summary.json",
        "determinism_file": "determinism.json",
        "provenance_file": "provenance.json",
        "candidate_decisions_file": "candidate_decisions.ndjson",
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    write_json(out / "determinism.json", determinism)
    write_json(out / "provenance.json", provenance)
    write_json(out / "manifest.json", manifest)
    write_ndjson(out / "candidate_decisions.ndjson", metrics["candidate_decisions"])
    print(json.dumps({"manifest": manifest, "summary": summary, "determinism": determinism}, indent=2))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
