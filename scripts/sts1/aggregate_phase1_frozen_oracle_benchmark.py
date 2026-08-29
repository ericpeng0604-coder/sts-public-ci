#!/usr/bin/env python3
"""Aggregate two complete sharded Phase-1 oracle benchmark repeats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roguelike_ai.sts1_teacher.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    EXPECTED_HELDOUT_SEEDS,
    ORACLE_MCTS_SIMS,
    canonical_decision_digest,
    phase1_baseline_gate,
    summarize_rows,
    write_json,
    write_ndjson,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"expected NDJSON object: {path}")
        rows.append(value)
    return rows


def _collect(root: Path, repeat: int) -> tuple[list[dict[str, Any]], list[int], dict[str, int], list[dict[str, Any]]]:
    shard_dirs = sorted(path for path in root.glob(f"repeat-{repeat}-shard-*") if path.is_dir())
    if not shard_dirs:
        raise RuntimeError(f"no shard directories for repeat {repeat}")
    rows: list[dict[str, Any]] = []
    seeds: list[int] = []
    counters = {"unresolved": 0, "illegal": 0, "leakage": 0}
    provenance: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    shard_count: int | None = None
    for directory in shard_dirs:
        summary = _read_json(directory / "benchmark_summary.json")
        prov = _read_json(directory / "provenance.json")
        if int(prov.get("repeat", -1)) != repeat:
            raise RuntimeError(f"repeat mismatch in {directory}")
        index = int(prov["shard_index"])
        count = int(prov["shard_count"])
        if index in seen_shards:
            raise RuntimeError(f"duplicate shard {repeat}:{index}")
        seen_shards.add(index)
        shard_count = count if shard_count is None else shard_count
        if shard_count != count:
            raise RuntimeError("inconsistent shard_count")
        if int(summary.get("oracle_mcts_sims", -1)) != ORACLE_MCTS_SIMS:
            raise RuntimeError("oracle MCTS budget drift")
        rows.extend(_read_ndjson(directory / "benchmark_decisions.ndjson"))
        seeds.extend(int(seed) for seed in summary.get("seeds", []))
        for key in counters:
            counters[key] += int(summary.get(key, 0))
        provenance.append(prov)
    if shard_count is None or seen_shards != set(range(shard_count)):
        raise RuntimeError(f"missing shards for repeat {repeat}: {sorted(seen_shards)} / {shard_count}")
    if len(seeds) != EXPECTED_HELDOUT_SEEDS or len(set(seeds)) != EXPECTED_HELDOUT_SEEDS:
        raise RuntimeError(f"repeat {repeat} seed coverage is not exactly 50 unique")
    rows.sort(key=lambda row: (
        int(row["seed"]), int(row["combat_index"]), int(row["decision_index"]), str(row["decision_signature"])
    ))
    return rows, seeds, counters, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run1_rows, run1_seeds, run1_counts, run1_prov = _collect(args.input_root, 1)
    run2_rows, run2_seeds, run2_counts, run2_prov = _collect(args.input_root, 2)
    run1_digest = canonical_decision_digest(run1_rows)
    run2_digest = canonical_decision_digest(run2_rows)
    deterministic = (
        run1_digest == run2_digest
        and run1_rows == run2_rows
        and sorted(run1_seeds) == sorted(run2_seeds)
        and run1_counts == run2_counts
    )

    summary = summarize_rows(run1_rows, seeds=run1_seeds, **run1_counts)
    gate_pass, gate_reasons = phase1_baseline_gate(summary, deterministic=deterministic)
    summary["armb_baseline_gate"] = "PASS" if gate_pass else "FAIL"
    summary["gate_reasons"] = gate_reasons

    all_prov = run1_prov + run2_prov
    seed_hashes = {str(item.get("heldout_seed_file_sha256")) for item in all_prov}
    agent_shas = {str(item.get("agent_sha")) for item in all_prov}
    sim_shas = {str(item.get("simulator_sha")) for item in all_prov}
    if len(seed_hashes) != 1 or len(agent_shas) != 1 or len(sim_shas) != 1:
        raise RuntimeError("provenance drift across benchmark shards/repeats")

    determinism = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "pass": deterministic,
        "full_repeats": 2,
        "repeat_1_decision_digest": run1_digest,
        "repeat_2_decision_digest": run2_digest,
        "repeat_1_decisions": len(run1_rows),
        "repeat_2_decisions": len(run2_rows),
        "repeat_1_counters": run1_counts,
        "repeat_2_counters": run2_counts,
        "exact_rows_equal": run1_rows == run2_rows,
    }
    provenance = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "agent_sha": next(iter(agent_shas)),
        "simulator_sha": next(iter(sim_shas)),
        "heldout_seed_file_sha256": next(iter(seed_hashes)),
        "heldout_seed_count": EXPECTED_HELDOUT_SEEDS,
        "oracle_mcts_sims": ORACLE_MCTS_SIMS,
        "oracle_role": "benchmark_reference_only",
        "oracle_uses_hidden_rng": True,
        "formal_teacher_uses_oracle": False,
        "full_repeats": 2,
        "shards_per_repeat": len(run1_prov),
    }
    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "PASS" if gate_pass else "FAIL",
        "phase1_gate_claimed": False,
        "note": "This is the Arm B baseline sub-gate only; full #345 still requires the remaining Phase-1 review/merge gates.",
        "summary_file": "benchmark_summary.json",
        "decisions_file": "benchmark_decisions.ndjson",
        "determinism_file": "determinism.json",
        "provenance_file": "provenance.json",
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_ndjson(out / "benchmark_decisions.ndjson", run1_rows)
    write_json(out / "benchmark_summary.json", summary)
    write_json(out / "determinism.json", determinism)
    write_json(out / "provenance.json", provenance)
    write_json(out / "benchmark_manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "summary": summary, "determinism": determinism}, indent=2))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
