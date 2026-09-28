#!/usr/bin/env python3
"""Compare Strategy Teacher labeling with 1/2/3/4 CPU workers on identical states.

This benchmark never reads or writes the durable Strategy state. It uses a
throwaway fixed seed set and the frozen upstream ArmG weight, verifies that all
worker counts produce identical Teacher labels, and reports wall-time scaling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOOP_PATH = ROOT / "scripts" / "sts1" / "sts1_armg_strategy_loop.py"
SPEC = importlib.util.spec_from_file_location("strategy_loop_parallel_benchmark", LOOP_PATH)
assert SPEC is not None and SPEC.loader is not None
loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loop)


def _stable_rows(path: Path) -> list[dict[str, Any]]:
    rows = loop._read_examples(path)
    return [
        {
            "seed": row["seed"],
            "floor": row["floor"],
            "act": row["act"],
            "kind": row["kind"],
            "current_armg_index": row["current_armg_index"],
            "teacher_best_index": row["teacher_best_index"],
            "branch_quality": row["branch_quality"],
            "target_probs": row["target_probs"],
            "state_bucket": row["state_bucket"],
        }
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mcts-sims", type=int, default=2000)
    parser.add_argument("--label-budget", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[314159265, 271828182])
    args = parser.parse_args()

    if args.label_budget < 2:
        raise RuntimeError("benchmark label budget must be at least 2")
    if not args.weight.is_file():
        raise RuntimeError(f"benchmark weight missing: {args.weight}")

    sts = loop._load_sts(args.module_dir)
    results: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] | None = None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sts1-parallel-benchmark-") as tmp:
        tmpdir = Path(tmp)
        for workers in (1, 2, 3, 4):
            armg = loop.ArmGNoncombatPolicy(
                root=args.armg_root,
                weight_path=args.weight,
            )
            dataset = tmpdir / f"teacher-w{workers}.jsonl"
            report = loop._collect_dataset(
                seeds=tuple(args.seeds),
                sts=sts,
                armg=armg,
                output=dataset,
                mcts_sims=args.mcts_sims,
                max_branch_points_per_game=12,
                max_branch_points_per_kind_per_game=3,
                max_choice_branches=20,
                max_game_steps=600,
                max_battle_steps=1200,
                temperature=2.0,
                teacher_label_budget=args.label_budget,
                min_labels_per_kind=0,
                collection_workers=workers,
                parallel_timeout_seconds=1800,
                parallel_parity_probes=0,
            )
            stable = _stable_rows(dataset)
            if reference is None:
                reference = stable
            elif stable != reference:
                raise RuntimeError(
                    f"parallel benchmark parity mismatch for workers={workers}"
                )
            parallel = dict(report["parallel_teacher"])
            results.append(
                {
                    "workers": workers,
                    "labels": report["selected_teacher_count"],
                    "candidate_pool": report["candidate_pool_count"],
                    "elapsed_seconds": parallel["elapsed_seconds"],
                    "mode": parallel["mode"],
                    "effective_workers": parallel["effective_workers"],
                    "fallback_reason": parallel["fallback_reason"],
                    "observed_parallelism": parallel["observed_parallelism"],
                    "available_cpu": parallel["available_cpu"],
                }
            )

    baseline = float(results[0]["elapsed_seconds"])
    for row in results:
        elapsed = float(row["elapsed_seconds"])
        row["speedup_vs_1"] = baseline / elapsed if elapsed > 0 else 0.0
        row["seconds_per_label"] = elapsed / int(row["labels"])

    best = min(results, key=lambda row: float(row["elapsed_seconds"]))
    payload = {
        "schema_version": "sts1-strategy-parallel-benchmark-v1",
        "result": "PASS_PARALLEL_TEACHER_BENCHMARK",
        "mcts_sims": args.mcts_sims,
        "seeds": list(args.seeds),
        "label_budget": args.label_budget,
        "parity": "PASS",
        "runs": results,
        "best_workers": int(best["workers"]),
        "best_elapsed_seconds": float(best["elapsed_seconds"]),
        "best_speedup_vs_1": float(best["speedup_vs_1"]),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("STS1_PARALLEL_TEACHER_BENCHMARK", json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
