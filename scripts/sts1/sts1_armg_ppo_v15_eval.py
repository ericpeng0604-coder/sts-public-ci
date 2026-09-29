#!/usr/bin/env python3
"""Parallel-friendly PPO v1.5 fixed-seed evaluation helpers.

Parent evaluation and each population Candidate run on separate GitHub runners.
Every evidence file is self-identifying by weight SHA, simulator identity,
MCTS budget, and the pinned 50-seed list so downstream reuse can fail closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import sts1_armg_ppo_v13_auto_gate as rt
import sts1_armg_ppo_v14_gate as gate
import sts1_armg_ppo_v15_tournament as tournament

PARENT_SCHEMA = "sts1-armg-ppo-v15-parent-eval-v1"
CANDIDATE_SCHEMA = "sts1-armg-ppo-v15-candidate-eval-v1"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_parent(args: argparse.Namespace) -> int:
    seeds = gate._read_seeds(args.dev_seed_file)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    simulator_id = rt._automatic_simulator_id(args.module_dir)
    parent_sha = rt._sha256(args.parent_weight)

    runs = gate._load_cache(
        state,
        key="dev_parent",
        weight_sha=parent_sha,
        seeds=seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
    )
    before = len(runs)
    runs = gate._run_missing(
        label="v15-dev-parent",
        weight=args.parent_weight,
        seeds=seeds[:30],
        cache=runs,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        output_dir=args.output_dir / "parent",
        mcts_sims=args.mcts_sims,
        all_seeds=seeds,
        workers=args.workers,
    )
    gate._save_cache(
        state,
        key="dev_parent",
        weight_sha=parent_sha,
        seeds=seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
        runs=runs,
    )
    args.state.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": PARENT_SCHEMA,
        "parent_weight_sha256": parent_sha,
        "simulator_id": simulator_id,
        "mcts_sims": args.mcts_sims,
        "all_dev_seeds": seeds,
        "evaluated_seeds": seeds[:30],
        "cache_hits_before": before,
        "fresh_runs": 30 - before,
        "runs": gate._ordered(runs, seeds[:30]),
    }
    _write(args.output_dir / "parent-eval.json", payload)
    _write(args.output_dir / "state.json", state)
    print(
        "PPO_V15_PARENT_EVAL",
        json.dumps(
            {
                "cache_hits_before": before,
                "fresh_runs": 30 - before,
                "parent_weight_sha256": parent_sha,
                "mcts_sims": args.mcts_sims,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _validate_parent_evidence(
    payload: dict[str, Any],
    *,
    dev_seeds: list[int],
    mcts_sims: int,
    simulator_id: str,
) -> list[dict[str, Any]]:
    if payload.get("schema_version") != PARENT_SCHEMA:
        raise RuntimeError("unexpected parent evidence schema")
    if payload.get("all_dev_seeds") != dev_seeds:
        raise RuntimeError("parent evidence seed identity mismatch")
    if int(payload.get("mcts_sims", -1)) != int(mcts_sims):
        raise RuntimeError("parent evidence MCTS mismatch")
    if payload.get("simulator_id") != simulator_id:
        raise RuntimeError("parent evidence simulator mismatch")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 30:
        raise RuntimeError("parent evidence must contain exactly 30 runs")
    by_seed = {int(row["seed"]): row for row in runs}
    return gate._ordered(by_seed, dev_seeds[:30])


def evaluate_candidate(args: argparse.Namespace) -> int:
    seeds = gate._read_seeds(args.dev_seed_file)
    simulator_id = rt._automatic_simulator_id(args.module_dir)
    parent_payload = json.loads(args.parent_evidence.read_text(encoding="utf-8"))
    parent_runs = _validate_parent_evidence(
        parent_payload,
        dev_seeds=seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
    )

    name, weight = tournament._parse_candidate(args.candidate)
    if not weight.is_file():
        raise RuntimeError(f"missing candidate {name}: {weight}")
    candidate_sha = rt._sha256(weight)
    candidate_runs: dict[int, dict[str, Any]] = {}
    candidate_runs = gate._run_missing(
        label=f"v15-dev-{name}",
        weight=weight,
        seeds=seeds[:30],
        cache=candidate_runs,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        output_dir=args.output_dir / name,
        mcts_sims=args.mcts_sims,
        all_seeds=seeds,
        workers=args.workers,
    )
    ordered_candidate = gate._ordered(candidate_runs, seeds[:30])
    dev = gate._dev_gate(parent_runs, ordered_candidate)
    row = tournament._candidate_row(
        name=name,
        weight=weight,
        dev_gate=dev,
        index=args.index,
    )
    payload = {
        "schema_version": CANDIDATE_SCHEMA,
        "parent_weight_sha256": parent_payload["parent_weight_sha256"],
        "candidate_weight_sha256": candidate_sha,
        "simulator_id": simulator_id,
        "mcts_sims": args.mcts_sims,
        "all_dev_seeds": seeds,
        "evaluated_seeds": seeds[:30],
        "candidate": row,
        "runs": ordered_candidate,
    }
    _write(args.output_dir / "candidate-eval.json", payload)
    print(
        "PPO_V15_CANDIDATE_EVAL",
        json.dumps(row, sort_keys=True),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    parent = sub.add_parser("parent")
    parent.add_argument("--module-dir", type=Path, required=True)
    parent.add_argument("--armg-root", type=Path, required=True)
    parent.add_argument("--parent-weight", type=Path, required=True)
    parent.add_argument("--dev-seed-file", type=Path, required=True)
    parent.add_argument("--state", type=Path, required=True)
    parent.add_argument("--output-dir", type=Path, required=True)
    parent.add_argument("--mcts-sims", type=int, default=2000)
    parent.add_argument("--workers", type=int, default=4)

    candidate = sub.add_parser("candidate")
    candidate.add_argument("--module-dir", type=Path, required=True)
    candidate.add_argument("--armg-root", type=Path, required=True)
    candidate.add_argument("--candidate", required=True)
    candidate.add_argument("--index", type=int, required=True)
    candidate.add_argument("--dev-seed-file", type=Path, required=True)
    candidate.add_argument("--parent-evidence", type=Path, required=True)
    candidate.add_argument("--output-dir", type=Path, required=True)
    candidate.add_argument("--mcts-sims", type=int, default=2000)
    candidate.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise RuntimeError("workers must be within 1..8")
    if args.mcts_sims != 2000:
        raise RuntimeError("v1.5 evaluation is pinned to MCTS-2000")
    if args.mode == "parent":
        return evaluate_parent(args)
    return evaluate_candidate(args)


if __name__ == "__main__":
    raise SystemExit(main())
