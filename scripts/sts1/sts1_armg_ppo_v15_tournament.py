#!/usr/bin/env python3
"""Select one PPO v1.5 population candidate on the same 30 Dev seeds.

This is only a pre-gate tournament. It never promotes a production Champion.
The selected candidate must still pass the existing v1.4 Dev + Final gate.
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


def _parse_candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, value = raw.split("=", 1)
    name = name.strip()
    if not name or not all(ch.isalnum() or ch in "-_" for ch in name):
        raise argparse.ArgumentTypeError("candidate name must be alnum/-/_")
    path = Path(value)
    return name, path


def _candidate_row(
    *,
    name: str,
    weight: Path,
    dev_gate: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    candidate = dev_gate["candidate"]
    safety = candidate["safety"]
    safe = all(int(v) == 0 for v in safety.values())
    complete = int(candidate["complete_runs"]) == 30
    row = {
        "name": name,
        "weight": str(weight),
        "index": index,
        "dev_status": dev_gate["status"],
        "dev_decision": dev_gate["decision"],
        "victories": int(candidate["victories"]),
        "mean_final_floor": float(candidate["mean_final_floor"]),
        "reach50_delta": int(dev_gate["reach50_delta"]),
        "win_delta": int(dev_gate["win_delta"]),
        "mean_paired_floor_delta": float(dev_gate["mean_paired_floor_delta"]),
        "safe": safe,
        "complete": complete,
        "safety": safety,
        "dev_gate": dev_gate,
    }
    row["rank_key"] = [
        int(safe and complete),
        int(dev_gate["status"] == "PASS"),
        row["victories"],
        row["mean_final_floor"],
        row["reach50_delta"],
        -index,
    ]
    return row


def choose_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("no population candidates")
    return max(rows, key=lambda row: tuple(row["rank_key"]))


PARENT_EVIDENCE_SCHEMA = "sts1-armg-ppo-v15-parent-eval-v1"
CANDIDATE_EVIDENCE_SCHEMA = "sts1-armg-ppo-v15-candidate-eval-v1"


def _runs_by_seed(rows: list[dict[str, Any]], seeds: list[int]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    allowed = set(seeds)
    for row in rows:
        seed = int(row["seed"])
        if seed not in allowed or seed in out:
            raise RuntimeError(f"invalid/duplicate evidence seed {seed}")
        if row.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
            raise RuntimeError("evaluation evidence contains incomplete run")
        out[seed] = dict(row)
    return out


def _assemble_evidence(args: argparse.Namespace) -> int:
    if args.parent_evidence is None or not args.candidate_evidence:
        raise RuntimeError("evidence mode requires parent and candidate evidence")
    if len(args.candidate_evidence) < 2:
        raise RuntimeError("population tournament requires at least two candidates")

    dev_seeds = gate._read_seeds(args.dev_seed_file)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulator_id = rt._automatic_simulator_id(args.module_dir)
    parent_sha = rt._sha256(args.parent_weight)

    parent_payload = json.loads(args.parent_evidence.read_text(encoding="utf-8"))
    if parent_payload.get("schema_version") != PARENT_EVIDENCE_SCHEMA:
        raise RuntimeError("unexpected parent evidence schema")
    if parent_payload.get("parent_weight_sha256") != parent_sha:
        raise RuntimeError("parent evidence weight mismatch")
    if parent_payload.get("simulator_id") != simulator_id:
        raise RuntimeError("parent evidence simulator mismatch")
    if int(parent_payload.get("mcts_sims", -1)) != int(args.mcts_sims):
        raise RuntimeError("parent evidence MCTS mismatch")
    if parent_payload.get("all_dev_seeds") != dev_seeds:
        raise RuntimeError("parent evidence seed mismatch")
    parent_runs = _runs_by_seed(parent_payload.get("runs", []), dev_seeds[:30])
    gate._ordered(parent_runs, dev_seeds[:30])

    rows: list[dict[str, Any]] = []
    evidence_by_name: dict[str, dict[str, Any]] = {}
    for evidence_path in args.candidate_evidence:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CANDIDATE_EVIDENCE_SCHEMA:
            raise RuntimeError(f"unexpected candidate evidence schema: {evidence_path}")
        if payload.get("parent_weight_sha256") != parent_sha:
            raise RuntimeError("candidate evidence parent mismatch")
        if payload.get("simulator_id") != simulator_id:
            raise RuntimeError("candidate evidence simulator mismatch")
        if int(payload.get("mcts_sims", -1)) != int(args.mcts_sims):
            raise RuntimeError("candidate evidence MCTS mismatch")
        if payload.get("all_dev_seeds") != dev_seeds:
            raise RuntimeError("candidate evidence seed mismatch")
        row = payload.get("candidate")
        if not isinstance(row, dict):
            raise RuntimeError("candidate evidence missing row")
        name = str(row.get("name", ""))
        if not name or name in evidence_by_name:
            raise RuntimeError(f"invalid/duplicate candidate evidence name {name!r}")
        weight = Path(str(row.get("weight", "")))
        if not weight.is_file():
            raise RuntimeError(f"candidate evidence weight missing: {weight}")
        if rt._sha256(weight) != payload.get("candidate_weight_sha256"):
            raise RuntimeError(f"candidate evidence weight SHA mismatch: {name}")
        runs = _runs_by_seed(payload.get("runs", []), dev_seeds[:30])
        gate._ordered(runs, dev_seeds[:30])
        evidence_by_name[name] = {**payload, "_runs_by_seed": runs}
        rows.append(row)
        print(
            "PPO_V15_TOURNAMENT_CANDIDATE_REUSED",
            json.dumps(row, sort_keys=True),
            flush=True,
        )

    winner = choose_winner(rows)
    winner_payload = evidence_by_name[winner["name"]]
    gate._save_cache(
        state,
        key="dev_parent",
        weight_sha=parent_sha,
        seeds=dev_seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
        runs=parent_runs,
    )
    gate._save_cache(
        state,
        key="dev_candidate",
        weight_sha=str(winner_payload["candidate_weight_sha256"]),
        seeds=dev_seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
        runs=winner_payload["_runs_by_seed"],
    )

    summary = {
        "schema_version": "sts1-armg-ppo-v15-population-tournament-v2",
        "parent_weight_sha256": parent_sha,
        "mcts_sims": args.mcts_sims,
        "dev_seeds": 30,
        "parallel_candidate_evaluation": True,
        "reused_by_gate": True,
        "selection_order": [
            "safe_and_complete",
            "existing_dev_gate_pass",
            "victories",
            "mean_final_floor",
            "reach50_delta",
            "stable_candidate_order",
        ],
        "candidates": rows,
        "winner": winner,
        "production_champion_replaced": False,
        "requires_existing_full_gate": True,
    }
    args.state.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "tournament-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "winner.env").write_text(
        f"WINNER_NAME={winner['name']}\nWINNER_WEIGHT={winner['weight']}\n",
        encoding="utf-8",
    )
    print("PPO_V15_TOURNAMENT_WINNER", json.dumps(winner, sort_keys=True), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module-dir", type=Path, required=True)
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--parent-weight", type=Path, required=True)
    p.add_argument("--candidate", action="append", type=_parse_candidate)
    p.add_argument("--parent-evidence", type=Path)
    p.add_argument("--candidate-evidence", action="append", type=Path)
    p.add_argument("--dev-seed-file", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mcts-sims", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    if args.parent_evidence is not None or args.candidate_evidence:
        return _assemble_evidence(args)

    if not args.candidate or len(args.candidate) < 2:
        raise RuntimeError("population tournament requires at least two candidates")
    names = [name for name, _ in args.candidate]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate candidate names")
    for name, path in args.candidate:
        if not path.is_file():
            raise RuntimeError(f"missing candidate {name}: {path}")

    dev_seeds = gate._read_seeds(args.dev_seed_file)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    simulator_id = rt._automatic_simulator_id(args.module_dir)
    parent_sha = rt._sha256(args.parent_weight)
    dev_parent = gate._load_cache(
        state,
        key="dev_parent",
        weight_sha=parent_sha,
        seeds=dev_seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
    )
    dev_parent = gate._run_missing(
        label="v15-dev-parent",
        weight=args.parent_weight,
        seeds=dev_seeds[:30],
        cache=dev_parent,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        output_dir=args.output_dir / "parent",
        mcts_sims=args.mcts_sims,
        all_seeds=dev_seeds,
        workers=args.workers,
    )
    gate._save_cache(
        state,
        key="dev_parent",
        weight_sha=parent_sha,
        seeds=dev_seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
        runs=dev_parent,
    )
    parent_runs = gate._ordered(dev_parent, dev_seeds[:30])

    rows: list[dict[str, Any]] = []
    for index, (name, weight) in enumerate(args.candidate):
        candidate_runs: dict[int, dict[str, Any]] = {}
        candidate_runs = gate._run_missing(
            label=f"v15-dev-{name}",
            weight=weight,
            seeds=dev_seeds[:30],
            cache=candidate_runs,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            output_dir=args.output_dir / name,
            mcts_sims=args.mcts_sims,
            all_seeds=dev_seeds,
            workers=args.workers,
        )
        dev = gate._dev_gate(
            parent_runs,
            gate._ordered(candidate_runs, dev_seeds[:30]),
        )
        row = _candidate_row(
            name=name,
            weight=weight,
            dev_gate=dev,
            index=index,
        )
        rows.append(row)
        print("PPO_V15_TOURNAMENT_CANDIDATE", json.dumps(row, sort_keys=True), flush=True)

    winner = choose_winner(rows)
    summary = {
        "schema_version": "sts1-armg-ppo-v15-population-tournament-v1",
        "parent_weight_sha256": parent_sha,
        "mcts_sims": args.mcts_sims,
        "dev_seeds": 30,
        "selection_order": [
            "safe_and_complete",
            "existing_dev_gate_pass",
            "victories",
            "mean_final_floor",
            "reach50_delta",
            "stable_candidate_order",
        ],
        "candidates": rows,
        "winner": winner,
        "production_champion_replaced": False,
        "requires_existing_full_gate": True,
    }
    args.state.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "tournament-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "winner.env").write_text(
        f"WINNER_NAME={winner['name']}\nWINNER_WEIGHT={winner['weight']}\n",
        encoding="utf-8",
    )
    print("PPO_V15_TOURNAMENT_WINNER", json.dumps(winner, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
