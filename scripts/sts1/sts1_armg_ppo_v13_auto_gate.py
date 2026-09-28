#!/usr/bin/env python3
"""Paired fixed-seed gate for the STS1 ArmG PPO v1.3 auto loop.

The current offline Champion and Candidate are evaluated on the exact same
formal seeds. The 30-seed fast gate must pass before the remaining 20 seeds
are evaluated for the formal 50-seed gate. HOLD is a normal result and exits
successfully; malformed/incomplete evidence raises and fails the workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.champion_gate import (
    FAST_GATE_POLICY,
    FORMAL_GATE_POLICY,
    evaluate_fixed_seed_gate,
)
from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_seeds(path: Path) -> list[int]:
    seeds = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise RuntimeError("formal seed file must contain exactly 50 unique seeds")
    return seeds


def _run_one(
    *,
    label: str,
    weight: Path,
    seed: int,
    sts: Any,
    armg_root: Path,
    evidence_root: Path,
    mcts_sims: int,
    heldout_seeds: list[int],
) -> dict[str, Any]:
    policy = ArmGNoncombatPolicy(root=armg_root, weight_path=weight)
    out = evidence_root / label
    out.mkdir(parents=True, exist_ok=True)
    result = run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=out / f"seed-{seed}.ndjson",
        armg_policy=policy,
        combat_mcts_sims=mcts_sims,
        heldout_seeds=heldout_seeds,
        collect_teacher=True,
    )
    if result.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
        raise RuntimeError(f"{label} seed {seed} incomplete: {result}")
    return dict(result)


def _evaluate_range(
    *,
    seeds: list[int],
    current_weight: Path,
    candidate_weight: Path,
    sts: Any,
    armg_root: Path,
    output_dir: Path,
    mcts_sims: int,
    all_formal_seeds: list[int],
    current_runs: list[dict[str, Any]],
    candidate_runs: list[dict[str, Any]],
) -> None:
    for seed in seeds:
        current = _run_one(
            label="current",
            weight=current_weight,
            seed=seed,
            sts=sts,
            armg_root=armg_root,
            evidence_root=output_dir,
            mcts_sims=mcts_sims,
            heldout_seeds=all_formal_seeds,
        )
        candidate = _run_one(
            label="candidate",
            weight=candidate_weight,
            seed=seed,
            sts=sts,
            armg_root=armg_root,
            evidence_root=output_dir,
            mcts_sims=mcts_sims,
            heldout_seeds=all_formal_seeds,
        )
        current_runs.append(current)
        candidate_runs.append(candidate)
        print(
            "PPO_V13_GATE",
            json.dumps(
                {
                    "done": len(current_runs),
                    "total": 50,
                    "seed": seed,
                    "current": {
                        "outcome": current.get("outcome"),
                        "final_floor": current.get("final_floor"),
                    },
                    "candidate": {
                        "outcome": candidate.get("outcome"),
                        "final_floor": candidate.get("final_floor"),
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _write_runs(path: Path, runs: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--current-weight", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=Path, required=True)
    parser.add_argument("--formal-seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mcts-sims", type=int, default=2000)
    args = parser.parse_args()

    seeds = _read_seeds(args.formal_seed_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sts = _load_sts(args.module_dir)

    current_runs: list[dict[str, Any]] = []
    candidate_runs: list[dict[str, Any]] = []

    _evaluate_range(
        seeds=seeds[:30],
        current_weight=args.current_weight,
        candidate_weight=args.candidate_weight,
        sts=sts,
        armg_root=args.armg_root,
        output_dir=args.output_dir,
        mcts_sims=args.mcts_sims,
        all_formal_seeds=seeds,
        current_runs=current_runs,
        candidate_runs=candidate_runs,
    )
    gate30 = evaluate_fixed_seed_gate(
        current_runs,
        candidate_runs,
        policy=FAST_GATE_POLICY,
    )

    if gate30["status"] == "PASS":
        _evaluate_range(
            seeds=seeds[30:],
            current_weight=args.current_weight,
            candidate_weight=args.candidate_weight,
            sts=sts,
            armg_root=args.armg_root,
            output_dir=args.output_dir,
            mcts_sims=args.mcts_sims,
            all_formal_seeds=seeds,
            current_runs=current_runs,
            candidate_runs=candidate_runs,
        )
        gate50 = evaluate_fixed_seed_gate(
            current_runs,
            candidate_runs,
            policy=FORMAL_GATE_POLICY,
        )
    else:
        gate50 = {
            "schema_version": "sts1-champion-gate-v1",
            "gate": "fixed-seed-50",
            "status": "SKIPPED",
            "reasons": ["fast_30_seed_gate_did_not_pass"],
        }

    decision = (
        "PROMOTE_OFFLINE"
        if gate30["status"] == "PASS" and gate50["status"] == "PASS"
        else "HOLD"
    )
    report = {
        "schema_version": "sts1-armg-ppo-v13-auto-gate-v1",
        "decision": decision,
        "production_champion_replaced": False,
        "real_game_gate_required_for_production": True,
        "current_weight_sha256": _sha256(args.current_weight),
        "candidate_weight_sha256": _sha256(args.candidate_weight),
        "mcts_sims": args.mcts_sims,
        "gate_30": gate30,
        "gate_50": gate50,
    }

    _write_runs(args.output_dir / "current-runs.json", current_runs)
    _write_runs(args.output_dir / "candidate-runs.json", candidate_runs)
    (args.output_dir / "gate-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PPO_V13_AUTO_GATE_RESULT", json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
