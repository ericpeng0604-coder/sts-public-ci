#!/usr/bin/env python3
"""Evaluate a pure-MCTS + ArmG STS1 A0 Teacher on the fixed 50-seed exam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, _load_sts, run_simulator_game


def read_seeds(path: Path) -> tuple[int, ...]:
    seeds = tuple(int(x.strip()) for x in path.read_text(encoding="utf-8").splitlines()
                  if x.strip() and not x.lstrip().startswith("#"))
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise RuntimeError("Teacher A0 exam requires exactly 50 unique fixed seeds")
    return seeds


def safety(rows: list[dict[str, Any]]) -> dict[str, int]:
    fields = ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count")
    return {f: sum(int(r.get(f, 0) or 0) for r in rows) for f in fields}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--module-dir", type=Path, required=True)
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--armg-weight", type=Path, required=True)
    p.add_argument("--armg-map-weight", type=Path)
    p.add_argument("--seed-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("single", "scheduled", "consensus"), default="single")
    p.add_argument("--mcts-sims", type=int, default=2000)
    p.add_argument("--late-mcts-sims", type=int, default=8000)
    p.add_argument("--late-floor", type=int, default=34)
    p.add_argument("--exploration", type=float)
    p.add_argument("--consensus-budgets", default="1000,2000,4000")
    args = p.parse_args()

    seeds = read_seeds(args.seed_file)
    sts = _load_sts(args.module_dir)
    map_weight = args.armg_map_weight if args.armg_map_weight and args.armg_map_weight.is_file() else None
    armg = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.armg_weight, map_weight_path=map_weight)
    rows: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds, 1):
        kwargs: dict[str, Any] = {}
        if args.mode == "consensus":
            kwargs["combat_mcts_budgets"] = tuple(int(x) for x in args.consensus_budgets.split(",") if x)
        else:
            kwargs["combat_mcts_sims"] = args.mcts_sims
            kwargs["combat_mcts_exploration"] = args.exploration
            if args.mode == "scheduled":
                kwargs["combat_mcts_late_sims"] = args.late_mcts_sims
                kwargs["combat_mcts_late_floor"] = args.late_floor
        row = run_simulator_game(
            student=None, sts=sts, seed=seed, armg_policy=armg,
            heldout_seeds=seeds, collect_teacher=False, **kwargs,
        )
        if row.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
            raise RuntimeError(f"Teacher exam blocked seed={seed}: {row}")
        rows.append(dict(row))
        wins = sum(r.get("outcome") == "victory" for r in rows)
        print(f"TEACHER_PROGRESS {i}/50 wins={wins} rate={wins/i:.3f}", flush=True)

    wins = sum(r.get("outcome") == "victory" for r in rows)
    floors = [float(r["final_floor"]) for r in rows if isinstance(r.get("final_floor"), (int, float))]
    safe = safety(rows)
    report = {
        "schema_version": "sts1-teacher-a0-50-v1",
        "goal": {"ascension": 0, "target_win_rate": 0.50, "fixed_seed_count": 50, "target_wins": 25},
        "mode": args.mode,
        "mcts_sims": args.mcts_sims if args.mode != "consensus" else None,
        "late_mcts_sims": args.late_mcts_sims if args.mode == "scheduled" else None,
        "late_floor": args.late_floor if args.mode == "scheduled" else None,
        "exploration": args.exploration,
        "consensus_budgets": args.consensus_budgets if args.mode == "consensus" else None,
        "seeds": 50,
        "victories": wins,
        "win_rate": wins / 50.0,
        "mean_final_floor": mean(floors) if floors else None,
        "safety": safe,
        "goal_reached_on_fixed_exam": wins >= 25 and all(v == 0 for v in safe.values()),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("TEACHER_A0_50_RESULT", json.dumps({k: report[k] for k in ("victories","win_rate","mean_final_floor","safety","goal_reached_on_fixed_exam")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
