#!/usr/bin/env python3
"""Mine winning and near-winning Teacher trajectories on fresh A0 seeds."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--armg-weight", type=Path, required=True)
    parser.add_argument("--armg-map-weight", type=Path)
    parser.add_argument("--formal-seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=20261001)
    parser.add_argument("--mcts-sims", type=int, default=2000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    formal = {
        int(line.strip())
        for line in args.formal_seed_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    rng = random.Random(args.rng_seed)
    generated: list[int] = []
    seen = set(formal)
    while len(generated) < args.seed_count:
        seed = rng.randint(1, 10**9)
        if seed not in seen:
            seen.add(seed)
            generated.append(seed)

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")

    seeds = generated[args.shard_index :: args.shard_count]
    if not seeds:
        raise ValueError("empty shard")

    sts = _load_sts(args.module_dir)
    map_weight = (
        args.armg_map_weight
        if args.armg_map_weight and args.armg_map_weight.is_file()
        else None
    )
    armg = ArmGNoncombatPolicy(
        root=args.armg_root,
        weight_path=args.armg_weight,
        map_weight_path=map_weight,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict] = []
    winners: list[dict] = []
    near_wins: list[dict] = []
    blocked: list[dict] = []

    for index, seed in enumerate(seeds, 1):
        evidence_path = args.output_dir / f"seed-{seed}.ndjson"
        row = dict(
            run_simulator_game(
                student=None,
                sts=sts,
                seed=seed,
                evidence_path=evidence_path,
                armg_policy=armg,
                combat_mcts_sims=args.mcts_sims,
                training_seeds=seeds,
                collect_teacher=True,
            )
        )

        if row.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
            blocked_path = args.output_dir / f"blocked-seed-{seed}.ndjson"
            if evidence_path.exists():
                evidence_path.replace(blocked_path)
            blocked.append(
                {
                    "seed": seed,
                    "result": row.get("result"),
                    "error": row.get("error"),
                    "final_floor": row.get("final_floor"),
                    "game_steps": row.get("game_steps"),
                    "evidence": blocked_path.name if blocked_path.exists() else None,
                }
            )
            print(
                f"WINNER_MINING_BLOCKED {index}/{len(seeds)} "
                f"seed={seed} result={row.get('result')} "
                f"floor={row.get('final_floor')} error={row.get('error')}",
                flush=True,
            )
            continue

        completed.append(row)
        if row.get("outcome") == "victory":
            winners.append(
                {"seed": seed, "summary": row, "evidence": evidence_path.name}
            )
        elif int(row.get("final_floor") or 0) >= 45:
            near_wins.append(
                {"seed": seed, "summary": row, "evidence": evidence_path.name}
            )

        print(
            f"WINNER_MINING {index}/{len(seeds)} "
            f"completed={len(completed)} blocked={len(blocked)} "
            f"wins={len(winners)} near={len(near_wins)} "
            f"floor={row.get('final_floor')}",
            flush=True,
        )

    if not completed:
        raise RuntimeError(
            f"all seeds blocked in shard {args.shard_index}; refusing empty mining report"
        )

    floors = [
        float(row["final_floor"])
        for row in completed
        if isinstance(row.get("final_floor"), (int, float))
    ]
    report = {
        "schema_version": "sts1-teacher-winner-mining-v2",
        "seed_source": "fresh_random_excluding_formal_50",
        "seed_count": len(completed),
        "completed_seed_count": len(completed),
        "attempted_seed_count": len(seeds),
        "blocked_seed_count": len(blocked),
        "requested_seed_count": args.seed_count,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "mcts_sims": args.mcts_sims,
        "victories": len(winners),
        "win_rate": len(winners) / len(completed),
        "near_wins": len(near_wins),
        "mean_final_floor": mean(floors) if floors else None,
        "winning_seeds": [item["seed"] for item in winners],
        "near_win_seeds": [item["seed"] for item in near_wins],
        "blocked_seeds": [item["seed"] for item in blocked],
        "winners": winners,
        "near_wins_detail": near_wins,
        "blocked_detail": blocked,
    }
    (args.output_dir / "winner-mining-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        "WINNER_MINING_RESULT",
        json.dumps(
            {
                key: report[key]
                for key in (
                    "attempted_seed_count",
                    "completed_seed_count",
                    "blocked_seed_count",
                    "victories",
                    "win_rate",
                    "near_wins",
                    "mean_final_floor",
                    "winning_seeds",
                    "blocked_seeds",
                )
            },
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
