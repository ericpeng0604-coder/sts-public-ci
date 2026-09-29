#!/usr/bin/env python3
"""Run an exact-seed paired ArmG benchmark shard."""
import argparse, json
from pathlib import Path
from statistics import mean

from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module-dir", type=Path, required=True)
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--baseline-weight", type=Path, required=True)
    p.add_argument("--candidate-weight", type=Path, required=True)
    p.add_argument("--seed-file", type=Path, required=True)
    p.add_argument("--start-index", type=int, required=True)
    p.add_argument("--seed-count", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mcts-sims", type=int, default=2000)
    a = p.parse_args()

    seeds = [
        int(x.strip())
        for x in a.seed_file.read_text().splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]
    assert len(seeds) == len(set(seeds)), "seed file contains duplicates"
    selected = seeds[a.start_index:a.start_index + a.seed_count]
    assert len(selected) == a.seed_count, (a.start_index, a.seed_count, len(selected))

    sts = _load_sts(a.module_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, seed in enumerate(selected, 1):
        pair = {}
        for label, weight in (
            ("baseline", a.baseline_weight),
            ("candidate", a.candidate_weight),
        ):
            policy = ArmGNoncombatPolicy(root=a.armg_root, weight_path=weight)
            out = a.output_dir / label
            out.mkdir(exist_ok=True)
            result = run_simulator_game(
                student=None,
                sts=sts,
                seed=seed,
                evidence_path=out / f"seed-{seed}.ndjson",
                armg_policy=policy,
                combat_mcts_sims=a.mcts_sims,
                training_seeds=selected,
                collect_teacher=True,
            )
            if result.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
                raise RuntimeError((label, seed, result))
            pair[label] = dict(result)

        rows.append({"seed": seed, **pair})
        print(
            f"PAIR_SHARD {i}/{len(selected)} seed={seed} "
            f"g3={pair['baseline'].get('outcome')}:{pair['baseline'].get('final_floor')} "
            f"g7={pair['candidate'].get('outcome')}:{pair['candidate'].get('final_floor')}",
            flush=True,
        )

    def stats(key):
        rs = [r[key] for r in rows]
        wins = sum(r.get("outcome") == "victory" for r in rs)
        floors = [
            r.get("final_floor")
            for r in rs
            if isinstance(r.get("final_floor"), (int, float))
        ]
        return {
            "wins": wins,
            "win_rate": wins / len(rs),
            "mean_final_floor": mean(floors) if floors else None,
        }

    report = {
        "schema_version": "sts1-paired-seed-shard-v1",
        "start_index": a.start_index,
        "seed_count": len(selected),
        "mcts_sims": a.mcts_sims,
        "baseline": stats("baseline"),
        "candidate": stats("candidate"),
        "rows": rows,
    }
    (a.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
