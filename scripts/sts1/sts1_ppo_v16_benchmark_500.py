#!/usr/bin/env python3
"""Run/aggregate a pinned 500-game PPO v1.6 held-out benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))



def read_seeds(path: Path) -> list[int]:
    seeds = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != 500 or len(set(seeds)) != 500:
        raise RuntimeError("benchmark seed file must contain exactly 500 unique seeds")
    return seeds


def safety(run: dict) -> dict[str, int]:
    return {
        key: int(run.get(key, 0) or 0)
        for key in ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count")
    }


def shard(args: argparse.Namespace) -> int:
    import sts1_armg_ppo_v13_auto_gate as rt

    seeds = read_seeds(args.seeds_file)
    selected = seeds[args.shard_index::args.shard_count]
    if not selected:
        raise RuntimeError("empty benchmark shard")
    sha = rt._sha256(args.weight)
    if args.expected_weight_sha and sha != args.expected_weight_sha:
        raise RuntimeError(f"weight SHA mismatch: {sha} != {args.expected_weight_sha}")
    simulator_id = rt._automatic_simulator_id(args.module_dir)
    label = f"ppo-v16-g7-benchmark-shard-{args.shard_index:02d}"
    requests = [(label, args.weight, seed) for seed in selected]
    results = rt._evaluate_parallel(
        requests=requests,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        output_dir=args.output_dir / "evidence",
        mcts_sims=args.mcts_sims,
        all_formal_seeds=seeds,
        workers=args.workers,
    )
    runs = [results[(label, seed)] for seed in selected]
    if len(runs) != len(selected):
        raise RuntimeError("benchmark shard incomplete")
    if any(any(v != 0 for v in safety(run).values()) for run in runs):
        raise RuntimeError("benchmark shard safety failure")
    payload = {
        "schema_version": "sts1-ppo-v16-benchmark-500-shard-v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "games": len(runs),
        "wins": sum(str(run.get("outcome", "")).lower() == "victory" for run in runs),
        "weight_sha256": sha,
        "simulator_id": simulator_id,
        "mcts_sims": args.mcts_sims,
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"shard-{args.shard_index:02d}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PPO_V16_BENCHMARK_SHARD", json.dumps({
        "shard": args.shard_index,
        "games": payload["games"],
        "wins": payload["wins"],
        "weight_sha256": sha,
    }, sort_keys=True), flush=True)
    return 0


def wilson(wins: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = wins / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / d
    return center-half, center+half


def aggregate(args: argparse.Namespace) -> int:
    seeds = read_seeds(args.seeds_file)
    files = sorted(args.input_dir.glob("**/shard-*.json"))
    if len(files) != args.shard_count:
        raise RuntimeError(f"expected {args.shard_count} shard reports, found {len(files)}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    indexes = {int(r["shard_index"]) for r in reports}
    if indexes != set(range(args.shard_count)):
        raise RuntimeError(f"shard index mismatch: {sorted(indexes)}")
    identities = {
        (r["weight_sha256"], r["simulator_id"], int(r["mcts_sims"]))
        for r in reports
    }
    if len(identities) != 1:
        raise RuntimeError("benchmark identity drift across shards")
    runs = [run for r in reports for run in r["runs"]]
    by_seed = {int(run["seed"]): run for run in runs}
    if len(runs) != 500 or set(by_seed) != set(seeds):
        raise RuntimeError("benchmark seed coverage mismatch")
    safety_totals = {k: 0 for k in ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count")}
    for run in runs:
        for key, value in safety(run).items():
            safety_totals[key] += value
    if any(safety_totals.values()):
        raise RuntimeError(f"benchmark safety failure: {safety_totals}")
    wins = sum(str(run.get("outcome", "")).lower() == "victory" for run in runs)
    floors = [float(run.get("final_floor") or 0) for run in runs]
    lo, hi = wilson(wins, 500)
    weight_sha, simulator_id, mcts_sims = next(iter(identities))
    summary = {
        "schema_version": "sts1-ppo-v16-benchmark-500-v1",
        "games": 500,
        "wins": wins,
        "defeats": 500-wins,
        "win_rate": wins/500,
        "win_rate_percent": wins/5,
        "wilson_95_low": lo,
        "wilson_95_high": hi,
        "mean_final_floor": statistics.mean(floors),
        "median_final_floor": statistics.median(floors),
        "weight_sha256": weight_sha,
        "simulator_id": simulator_id,
        "mcts_sims": mcts_sims,
        "safety": safety_totals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PPO_V16_BENCHMARK_500_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return 0


def main() -> int:
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode", required=True)
    s=sub.add_parser("shard")
    s.add_argument("--module-dir", type=Path, required=True)
    s.add_argument("--armg-root", type=Path, required=True)
    s.add_argument("--weight", type=Path, required=True)
    s.add_argument("--expected-weight-sha")
    s.add_argument("--seeds-file", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--shard-index", type=int, required=True)
    s.add_argument("--shard-count", type=int, default=20)
    s.add_argument("--mcts-sims", type=int, default=2000)
    s.add_argument("--workers", type=int, default=4)
    a=sub.add_parser("aggregate")
    a.add_argument("--input-dir", type=Path, required=True)
    a.add_argument("--seeds-file", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--shard-count", type=int, default=20)
    args=p.parse_args()
    if args.mode=="shard":
        if args.shard_count != 20:
            raise RuntimeError("benchmark is pinned to 20 shards")
        if not 0 <= args.shard_index < args.shard_count:
            raise RuntimeError("invalid shard index")
        if args.mcts_sims != 2000:
            raise RuntimeError("benchmark is pinned to MCTS-2000")
        if not 1 <= args.workers <= 8:
            raise RuntimeError("workers must be 1..8")
        return shard(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
