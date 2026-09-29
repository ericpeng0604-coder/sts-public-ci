#!/usr/bin/env python3
"""Fail-closed fast path for reusing PPO v1.5 Parent Dev evaluation.

This helper intentionally uses only repository-local files plus the pinned
simulator commit ID. It can therefore validate a persisted Parent cache before
hydrating external dependencies or installing Torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_SCHEMA = "sts1-armg-ppo-v14-eval-cache-v1"
EVIDENCE_SCHEMA = "sts1-armg-ppo-v15-parent-eval-v1"
MISS_EXIT = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def simulator_id(sim_head: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return ":".join(
        [
            sim_head,
            sha256(root / "src" / "roguelike_ai" / "sts1_phase3" / "simulator.py"),
            sha256(root / "src" / "roguelike_ai" / "sts1_phase3" / "champion_gate.py"),
        ]
    )


def _safe_complete(row: dict[str, Any]) -> bool:
    if row.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
        return False
    return all(
        int(row.get(key, 0) or 0) == 0
        for key in (
            "illegal_action_count",
            "crash_count",
            "timeout_count",
            "remote_error_count",
        )
    )


def materialize_cached_parent(
    *,
    state_path: Path,
    parent_weight: Path,
    output_dir: Path,
    mcts_sims: int,
    sim_head: str,
) -> bool:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    cache = (state.get("v14_eval_caches") or {}).get("dev_parent")
    if not isinstance(cache, dict) or cache.get("schema_version") != CACHE_SCHEMA:
        return False

    seeds_raw = cache.get("seeds")
    if not isinstance(seeds_raw, list):
        return False
    try:
        seeds = [int(seed) for seed in seeds_raw]
    except (TypeError, ValueError):
        return False
    if len(seeds) != 50 or len(set(seeds)) != 50:
        return False

    expected_simulator = simulator_id(sim_head)
    parent_sha = sha256(parent_weight)
    if cache.get("weight_sha256") != parent_sha:
        return False
    if int(cache.get("mcts_sims", -1)) != int(mcts_sims):
        return False
    if cache.get("simulator_id") != expected_simulator:
        return False

    by_seed: dict[int, dict[str, Any]] = {}
    for row in cache.get("runs", []):
        try:
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            return False
        if seed in by_seed or seed not in set(seeds):
            return False
        if not _safe_complete(row):
            return False
        by_seed[seed] = dict(row)

    try:
        runs = [by_seed[seed] for seed in seeds[:30]]
    except KeyError:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "parent_weight_sha256": parent_sha,
        "simulator_id": expected_simulator,
        "mcts_sims": int(mcts_sims),
        "all_dev_seeds": seeds,
        "evaluated_seeds": seeds[:30],
        "cache_hits_before": 30,
        "fresh_runs": 0,
        "runs": runs,
    }
    (output_dir / "parent-eval.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--parent-weight", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mcts-sims", type=int, default=2000)
    p.add_argument("--sim-head", required=True)
    args = p.parse_args()

    hit = materialize_cached_parent(
        state_path=args.state,
        parent_weight=args.parent_weight,
        output_dir=args.output_dir,
        mcts_sims=args.mcts_sims,
        sim_head=args.sim_head,
    )
    if not hit:
        print("PPO_V15_PARENT_FAST_CACHE_MISS", flush=True)
        return MISS_EXIT
    print("PPO_V15_PARENT_FAST_CACHE_HIT runs=30", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
