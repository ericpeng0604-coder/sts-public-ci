#!/usr/bin/env python3
"""Diagnose whether stronger Boss-only MCTS rescues PPO G7 losses."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, _load_sts, run_simulator_game

BOSS_FLOORS = (16, 33, 50)
MODES = {
    "base_2k": None,
    "boss_10k": 10_000,
    "boss_50k": 50_000,
}
SAFETY_KEYS = ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count")


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_seeds(path: Path) -> list[int]:
    seeds=[
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise RuntimeError("diagnostic seed file must contain exactly 50 unique seeds")
    return seeds


def _one_mode(
    *,
    seed: int,
    mode: str,
    module_dir: Path,
    armg_root: Path,
    weight: Path,
    output_dir: Path,
    heldout: list[int],
) -> dict[str, Any]:
    sts=_load_sts(module_dir)
    policy=ArmGNoncombatPolicy(root=armg_root, weight_path=weight)
    boss_sims=MODES[mode]
    out=output_dir / mode
    out.mkdir(parents=True, exist_ok=True)
    result=run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=out / f"seed-{seed}.ndjson",
        armg_policy=policy,
        combat_mcts_sims=2000,
        combat_mcts_boss_sims=boss_sims,
        combat_mcts_boss_floors=BOSS_FLOORS,
        heldout_seeds=heldout,
    )
    if result.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
        raise RuntimeError(f"{mode} seed {seed} incomplete: {result}")
    if any(int(result.get(key,0) or 0) != 0 for key in SAFETY_KEYS):
        raise RuntimeError(f"{mode} seed {seed} safety failure: {result}")
    return dict(result)


def _seed_task(task: tuple[int,str,str,str,str,tuple[int,...]]) -> dict[str, Any]:
    seed,module_dir,armg_root,weight,output_dir,heldout=task
    rows={}
    for mode in MODES:
        rows[mode]=_one_mode(
            seed=seed,
            mode=mode,
            module_dir=Path(module_dir),
            armg_root=Path(armg_root),
            weight=Path(weight),
            output_dir=Path(output_dir) / f"seed-{seed}",
            heldout=list(heldout),
        )
    return {"seed":seed,"modes":rows}


def shard(args: argparse.Namespace) -> int:
    seeds=_read_seeds(args.seeds_file)
    selected=seeds[args.shard_index::args.shard_count]
    if not selected:
        raise RuntimeError("empty diagnostic shard")
    tasks=[
        (
            seed,
            str(args.module_dir),
            str(args.armg_root),
            str(args.weight),
            str(args.output_dir / "evidence"),
            tuple(seeds),
        )
        for seed in selected
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks))) as pool:
        rows=list(pool.map(_seed_task,tasks))
    payload={
        "schema_version":"sts1-ppo-v20-boss-search-diagnostic-shard-v1",
        "shard_index":args.shard_index,
        "shard_count":args.shard_count,
        "weight_sha256":_sha256(args.weight),
        "boss_floors":list(BOSS_FLOORS),
        "modes":{"base_2k":2000,"boss_10k":10000,"boss_50k":50000},
        "rows":rows,
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    p=args.output_dir / f"shard-{args.shard_index:02d}.json"
    p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("PPO_V20_BOSS_DIAGNOSTIC_SHARD",json.dumps({
        "shard":args.shard_index,
        "seeds":len(rows),
        "base_wins":sum(str(r["modes"]["base_2k"].get("outcome","")).lower()=="victory" for r in rows),
        "boss10_wins":sum(str(r["modes"]["boss_10k"].get("outcome","")).lower()=="victory" for r in rows),
        "boss50_wins":sum(str(r["modes"]["boss_50k"].get("outcome","")).lower()=="victory" for r in rows),
    },sort_keys=True),flush=True)
    return 0


def _is_win(row: dict[str,Any]) -> bool:
    return str(row.get("outcome","")).lower()=="victory"


def aggregate(args: argparse.Namespace) -> int:
    seeds=_read_seeds(args.seeds_file)
    files=sorted(args.input_dir.glob("**/shard-*.json"))
    if len(files) != args.shard_count:
        raise RuntimeError(f"expected {args.shard_count} shard reports, got {len(files)}")
    reports=[json.loads(p.read_text(encoding="utf-8")) for p in files]
    rows=[r for report in reports for r in report["rows"]]
    by_seed={int(r["seed"]):r for r in rows}
    if set(by_seed) != set(seeds):
        raise RuntimeError("diagnostic seed coverage mismatch")
    weight_shas={r["weight_sha256"] for r in reports}
    if len(weight_shas)!=1:
        raise RuntimeError("weight identity drift")

    summary={
        "schema_version":"sts1-ppo-v20-boss-search-diagnostic-v1",
        "games_per_mode":len(seeds),
        "weight_sha256":next(iter(weight_shas)),
        "boss_floors":list(BOSS_FLOORS),
        "modes":{},
        "paired_rescues":{},
    }
    for mode in MODES:
        runs=[by_seed[s]["modes"][mode] for s in seeds]
        wins=sum(_is_win(r) for r in runs)
        floors=[int(r.get("final_floor") or 0) for r in runs]
        summary["modes"][mode]={
            "wins":wins,
            "win_rate":wins/len(runs),
            "mean_final_floor":statistics.mean(floors),
            "median_final_floor":statistics.median(floors),
        }

    base={s:_is_win(by_seed[s]["modes"]["base_2k"]) for s in seeds}
    for mode in ("boss_10k","boss_50k"):
        rescued=[
            s for s in seeds
            if not base[s] and _is_win(by_seed[s]["modes"][mode])
        ]
        regressed=[
            s for s in seeds
            if base[s] and not _is_win(by_seed[s]["modes"][mode])
        ]
        baseline_losses=sum(not base[s] for s in seeds)
        summary["paired_rescues"][mode]={
            "rescued_losses":len(rescued),
            "regressed_wins":len(regressed),
            "net_win_delta":len(rescued)-len(regressed),
            "rescue_fraction_of_baseline_losses":(
                len(rescued)/baseline_losses if baseline_losses else 0.0
            ),
            "rescued_seeds":rescued,
            "regressed_seeds":regressed,
        }

    build_limited=[
        s for s in seeds
        if not base[s] and not _is_win(by_seed[s]["modes"]["boss_50k"])
    ]
    summary["build_limited_after_boss_50k"]={
        "count":len(build_limited),
        "seeds":build_limited,
        "fraction_of_baseline_losses":(
            len(build_limited)/sum(not base[s] for s in seeds)
            if any(not base[s] for s in seeds)
            else 0.0
        ),
    }

    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("PPO_V20_BOSS_DIAGNOSTIC_RESULT",json.dumps(summary,sort_keys=True),flush=True)
    return 0


def main() -> int:
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode",required=True)
    s=sub.add_parser("shard")
    for name in ("module-dir","armg-root","weight","seeds-file","output-dir"):
        s.add_argument("--"+name,type=Path,required=True)
    s.add_argument("--shard-index",type=int,required=True)
    s.add_argument("--shard-count",type=int,default=10)
    s.add_argument("--workers",type=int,default=2)
    a=sub.add_parser("aggregate")
    a.add_argument("--input-dir",type=Path,required=True)
    a.add_argument("--seeds-file",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--shard-count",type=int,default=10)
    args=p.parse_args()
    if args.mode=="shard":
        if args.shard_count != 10:
            raise RuntimeError("diagnostic is pinned to 10 shards")
        if not 0 <= args.shard_index < args.shard_count:
            raise RuntimeError("invalid shard index")
        if not 1 <= args.workers <= 4:
            raise RuntimeError("workers must be within 1..4")
        return shard(args)
    return aggregate(args)


if __name__=="__main__":
    raise SystemExit(main())
