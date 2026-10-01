#!/usr/bin/env python3
"""Screen an isolated STS1 training pool and rank near-win losses for mining."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)

SAFETY = ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count")


def _read(path: Path) -> list[int]:
    seeds=[int(x.strip()) for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if not seeds or len(seeds)!=len(set(seeds)):
        raise RuntimeError("seed file must be non-empty and unique")
    return seeds


def _safe(r: dict[str, Any]) -> bool:
    return (
        r.get("result")=="PASS_SIMULATOR_COMPLETE_RUN"
        and all(int(r.get(k,0) or 0)==0 for k in SAFETY)
    )


def _task(task):
    seed,module_dir,armg_root,weight,heldout=task
    sts=_load_sts(Path(module_dir))
    policy=ArmGNoncombatPolicy(root=Path(armg_root),weight_path=Path(weight))
    result=run_simulator_game(
        student=None,
        sts=sts,
        seed=int(seed),
        evidence_path=None,
        armg_policy=policy,
        combat_mcts_sims=2000,
        heldout_seeds=list(heldout),
    )
    if not _safe(result):
        raise RuntimeError(f"unsafe/incomplete seed {seed}: {result}")
    return {
        "seed":int(seed),
        "outcome":result.get("outcome"),
        "final_floor":int(result.get("final_floor") or result.get("max_floor") or 0),
        "final_hp":int(result.get("final_hp") or 0),
        "max_hp":int(result.get("max_hp") or 0),
        "result":result,
    }


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--weight",type=Path,required=True)
    p.add_argument("--seed-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--priority-output",type=Path,required=True)
    p.add_argument("--priority-count",type=int,default=20)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=8:
        raise RuntimeError("workers outside 1..8")
    seeds=_read(a.seed_file)
    if not 1<=a.priority_count<=len(seeds):
        raise RuntimeError("priority-count outside valid range")
    tasks=[(s,str(a.module_dir),str(a.armg_root),str(a.weight),tuple(seeds)) for s in seeds]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(tasks))) as pool:
        rows=list(pool.map(_task,tasks))
    by={r["seed"]:r for r in rows}
    rows=[by[s] for s in seeds]
    wins=[r for r in rows if str(r["outcome"]).lower()=="victory"]
    losses=[r for r in rows if str(r["outcome"]).lower()!="victory"]
    losses.sort(key=lambda r:(int(r["final_floor"]),int(r["final_hp"])),reverse=True)
    priority=losses[:a.priority_count]
    payload={
        "schema_version":"sts1-build-rescue-screen-v31",
        "seeds":len(rows),
        "wins":len(wins),
        "losses":len(losses),
        "priority_count":len(priority),
        "priority_seeds":[int(r["seed"]) for r in priority],
        "priority_rows":priority,
        "rows":rows,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    a.priority_output.parent.mkdir(parents=True,exist_ok=True)
    a.priority_output.write_text("\n".join(str(r["seed"]) for r in priority)+"\n")
    print("V31_SCREEN_PASS",json.dumps({
        "seeds":len(rows),
        "wins":len(wins),
        "losses":len(losses),
        "priority_seeds":[int(r["seed"]) for r in priority],
        "priority_floors":[int(r["final_floor"]) for r in priority],
    },sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
