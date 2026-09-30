#!/usr/bin/env python3
"""Scout fresh training seeds with frozen G7 + formal MCTS-2000."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, _load_sts, run_simulator_game

SAFETY=("illegal_action_count","crash_count","timeout_count","remote_error_count")

def _read(path:Path)->list[int]:
    xs=[int(x.strip()) for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if not xs or len(xs)!=len(set(xs)): raise RuntimeError("seed file invalid")
    return xs

def _safe(r:dict[str,Any])->bool:
    return r.get("result")=="PASS_SIMULATOR_COMPLETE_RUN" and all(int(r.get(k,0) or 0)==0 for k in SAFETY)

def _task(t):
    seed,module_dir,root,weight,all_seeds=t
    sts=_load_sts(Path(module_dir))
    policy=ArmGNoncombatPolicy(root=Path(root),weight_path=Path(weight))
    r=run_simulator_game(
        student=None,sts=sts,seed=seed,evidence_path=None,armg_policy=policy,
        combat_mcts_sims=2000,training_seeds=list(all_seeds)
    )
    if not _safe(r): raise RuntimeError(f"unsafe scout seed {seed}: {r}")
    return {
        "seed":seed,
        "outcome":r.get("outcome"),
        "final_floor":int(r.get("final_floor") or r.get("max_floor") or 0),
        "final_hp":int(r.get("final_hp") or 0),
        "max_act":int(r.get("max_act") or 0),
    }

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--weight",type=Path,required=True)
    p.add_argument("--seed-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--priority-output",type=Path,required=True)
    p.add_argument("--fallback-output",type=Path,required=True)
    p.add_argument("--priority-count",type=int,default=12)
    p.add_argument("--fallback-count",type=int,default=12)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    seeds=_read(a.seed_file)
    tasks=[(s,str(a.module_dir),str(a.armg_root),str(a.weight),tuple(seeds)) for s in seeds]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(tasks))) as pool:
        rows=list(pool.map(_task,tasks))
    losses=[r for r in rows if str(r["outcome"]).lower()!="victory"]
    losses.sort(key=lambda r:(r["final_floor"],r["max_act"],r["final_hp"]),reverse=True)
    priority=[r["seed"] for r in losses[:a.priority_count]]
    fallback=[r["seed"] for r in losses[a.priority_count:a.priority_count+a.fallback_count]]
    if len(priority)<min(4,a.priority_count):
        raise RuntimeError(f"too few failed training seeds to mine: {len(priority)}")
    payload={
        "schema_version":"sts1-v31-formal-scout",
        "seeds":len(seeds),
        "wins":sum(str(r["outcome"]).lower()=="victory" for r in rows),
        "losses":len(losses),
        "priority_seeds":priority,
        "fallback_seeds":fallback,
        "rows":rows,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    a.priority_output.write_text("\n".join(map(str,priority))+"\n")
    a.fallback_output.write_text("\n".join(map(str,fallback))+"\n")
    print("V31_FORMAL_SCOUT",json.dumps({
        "seeds":len(seeds),"wins":payload["wins"],"losses":payload["losses"],
        "priority_seeds":priority,"fallback_seeds":fallback
    },sort_keys=True),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
