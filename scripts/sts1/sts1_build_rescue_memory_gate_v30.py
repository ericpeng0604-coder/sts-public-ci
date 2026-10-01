#!/usr/bin/env python3
"""Paired fixed-seed gate: frozen G7 vs G7 + exact Rescue Memory."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.rescue_memory import ArmGRescueMemoryPolicy
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, _load_sts, run_simulator_game

SAFETY=("illegal_action_count","crash_count","timeout_count","remote_error_count")


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _read_seeds(path:Path)->list[int]:
    seeds=[int(x.strip()) for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if not seeds or len(seeds)!=len(set(seeds)):
        raise RuntimeError("seed file must be non-empty and unique")
    return seeds


def _is_win(r:dict[str,Any])->bool:
    return str(r.get("outcome","")).lower()=="victory"


def _safe(r:dict[str,Any])->bool:
    return r.get("result")=="PASS_SIMULATOR_COMPLETE_RUN" and all(int(r.get(k,0) or 0)==0 for k in SAFETY)


def _pair_task(task):
    seed,module_dir,armg_root,weight,memory,mcts_sims,boss_mcts_sims,boss_floors,heldout=task
    sts=_load_sts(Path(module_dir))
    parent=ArmGNoncombatPolicy(root=Path(armg_root),weight_path=Path(weight))
    candidate=ArmGRescueMemoryPolicy(root=Path(armg_root),weight_path=Path(weight),memory_path=Path(memory))
    kwargs=dict(
        student=None,
        sts=sts,
        seed=int(seed),
        evidence_path=None,
        combat_mcts_sims=int(mcts_sims),
        heldout_seeds=list(heldout),
    )
    if boss_mcts_sims is not None:
        kwargs["combat_mcts_boss_sims"]=int(boss_mcts_sims)
        kwargs["combat_mcts_boss_floors"]=tuple(int(x) for x in boss_floors)
    pr=run_simulator_game(armg_policy=parent,**kwargs)
    cr=run_simulator_game(armg_policy=candidate,**kwargs)
    if not _safe(pr) or not _safe(cr):
        raise RuntimeError(f"unsafe/incomplete paired seed {seed}: parent={pr} candidate={cr}")
    return {
        "seed":int(seed),
        "parent":pr,
        "candidate":cr,
        "candidate_memory_hits":int(candidate.memory_hits),
    }


def _sign_pvalue(c:int,p:int)->float:
    n=c+p
    if n<=0:
        return 1.0
    return sum(math.comb(n,k) for k in range(c,n+1))/(2.0**n)


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--base-weight",type=Path,required=True)
    p.add_argument("--memory",type=Path,required=True)
    p.add_argument("--seed-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mcts-sims",type=int,default=2000)
    p.add_argument("--boss-mcts-sims",type=int)
    p.add_argument("--boss-floors",default="16,33,50")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--min-win-delta",type=int,default=1)
    p.add_argument("--max-pvalue",type=float,default=1.0)
    p.add_argument("--label",default="memory-gate")
    a=p.parse_args()
    if not 1<=a.workers<=8:
        raise RuntimeError("workers outside 1..8")
    if a.mcts_sims<1:
        raise RuntimeError("mcts-sims must be positive")
    if a.boss_mcts_sims is not None and a.boss_mcts_sims<1:
        raise RuntimeError("boss-mcts-sims must be positive")
    if not 0<a.max_pvalue<=1:
        raise RuntimeError("max-pvalue outside (0,1]")
    boss_floors=tuple(int(x) for x in str(a.boss_floors).split(",") if x.strip())
    if a.boss_mcts_sims is not None and not boss_floors:
        raise RuntimeError("boss-floors cannot be empty with boss-mcts-sims")
    seeds=_read_seeds(a.seed_file)
    heldout=tuple(seeds)
    tasks=[
        (
            s,str(a.module_dir),str(a.armg_root),str(a.base_weight),str(a.memory),
            a.mcts_sims,a.boss_mcts_sims,boss_floors,heldout
        )
        for s in seeds
    ]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(tasks))) as pool:
        rows=list(pool.map(_pair_task,tasks))
    by={r["seed"]:r for r in rows}
    rows=[by[s] for s in seeds]
    pw=sum(_is_win(r["parent"]) for r in rows)
    cw=sum(_is_win(r["candidate"]) for r in rows)
    co=sum((not _is_win(r["parent"])) and _is_win(r["candidate"]) for r in rows)
    po=sum(_is_win(r["parent"]) and (not _is_win(r["candidate"])) for r in rows)
    hits=sum(r["candidate_memory_hits"] for r in rows)
    pval=_sign_pvalue(co,po)
    delta=cw-pw
    passed=delta>=a.min_win_delta and pval<=a.max_pvalue
    payload={
        "schema_version":"sts1-rescue-memory-gate-v30",
        "label":a.label,
        "seeds":len(seeds),
        "mcts_sims":a.mcts_sims,
        "boss_mcts_sims":a.boss_mcts_sims,
        "boss_floors":list(boss_floors),
        "base_sha256":_sha256(a.base_weight),
        "memory_sha256":_sha256(a.memory),
        "parent_wins":pw,
        "candidate_wins":cw,
        "win_delta":delta,
        "candidate_only_wins":co,
        "parent_only_wins":po,
        "candidate_memory_hits":hits,
        "paired_sign_pvalue_one_sided":pval,
        "pass":passed,
        "rows":rows,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print("V30_RESCUE_MEMORY_GATE_RESULT",json.dumps({k:payload[k] for k in (
        "label","seeds","parent_wins","candidate_wins","win_delta",
        "candidate_only_wins","parent_only_wins","candidate_memory_hits",
        "paired_sign_pvalue_one_sided","pass"
    )},sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
