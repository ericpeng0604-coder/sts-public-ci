#!/usr/bin/env python3
"""Generic paired fixed-seed gate for Build Rescue v2.1."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, _load_sts, run_simulator_game


SAFETY=("illegal_action_count","crash_count","timeout_count","remote_error_count")


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _read_seeds(path:Path)->list[int]:
    seeds=[int(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if not seeds or len(seeds)!=len(set(seeds)):
        raise RuntimeError("seed file must be non-empty and unique")
    return seeds


def _is_win(r:dict[str,Any])->bool:
    return str(r.get("outcome","")).lower()=="victory"


def _safe(r:dict[str,Any])->bool:
    return r.get("result")=="PASS_SIMULATOR_COMPLETE_RUN" and all(int(r.get(k,0) or 0)==0 for k in SAFETY)


def _pair_task(task:tuple[int,str,str,str,str,int,tuple[int,...]])->dict[str,Any]:
    seed,module_dir,armg_root,parent_weight,candidate_weight,mcts_sims,heldout=task
    sts=_load_sts(Path(module_dir))
    parent=ArmGNoncombatPolicy(root=Path(armg_root),weight_path=Path(parent_weight))
    candidate=ArmGNoncombatPolicy(root=Path(armg_root),weight_path=Path(candidate_weight))
    kwargs=dict(student=None,sts=sts,seed=int(seed),evidence_path=None,combat_mcts_sims=int(mcts_sims),heldout_seeds=list(heldout))
    pr=run_simulator_game(armg_policy=parent,**kwargs)
    cr=run_simulator_game(armg_policy=candidate,**kwargs)
    if not _safe(pr) or not _safe(cr):
        raise RuntimeError(f"unsafe/incomplete paired seed {seed}: parent={pr} candidate={cr}")
    return {"seed":int(seed),"parent":pr,"candidate":cr}


def _sign_pvalue(candidate_only:int,parent_only:int)->float:
    n=int(candidate_only)+int(parent_only)
    if n<=0:
        return 1.0
    return sum(math.comb(n,k) for k in range(int(candidate_only),n+1))/(2.0**n)


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--parent-weight",type=Path,required=True)
    p.add_argument("--candidate-weight",type=Path,required=True)
    p.add_argument("--seed-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mcts-sims",type=int,default=2000)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--min-win-delta",type=int,default=1)
    p.add_argument("--max-pvalue",type=float,default=1.0)
    p.add_argument("--label",default="gate")
    a=p.parse_args()
    if a.workers<1 or a.workers>8: raise RuntimeError("workers outside 1..8")
    if a.mcts_sims<1: raise RuntimeError("mcts-sims must be positive")
    if not 0<a.max_pvalue<=1: raise RuntimeError("max-pvalue outside (0,1]")

    seeds=_read_seeds(a.seed_file)
    tasks=[(s,str(a.module_dir),str(a.armg_root),str(a.parent_weight),str(a.candidate_weight),a.mcts_sims,tuple(seeds)) for s in seeds]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(tasks))) as pool:
        rows=list(pool.map(_pair_task,tasks))
    by={int(r["seed"]):r for r in rows}
    rows=[by[s] for s in seeds]

    pw=sum(_is_win(r["parent"]) for r in rows)
    cw=sum(_is_win(r["candidate"]) for r in rows)
    c_only=sum((not _is_win(r["parent"])) and _is_win(r["candidate"]) for r in rows)
    p_only=sum(_is_win(r["parent"]) and (not _is_win(r["candidate"])) for r in rows)
    pval=_sign_pvalue(c_only,p_only)
    delta=cw-pw
    passed=delta>=a.min_win_delta and pval<=a.max_pvalue
    payload={
        "schema_version":"sts1-build-rescue-gate-v21",
        "label":a.label,
        "seeds":len(seeds),
        "mcts_sims":a.mcts_sims,
        "parent_sha256":_sha256(a.parent_weight),
        "candidate_sha256":_sha256(a.candidate_weight),
        "parent_wins":pw,
        "candidate_wins":cw,
        "win_delta":delta,
        "candidate_only_wins":c_only,
        "parent_only_wins":p_only,
        "paired_sign_pvalue_one_sided":pval,
        "min_win_delta":a.min_win_delta,
        "max_pvalue":a.max_pvalue,
        "pass":passed,
        "rows":rows,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("V21_BUILD_GATE_RESULT",json.dumps({k:payload[k] for k in (
        "label","seeds","parent_wins","candidate_wins","win_delta",
        "candidate_only_wins","parent_only_wins","paired_sign_pvalue_one_sided","pass"
    )},sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
