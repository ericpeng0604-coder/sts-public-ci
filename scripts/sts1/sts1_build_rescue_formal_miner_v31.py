#!/usr/bin/env python3
"""Mine Build decisions that create a full win under the formal MCTS-2000 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.simulator import _load_sts, run_simulator_game
from sts1_build_rescue_new_win_miner_v24 import ReplayProductionArmG, _force_spec, _rank_alternatives

SCHEMA="sts1-armg-strategy-branch-dataset-v1"
SAFETY=("illegal_action_count","crash_count","timeout_count","remote_error_count")

def _read(path:Path)->list[int]:
    xs=[int(x.strip()) for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if not xs or len(xs)!=len(set(xs)): raise RuntimeError("seed file invalid")
    return xs

def _safe(r:dict[str,Any])->bool:
    return r.get("result")=="PASS_SIMULATOR_COMPLETE_RUN" and all(int(r.get(k,0) or 0)==0 for k in SAFETY)

def _win(r:dict[str,Any])->bool:
    return str(r.get("outcome","")).lower()=="victory"

def _run(*,seed:int,module_dir:Path,root:Path,weight:Path,training:list[int],forced=None):
    sts=_load_sts(module_dir)
    policy=ReplayProductionArmG(root=root,weight_path=weight,forced=forced)
    r=run_simulator_game(
        student=None,sts=sts,seed=seed,evidence_path=None,armg_policy=policy,
        combat_mcts_sims=2000,training_seeds=training
    )
    if not _safe(r): raise RuntimeError(f"unsafe formal mining seed {seed}: {r}")
    return dict(r),list(policy.records)

def _teacher(seed:int,row:dict[str,Any],alt:int,step:int,final:dict[str,Any])->dict[str,Any]:
    probs=[0.0]*len(row["descs"]); probs[alt]=1.0
    return {
        "schema_version":SCHEMA,
        "type":"v31_formal_mcts2000_full_victory",
        "source":"sts1-build-rescue-formal-miner-v31",
        "seed":seed,"floor":int(row["floor"]),"act":int(row["act"]),"kind":str(row["kind"]),
        "obs":row["obs"],"descs":row["descs"],
        "current_armg_index":int(row["selected_index"]),"teacher_best_index":int(alt),
        "target_probs":probs,"priority":6.0,"confidence_weight":1.0,
        "teacher_consensus_fraction":1.0,"combat_policy":"mcts_2000",
        "confirmation_policy":"formal_mcts_2000_full_victory",
        "intervention_step":step,
        "original_choice":row["semantics"][int(row["selected_index"])],
        "teacher_choice":row["semantics"][int(alt)],
        "formal_result":final,
    }

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--weight",type=Path,required=True)
    p.add_argument("--training-seeds-file",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--teacher-output",type=Path,required=True)
    p.add_argument("--max-states",type=int,default=12)
    p.add_argument("--max-alternatives",type=int,default=1)
    p.add_argument("--max-two-step-states",type=int,default=6)
    p.add_argument("--max-second-states",type=int,default=3)
    p.add_argument("--max-second-alternatives",type=int,default=1)
    a=p.parse_args()
    training=_read(a.training_seeds_file)
    if a.seed not in training: raise RuntimeError("seed not in frozen training pool")

    base,records=_run(seed=a.seed,module_dir=a.module_dir,root=a.armg_root,weight=a.weight,training=training)
    if _win(base):
        payload={"seed":a.seed,"status":"BASE_ALREADY_WIN","base":base,"teachers":[]}
    else:
        candidates=list(reversed([r for r in records if len(r.get("descs",[]))>=2][-a.max_states:]))
        attempts1=0
        rescue=None
        first_cache=[]
        for row in candidates:
            for alt in _rank_alternatives(row,a.max_alternatives):
                attempts1+=1
                forced={int(row["branch_index"]):_force_spec(row,alt)}
                result,rec=_run(seed=a.seed,module_dir=a.module_dir,root=a.armg_root,weight=a.weight,training=training,forced=forced)
                first_cache.append((row,alt,result,rec))
                if _win(result):
                    rescue={
                        "status":"FORMAL_FULL_WIN_ONE_STEP",
                        "teachers":[_teacher(a.seed,row,alt,1,result)],
                        "first":{"branch_index":int(row["branch_index"]),"floor":int(row["floor"]),"kind":str(row["kind"]),"alternative_index":int(alt)},
                        "formal_result":result,
                        "attempted_one_step":attempts1,"attempted_two_step":0,
                    }
                    break
            if rescue: break

        attempts2=0
        if rescue is None:
            for first,alt1,r1,rec1 in first_cache[:a.max_two_step_states*a.max_alternatives]:
                first_idx=int(first["branch_index"])
                later=[r for r in rec1 if int(r["branch_index"])>first_idx and len(r.get("descs",[]))>=2]
                later=list(reversed(later[-a.max_second_states:]))
                for second in later:
                    for alt2 in _rank_alternatives(second,a.max_second_alternatives):
                        attempts2+=1
                        forced={
                            first_idx:_force_spec(first,alt1),
                            int(second["branch_index"]):_force_spec(second,alt2),
                        }
                        result,_=_run(seed=a.seed,module_dir=a.module_dir,root=a.armg_root,weight=a.weight,training=training,forced=forced)
                        if _win(result):
                            rescue={
                                "status":"FORMAL_FULL_WIN_TWO_STEP",
                                "teachers":[
                                    _teacher(a.seed,first,alt1,1,result),
                                    _teacher(a.seed,second,alt2,2,result),
                                ],
                                "first":{"branch_index":first_idx,"floor":int(first["floor"]),"kind":str(first["kind"]),"alternative_index":int(alt1)},
                                "second":{"branch_index":int(second["branch_index"]),"floor":int(second["floor"]),"kind":str(second["kind"]),"alternative_index":int(alt2)},
                                "formal_result":result,
                                "attempted_one_step":attempts1,"attempted_two_step":attempts2,
                            }
                            break
                    if rescue: break
                if rescue: break

        if rescue is None:
            rescue={"status":"NO_FORMAL_NEW_WIN","teachers":[],"attempted_one_step":attempts1,"attempted_two_step":attempts2}
        payload={"seed":a.seed,"base":base,**rescue}

    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    teachers=list(payload.get("teachers") or [])
    a.teacher_output.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in teachers))
    print("V31_FORMAL_MINER_RESULT",json.dumps({
        "seed":a.seed,"status":payload["status"],"base_floor":int((payload.get("base") or {}).get("final_floor") or 0),
        "teacher_examples":len(teachers),"attempted_one_step":payload.get("attempted_one_step",0),
        "attempted_two_step":payload.get("attempted_two_step",0)
    },sort_keys=True),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
