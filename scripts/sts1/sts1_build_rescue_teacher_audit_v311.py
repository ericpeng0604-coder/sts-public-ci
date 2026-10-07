#!/usr/bin/env python3
"""Bounded per-Teacher audit of the v3.10 residual and v3.11 target guards."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from roguelike_ai.sts1_phase3.residual_scoring import (
    adapter_kind_enabled,
    apply_residual_scores,
    choice_relative_margin_shortfall,
    top1_index,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    rows=[]
    for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        row=json.loads(line)
        if not isinstance(row,dict):
            raise RuntimeError(f"{path}:{line_no} is not an object")
        rows.append(row)
    return rows


def _state_key(obs: Any, descs: Any) -> str:
    h=hashlib.sha256()
    o=np.asarray(obs,dtype=np.float32)
    d=np.asarray(descs,dtype=np.float32)
    h.update(np.asarray(o.shape,dtype=np.int64).tobytes()); h.update(o.tobytes())
    h.update(np.asarray(d.shape,dtype=np.int64).tobytes()); h.update(d.tobytes())
    return h.hexdigest()


def _gate_metrics(torch, gate: dict[str,Any], kind: str, obs):
    pos=(gate.get("positive_centers") or {}).get(kind)
    neg=(gate.get("negative_centers") or {}).get(kind)
    ratio=float(gate["positive_to_negative_ratio"])
    max_dist=(gate.get("max_positive_distance") or {}).get(kind)
    if pos is None or int(pos.numel())==0:
        return {"allowed":False,"d_positive":None,"d_negative":None,
                "max_positive_distance":max_dist,"positive_to_negative_ratio":ratio}
    z=(obs-gate["obs_mean"])/gate["obs_std"]
    dpos=float(torch.mean((pos-z.unsqueeze(0))**2,dim=1).min())
    dneg=None
    if neg is not None and int(neg.numel())>0:
        dneg=float(torch.mean((neg-z.unsqueeze(0))**2,dim=1).min())
    allowed=max_dist is None or dpos<=float(max_dist)
    if allowed and dneg is not None:
        allowed=dpos<=dneg*ratio
    return {"allowed":bool(allowed),"d_positive":dpos,"d_negative":dneg,
            "max_positive_distance":max_dist,"positive_to_negative_ratio":ratio}


def run_audit(*,armg_root:Path,base_weight:Path,adapter_sidecar:Path,elite_replay:Path,
              teacher_replay:Path,preservation_replay:Path,margin:float,
              candidate_run_id:int,output_dir:Path)->dict[str,Any]:
    sys.path.insert(0,str(armg_root))
    sim_dir=armg_root/"sim"/"sts_lightspeed"/"build312"
    if not sim_dir.exists():
        raise RuntimeError(f"pinned slaythespire binding is missing: {sim_dir}")
    sys.path.insert(0,str(sim_dir))
    import sts1_build_rescue_adapter_v34 as v34
    import sts1_build_rescue_bc_v28 as base
    m=importlib.import_module("armG_train")
    m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)
    torch=importlib.import_module("torch")
    teachers=base._load_rescues(teacher_replay)
    preservation=base._load_preservation(preservation_replay)
    elite=base._load_elite(elite_replay)
    if len(teachers)!=62 or len({int(r["seed"]) for r in teachers})!=56:
        raise RuntimeError("bounded audit expected the verified 62 decisions / 56 seeds")
    parent=m.Scorer((128,128))
    parent.load_state_dict(torch.load(base_weight,weights_only=True,map_location="cpu"))
    parent.eval()
    payload=torch.load(adapter_sidecar,weights_only=True,map_location="cpu")
    if payload.get("schema_version")!="sts1-armg-residual-adapter-v2":
        raise RuntimeError("v3.10 reference adapter schema mismatch")
    input_dim=int(payload["input_dim"]); hidden_dim=int(payload["hidden_dim"])
    adapter=torch.nn.Sequential(torch.nn.Linear(input_dim,hidden_dim),torch.nn.Tanh(),torch.nn.Linear(hidden_dim,1))
    adapter.load_state_dict(payload["state_dict"]); adapter.eval()
    adapter_kinds={str(kind) for kind in payload.get("kinds", [])}
    if not adapter_kinds:
        raise RuntimeError("v3.10 reference adapter has no enabled decision kinds")
    gate=payload.get("gate")
    if not isinstance(gate,dict):
        raise RuntimeError("v3.10 reference adapter is missing gate")
    gate["positive_to_negative_ratio"]=float(gate.get("positive_to_negative_ratio",0.8))

    # Match the exact deterministic retention evaluation slice used by v3.5.
    rng=np.random.default_rng(20261006)
    perm=rng.permutation(len(elite["action"]))
    eval_count=min(2048,max(64,len(perm)//5))
    retention_eval=perm[:eval_count]
    elite_keys={}
    for raw_i in retention_eval:
        i=int(raw_i); lo=int(elite["offsets"][i]); hi=int(elite["offsets"][i+1])
        key=_state_key(elite["obs"][i],elite["desc"][lo:hi])
        obs=torch.tensor(elite["obs"][i],dtype=torch.float32)
        desc=torch.tensor(elite["desc"][lo:hi],dtype=torch.float32)
        x=torch.cat([obs.repeat(len(desc),1),desc],1)
        with torch.no_grad():
            logits=parent.net(x).squeeze(1)
            probs=torch.softmax(logits,0)
        p=top1_index(logits)
        top_probs=torch.topk(probs,k=2).values
        gap=float(top_probs[0]-top_probs[1])
        elite_keys.setdefault(key,[]).append({"parent_top1":p,"parent_prob_gap":gap,"index":i})

    preserve_by_key={}
    for r in preservation:
        key=_state_key(r["obs"],r["descs"])
        preserve_by_key.setdefault(key,[]).append(int(r["parent_selected_index"]))

    decisions=[]
    counts={"gate_checks":0,"gate_allowed":0,"gate_blocked":0,"adapter_applied":0,"parent_top_teacher":0,
            "teacher_target_would_change_if_allowed":0,"teacher_target_blocked_by_gate":0,
            "top1_changed":0,"top1_changed_when_gate_blocked":0,
            "high_confidence_guard_overlaps":0,"preservation_guard_overlaps":0,
            "high_confidence_guard_conflicts":0,"preservation_guard_conflicts":0,
            "nonfinite_or_invalid":0,"all_alternative_pairs":0}
    high_conf=float(base.PARENT_HIGH_CONFIDENCE_PROB_GAP)
    for row_idx,row in enumerate(teachers):
        obs=torch.tensor(row["obs"],dtype=torch.float32)
        desc=torch.tensor(row["descs"],dtype=torch.float32)
        if len(desc)<2 or len(desc)!=len(row["target_probs"]):
            raise RuntimeError(f"teacher row {row_idx} has malformed legal choice set")
        target=int(row["teacher_best_index"])
        if not 0<=target<len(desc):
            raise RuntimeError(f"teacher row {row_idx} target index is outside legal choices")
        x=torch.cat([obs.repeat(len(desc),1),desc],1)
        with torch.no_grad():
            parent_scores=parent.net(x).squeeze(1)
            residual=adapter(x).squeeze(1)
        p=top1_index(parent_scores)
        gate_checked=adapter_kind_enabled(str(row["kind"]),adapter_kinds)
        gate_result=(_gate_metrics(torch,gate,str(row["kind"]),obs) if gate_checked else
            {"allowed":False,"d_positive":None,"d_negative":None,
             "max_positive_distance":(gate.get("max_positive_distance") or {}).get(str(row["kind"])),
             "positive_to_negative_ratio":float(gate["positive_to_negative_ratio"])})
        applied=bool(gate_checked and gate_result["allowed"])
        effective=apply_residual_scores(parent_scores,residual,applied)
        pred=top1_index(effective)
        shortfall,parent_top=choice_relative_margin_shortfall(parent_scores,residual,target,margin)
        alternatives=[]
        for i in range(len(desc)):
            if i==p:
                continue
            sf,_=choice_relative_margin_shortfall(parent_scores,residual,i,margin)
            alternatives.append({"index":i,"parent_gap":float(parent_scores[p]-parent_scores[i]),
                "relative_delta":float(residual[i]-residual[p]),"target_shortfall":float(sf)})
        counts["all_alternative_pairs"]+=len(alternatives)
        key=_state_key(row["obs"],row["descs"])
        preserve_targets=preserve_by_key.get(key,[])
        elite_matches=elite_keys.get(key,[])
        high_matches=[x for x in elite_matches if x["parent_prob_gap"]>=high_conf]
        high_conflict=bool(high_matches and target!=p)
        preserve_conflict=bool(preserve_targets and any(x!=target for x in preserve_targets))
        counts["gate_checks"]+=int(gate_checked)
        counts["gate_allowed"]+=int(gate_checked and gate_result["allowed"])
        counts["gate_blocked"]+=int(gate_checked and not gate_result["allowed"])
        counts["adapter_applied"]+=int(applied)
        counts["parent_top_teacher"]+=int(target==p)
        counts["teacher_target_would_change_if_allowed"]+=int(target!=p and top1_index(apply_residual_scores(parent_scores,residual,True))==target)
        counts["teacher_target_blocked_by_gate"]+=int(target!=p and not applied and top1_index(apply_residual_scores(parent_scores,residual,True))==target)
        counts["top1_changed"]+=int(pred!=p)
        counts["top1_changed_when_gate_blocked"]+=int(not applied and top1_index(apply_residual_scores(parent_scores,residual,True))!=p)
        counts["high_confidence_guard_overlaps"]+=len(high_matches)
        counts["preservation_guard_overlaps"]+=len(preserve_targets)
        counts["high_confidence_guard_conflicts"]+=int(high_conflict)
        counts["preservation_guard_conflicts"]+=int(preserve_conflict)
        if not all(math.isfinite(float(x)) for x in [*parent_scores,*residual,float(shortfall)]):
            counts["nonfinite_or_invalid"]+=1
        decisions.append({
            "source_file":teacher_replay.name,"source_row":row_idx,"candidate_run_id":candidate_run_id,
            "seed":int(row["seed"]),"floor":int(row.get("floor",0)),"act":int(row.get("act",0)),
            "kind":str(row["kind"]),"hp":row.get("hp"),"gold":row.get("gold"),
            "current_armg_index":int(row["current_armg_index"]),"teacher_index":target,
            "teacher_is_parent_top1":target==p,"parent_top1_index":p,
            "legal_choice_count":len(desc),"legal_choice_descriptor_sha256":[hashlib.sha256(np.asarray(d,dtype=np.float32).tobytes()).hexdigest() for d in row["descs"]],
            "parent_scores":[float(x) for x in parent_scores],"residual_scores":[float(x) for x in residual],
            "parent_gap":float(parent_scores[p]-parent_scores[target]),
            "relative_delta":float(residual[target]-residual[p]),"target_shortfall":float(shortfall),
            "other_alternative_pairs":alternatives,"gate_checked":gate_checked,
            "gate":gate_result,"adapter_applied":applied,
            "candidate_top1_index":pred,"top1_changed":pred!=p,"teacher_target_would_win_if_allowed":top1_index(apply_residual_scores(parent_scores,residual,True))==target,
            "preservation_parent_indices":preserve_targets,"preservation_guard_conflict":preserve_conflict,
            "retention_eval_matches":[{"index":x["index"],"parent_top1":x["parent_top1"],"parent_prob_gap":x["parent_prob_gap"]} for x in elite_matches],
            "high_confidence_guard_conflict":high_conflict,
            "teacher_source_metadata":{k:row[k] for k in row if k in {"confidence_weight","teacher_consensus_fraction","source_type","source_run_id","cluster_id","cluster_seed_count","cluster_weight"}},
        })
    summary={"schema_version":"sts1-v311-teacher-target-audit","candidate_run_id":candidate_run_id,
        "teacher_examples":len(teachers),"teacher_seeds":len({int(r["seed"]) for r in teachers}),
        "margin":margin,"retention_eval_rows":len(retention_eval),"high_confidence_prob_gap_threshold":high_conf,
        "counts":counts,"training_conflicts":counts["preservation_guard_conflicts"]+counts["high_confidence_guard_conflicts"],
        "scale_contract":"sidecar weights are evaluated without any extra inference multiplier; logits use original legal-choice indices",
        "source_teacher_file":str(teacher_replay),"source_adapter_file":str(adapter_sidecar),"source_preservation_file":str(preservation_replay),
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (output_dir/"decisions.jsonl").open("w",encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    print("V311_TEACHER_AUDIT",json.dumps(summary,sort_keys=True),flush=True)
    if counts["nonfinite_or_invalid"]:
        raise RuntimeError("Teacher audit found invalid or non-finite scores")
    if summary["training_conflicts"]:
        raise RuntimeError("Teacher targets conflict with preservation/high-confidence retention rows")
    return summary


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--base-weight",type=Path,required=True)
    p.add_argument("--adapter-sidecar",type=Path,required=True)
    p.add_argument("--elite-replay",type=Path,required=True)
    p.add_argument("--teacher-replay",type=Path,required=True)
    p.add_argument("--preservation-replay",type=Path,required=True)
    p.add_argument("--candidate-run-id",type=int,required=True)
    p.add_argument("--margin",type=float,default=0.75)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    if not 0<=a.margin<=5: raise RuntimeError("margin outside safe bound")
    run_audit(armg_root=a.armg_root,base_weight=a.base_weight,adapter_sidecar=a.adapter_sidecar,
        elite_replay=a.elite_replay,teacher_replay=a.teacher_replay,preservation_replay=a.preservation_replay,
        margin=a.margin,candidate_run_id=a.candidate_run_id,output_dir=a.output_dir)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
