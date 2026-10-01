#!/usr/bin/env python3
"""Calibrate a local similarity Rescue Adapter against frozen G7 winner replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA="sts1-local-rescue-adapter-v31"
FACTORS=(0.25,0.50,0.75,0.90,1.00,1.10,1.25,1.50,2.00)


def _load_jsonl(path:Path)->list[dict[str,Any]]:
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty jsonl: {path}")
    return rows


def _load_elite(path:Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as d:
        required=("obs","desc","offsets","action","game_victory")
        missing=[k for k in required if k not in d]
        if missing:
            raise RuntimeError(f"elite replay missing keys: {missing}")
        out={k:np.asarray(d[k]) for k in required}
    if len(out["offsets"])!=len(out["action"])+1:
        raise RuntimeError("elite offsets invalid")
    if not np.all(out["game_victory"]):
        raise RuntimeError("elite replay contains non-victory decisions")
    return out


def _scale(x:np.ndarray,floor:float)->np.ndarray:
    s=np.std(np.asarray(x,dtype=np.float64),axis=0)
    return np.maximum(s,float(floor))


def _distance(
    obs:np.ndarray,desc:np.ndarray,p_obs:np.ndarray,p_desc:np.ndarray,
    obs_scale:np.ndarray,desc_scale:np.ndarray,
)->float:
    a=np.sum(((obs-p_obs)/obs_scale)**2)
    b=np.sum(((desc-p_desc)/desc_scale)**2)
    return float(np.sqrt((a+b)/(len(obs_scale)+len(desc_scale))))


def _decision_proto_best(
    obs:np.ndarray,
    descs:np.ndarray,
    prototypes:list[dict[str,Any]],
    obs_scale:np.ndarray,
    desc_scale:np.ndarray,
)->tuple[np.ndarray,np.ndarray]:
    p=len(prototypes)
    best_dist=np.full(p,np.inf,dtype=np.float64)
    best_idx=np.full(p,-1,dtype=np.int64)
    for k,proto in enumerate(prototypes):
        po=np.asarray(proto["obs"],dtype=np.float64)
        pd=np.asarray(proto["teacher_desc"],dtype=np.float64)
        obs_term=float(np.sum(((obs-po)/obs_scale)**2))
        if len(descs)==0:
            continue
        dterm=np.sum(((descs-pd)/desc_scale)**2,axis=1)
        total=np.sqrt((obs_term+dterm)/(len(obs_scale)+len(desc_scale)))
        j=int(np.argmin(total))
        best_dist[k]=float(total[j])
        best_idx[k]=j
    return best_dist,best_idx


def _selected_override(
    dists:np.ndarray,
    idxs:np.ndarray,
    radii:np.ndarray,
    *,
    allowed_proto:np.ndarray|None=None,
)->tuple[int|None,float|None,int|None]:
    best=None
    for k,(d,j,r) in enumerate(zip(dists,idxs,radii)):
        if allowed_proto is not None and not bool(allowed_proto[k]):
            continue
        if int(j)<0 or float(d)>float(r)+1e-12:
            continue
        key=(float(d),int(k),int(j))
        if best is None or key<best[0]:
            best=(key,int(j),float(d),int(k))
    if best is None:
        return None,None,None
    return best[1],best[2],best[3]


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--elite-replay",type=Path,required=True)
    p.add_argument("--teacher-memory",type=Path,required=True)
    p.add_argument("--preservation-replay",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--scale-floor",type=float,default=0.05)
    p.add_argument("--max-factor",type=float,default=2.0)
    a=p.parse_args()
    if not 1e-4<=a.scale_floor<=1.0:
        raise RuntimeError("scale-floor outside safe bound")
    if not 0.25<=a.max_factor<=2.0:
        raise RuntimeError("max-factor outside safe bound")

    teachers=_load_jsonl(a.teacher_memory)
    if len(teachers)!=7:
        raise RuntimeError(f"v3.1 expects 7 verified teachers; got {len(teachers)}")
    preservation=_load_jsonl(a.preservation_replay)
    elite=_load_elite(a.elite_replay)

    prototypes=[]
    for row in teachers:
        obs=np.asarray(row["obs"],dtype=np.float64)
        descs=np.asarray(row["descs"],dtype=np.float64)
        teacher=int(row["teacher_best_index"])
        if obs.shape!=(412,) or descs.ndim!=2 or descs.shape[1]!=368:
            raise RuntimeError("teacher feature dimension mismatch")
        if not 0<=teacher<len(descs):
            raise RuntimeError("teacher index invalid")
        prototypes.append({
            "seed":int(row["seed"]),
            "floor":int(row.get("floor",0) or 0),
            "kind":str(row["kind"]),
            "obs":obs.tolist(),
            "teacher_desc":descs[teacher].tolist(),
            "teacher_best_index":teacher,
            "source_type":str(row.get("type","")),
        })

    obs_scale=_scale(elite["obs"],a.scale_floor)
    desc_scale=_scale(elite["desc"],a.scale_floor)

    n=len(elite["action"])
    pcount=len(prototypes)
    nearest=np.full(pcount,np.inf,dtype=np.float64)
    decision_dists=np.full((n,pcount),np.inf,dtype=np.float32)
    decision_idxs=np.full((n,pcount),-1,dtype=np.int16)

    for i in range(n):
        lo=int(elite["offsets"][i]); hi=int(elite["offsets"][i+1])
        if hi<=lo:
            continue
        d,j=_decision_proto_best(
            np.asarray(elite["obs"][i],dtype=np.float64),
            np.asarray(elite["desc"][lo:hi],dtype=np.float64),
            prototypes,obs_scale,desc_scale,
        )
        decision_dists[i]=d.astype(np.float32)
        decision_idxs[i]=j.astype(np.int16)
        nearest=np.minimum(nearest,d)

    # Preservation candidates are also hard protected points.
    preservation_cache=[]
    for row in preservation:
        obs=np.asarray(row["obs"],dtype=np.float64)
        descs=np.asarray(row["descs"],dtype=np.float64)
        d,j=_decision_proto_best(obs,descs,prototypes,obs_scale,desc_scale)
        allowed=np.asarray(
            [proto["kind"]==str(row["kind"]) for proto in prototypes],
            dtype=bool,
        )
        for k in range(pcount):
            if allowed[k]:
                nearest[k]=min(float(nearest[k]),float(d[k]))
        preservation_cache.append((row,d,j,allowed))

    if not np.all(np.isfinite(nearest)):
        raise RuntimeError("unable to calibrate nearest protected distance")
    nearest=np.maximum(nearest,1e-9)

    factor_trials=[]
    best_factor=None
    for factor in FACTORS:
        if factor>a.max_factor+1e-12:
            continue
        radii=nearest*float(factor)
        changes=0
        hits=0
        for i in range(n):
            chosen,_,_=_selected_override(
                decision_dists[i],decision_idxs[i],radii
            )
            if chosen is None:
                continue
            hits+=1
            if int(chosen)!=int(elite["action"][i]):
                changes+=1

        preservation_changes=0
        preservation_hits=0
        for row,d,j,allowed in preservation_cache:
            chosen,_,_=_selected_override(d,j,radii,allowed_proto=allowed)
            if chosen is None:
                continue
            preservation_hits+=1
            if int(chosen)!=int(row["parent_selected_index"]):
                preservation_changes+=1

        safe=(changes==0 and preservation_changes==0)
        trial={
            "factor":float(factor),
            "winner_replay_hits":int(hits),
            "winner_replay_changes":int(changes),
            "preservation_hits":int(preservation_hits),
            "preservation_changes":int(preservation_changes),
            "safe":bool(safe),
        }
        factor_trials.append(trial)
        print("V31_ADAPTER_FACTOR",json.dumps(trial,sort_keys=True),flush=True)
        if safe:
            best_factor=float(factor)

    if best_factor is None:
        raise RuntimeError("no local-adapter radius factor preserves protected replay")
    radii=nearest*best_factor

    # Every training teacher must self-replay to its teacher action.
    self_replay=[]
    for row in teachers:
        obs=np.asarray(row["obs"],dtype=np.float64)
        descs=np.asarray(row["descs"],dtype=np.float64)
        d,j=_decision_proto_best(obs,descs,prototypes,obs_scale,desc_scale)
        allowed=np.asarray(
            [proto["kind"]==str(row["kind"]) for proto in prototypes],
            dtype=bool,
        )
        chosen,distance,k=_selected_override(d,j,radii,allowed_proto=allowed)
        passed=(chosen==int(row["teacher_best_index"]))
        self_replay.append({
            "seed":int(row["seed"]),
            "floor":int(row.get("floor",0) or 0),
            "kind":str(row["kind"]),
            "expected":int(row["teacher_best_index"]),
            "chosen":None if chosen is None else int(chosen),
            "distance":distance,
            "prototype_index":k,
            "pass":bool(passed),
        })
    if not all(x["pass"] for x in self_replay):
        raise RuntimeError(f"local adapter failed teacher self-replay: {self_replay}")

    out_prototypes=[]
    for proto,near,radius in zip(prototypes,nearest,radii):
        out_prototypes.append({
            **proto,
            "nearest_protected_distance":float(near),
            "radius":float(radius),
        })

    payload={
        "schema_version":SCHEMA,
        "mode":"calibrated-local-nearest-prototype",
        "calibration_factor":best_factor,
        "obs_scale":[float(x) for x in obs_scale],
        "desc_scale":[float(x) for x in desc_scale],
        "prototypes":out_prototypes,
        "protected_winner_decisions":int(n),
        "calibration_guarantee":"0 winner-replay changes and 0 regression-anchor changes on calibration set",
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,separators=(",",":"),sort_keys=True)+"\n",encoding="utf-8")
    report={
        "schema_version":"sts1-local-rescue-adapter-v31-report",
        "teacher_examples":len(teachers),
        "teacher_seeds":len({int(x["seed"]) for x in teachers}),
        "selected_factor":best_factor,
        "nearest_protected_distance":[float(x) for x in nearest],
        "radii":[float(x) for x in radii],
        "factor_trials":factor_trials,
        "teacher_self_replay":self_replay,
        "winner_replay_decisions":int(n),
        "preservation_examples":len(preservation),
    }
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("V31_LOCAL_ADAPTER_BUILD_PASS",json.dumps({
        "teacher_examples":len(teachers),
        "teacher_seeds":len({int(x["seed"]) for x in teachers}),
        "selected_factor":best_factor,
        "winner_replay_decisions":int(n),
        "self_replay_pass":True,
    },sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
