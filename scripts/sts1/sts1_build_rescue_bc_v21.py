#!/usr/bin/env python3
"""Build-only behavior cloning for verified STS1 rescue decisions.

This updates only the ArmG actor. Combat remains MCTS. The candidate learns
verified rescue choices while rehearsing proven-win elite decisions and staying
close to the frozen G7 parent.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "sts1-armg-strategy-branch-dataset-v1"


def _load_rescues(path: Path) -> list[dict[str, Any]]:
    rows=[]
    seen=set()
    for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        row=json.loads(line)
        if row.get("schema_version")!=SCHEMA:
            raise RuntimeError(f"rescue schema mismatch line {line_no}")
        obs=[float(v) for v in row["obs"]]
        descs=[[float(v) for v in d] for d in row["descs"]]
        probs=[float(v) for v in row["target_probs"]]
        cur=int(row["current_armg_index"])
        teacher=int(row["teacher_best_index"])
        if not descs or len(probs)!=len(descs):
            raise RuntimeError(f"rescue candidate shape mismatch line {line_no}")
        if not 0<=cur<len(descs) or not 0<=teacher<len(descs):
            raise RuntimeError(f"rescue index mismatch line {line_no}")
        if cur==teacher:
            raise RuntimeError(f"rescue must be a disagreement line {line_no}")
        if abs(sum(probs)-1.0)>1e-6 or any(v<0 for v in probs):
            raise RuntimeError(f"rescue target invalid line {line_no}")
        if float(row.get("confidence_weight",0.0))<1.0:
            raise RuntimeError(f"rescue confidence below verified contract line {line_no}")
        if float(row.get("teacher_consensus_fraction",0.0))<1.0:
            raise RuntimeError(f"rescue consensus below verified contract line {line_no}")
        key=(int(row["seed"]),int(row["floor"]),str(row["kind"]),cur,teacher)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            **row,
            "obs":obs,
            "descs":descs,
            "target_probs":probs,
        })
    if not rows:
        raise RuntimeError("verified rescue dataset is empty")
    return rows


def _load_elite(path: Path) -> dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as d:
        out={k:np.asarray(d[k]) for k in ("obs","desc","offsets","action","game_victory")}
    if len(out["offsets"])!=len(out["action"])+1:
        raise RuntimeError("elite offsets invalid")
    if not np.all(out["game_victory"]):
        raise RuntimeError("elite replay contains non-victory decisions")
    return out


def _teacher_loss(actor,torch,row:dict[str,Any]):
    obs=torch.tensor(row["obs"],dtype=torch.float32)
    desc=torch.tensor(row["descs"],dtype=torch.float32)
    logits=actor.net(torch.cat([obs.repeat(len(desc),1),desc],1)).squeeze(1)
    target=torch.tensor(row["target_probs"],dtype=torch.float32)
    return -(target*torch.log_softmax(logits,0)).sum(),logits


def _elite_loss(actor,parent,torch,elite:dict[str,np.ndarray],i:int):
    lo=int(elite["offsets"][i]); hi=int(elite["offsets"][i+1])
    obs=torch.tensor(elite["obs"][i],dtype=torch.float32)
    desc=torch.tensor(elite["desc"][lo:hi],dtype=torch.float32)
    x=torch.cat([obs.repeat(len(desc),1),desc],1)
    logits=actor.net(x).squeeze(1)
    with torch.no_grad():
        parent_logits=parent.net(x).squeeze(1)
        p=torch.softmax(parent_logits,0)
    logp=torch.log_softmax(logits,0)
    action=int(elite["action"][i])
    ce=-logp[action]
    kl=(p*(torch.log_softmax(parent_logits,0)-logp)).sum()
    return ce,kl,logits


def _eval_rescue(actor,torch,rows):
    losses=[]; top1=0
    with torch.no_grad():
        for row in rows:
            loss,logits=_teacher_loss(actor,torch,row)
            losses.append(float(loss))
            top1 += int(int(torch.argmax(logits))==int(row["teacher_best_index"]))
    return {"loss":float(np.mean(losses)),"top1":top1/len(rows)}


def _eval_elite(actor,parent,torch,elite,indices):
    correct=0; kls=[]
    with torch.no_grad():
        for raw in indices:
            i=int(raw)
            _,kl,logits=_elite_loss(actor,parent,torch,elite,i)
            correct += int(int(torch.argmax(logits))==int(elite["action"][i]))
            kls.append(float(kl))
    return {"top1":correct/len(indices),"parent_kl":float(np.mean(kls))}


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--base-weight",type=Path,required=True)
    p.add_argument("--elite-replay",type=Path,required=True)
    p.add_argument("--rescue-replay",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--elite-max-decisions",type=int,default=1024)
    p.add_argument("--rescue-repeats",type=int,default=16)
    p.add_argument("--elite-coef",type=float,default=0.40)
    p.add_argument("--rescue-coef",type=float,default=1.0)
    p.add_argument("--anchor-coef",type=float,default=0.03)
    p.add_argument("--max-parent-kl",type=float,default=0.08)
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()

    if not 1<=a.epochs<=20: raise RuntimeError("epochs outside 1..20")
    if not 0<a.lr<=5e-5: raise RuntimeError("lr outside safe bound")
    if not 1<=a.elite_max_decisions<=4096: raise RuntimeError("elite-max-decisions outside safe bound")
    if not 1<=a.rescue_repeats<=32: raise RuntimeError("rescue-repeats outside safe bound")
    if not 0<a.elite_coef<=1 or not 0<a.rescue_coef<=2: raise RuntimeError("BC coef outside safe bound")
    if not 0<=a.anchor_coef<=0.20: raise RuntimeError("anchor coef outside safe bound")

    os.environ["STS_BOT_DIR"]=str(a.armg_root)
    sys.path.insert(0,str(a.armg_root))
    sim_dir=a.armg_root/"sim"/"sts_lightspeed"/"build312"
    if sim_dir.exists():
        sys.path.insert(0,str(sim_dir))
    torch=importlib.import_module("torch")
    torch.set_num_threads(a.threads)
    torch.manual_seed(20260930)
    m=importlib.import_module("armG_train")
    m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)

    actor=m.Scorer((128,128))
    state=torch.load(a.base_weight,weights_only=True,map_location="cpu")
    actor.load_state_dict(state)
    parent=m.Scorer((128,128)); parent.load_state_dict(state); parent.eval()
    for param in parent.parameters(): param.requires_grad_(False)

    rescues=_load_rescues(a.rescue_replay)
    elite=_load_elite(a.elite_replay)
    if len(rescues)<5:
        raise RuntimeError(f"need at least 5 verified rescue examples, got {len(rescues)}")

    expected_obs=int(m.OBS_DIM)
    first_linear=next(layer for layer in actor.net if hasattr(layer,"in_features"))
    expected_desc=int(first_linear.in_features)-expected_obs
    for row in rescues:
        if len(row["obs"])!=expected_obs or any(len(d)!=expected_desc for d in row["descs"]):
            raise RuntimeError("rescue feature dimension mismatch")

    rng=np.random.default_rng(20260930)
    elite_count=min(a.elite_max_decisions,len(elite["action"]))
    elite_train=np.asarray(rng.choice(len(elite["action"]),size=elite_count,replace=False),dtype=np.int64)
    elite_eval=np.asarray(rng.choice(len(elite["action"]),size=min(256,len(elite["action"])),replace=False),dtype=np.int64)

    before_rescue=_eval_rescue(actor,torch,rescues)
    before_elite=_eval_elite(actor,parent,torch,elite,elite_eval)

    opt=torch.optim.Adam(actor.parameters(),lr=a.lr)
    history=[]
    best_state={k:v.detach().clone() for k,v in actor.state_dict().items()}
    best_score=(-1e9,1e9)

    for ep in range(a.epochs):
        actor.train()
        rescue_order=[i for i in range(len(rescues)) for _ in range(a.rescue_repeats)]
        rng.shuffle(rescue_order)
        elite_order=elite_train.copy(); rng.shuffle(elite_order)
        steps=max(len(rescue_order),len(elite_order))
        losses=[]
        for start in range(0,steps,32):
            terms=[]
            for j in range(start,min(start+32,steps)):
                if j<len(rescue_order):
                    row=rescues[int(rescue_order[j])]
                    ce,_=_teacher_loss(actor,torch,row)
                    priority=min(5.0,max(1.0,float(row.get("priority",1.0))))
                    terms.append(a.rescue_coef*priority*ce)
                if j<len(elite_order):
                    ce,kl,_=_elite_loss(actor,parent,torch,elite,int(elite_order[j]))
                    terms.append(a.elite_coef*ce+a.anchor_coef*kl)
            if not terms: continue
            loss=torch.stack(terms).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()),1.0)
            opt.step(); losses.append(float(loss.detach()))
        actor.eval()
        rescue_eval=_eval_rescue(actor,torch,rescues)
        elite_eval_stats=_eval_elite(actor,parent,torch,elite,elite_eval)
        score=(rescue_eval["top1"],rescue_eval["loss"])
        if elite_eval_stats["parent_kl"]<=a.max_parent_kl and (score[0]>best_score[0] or (score[0]==best_score[0] and score[1]<best_score[1])):
            best_score=score
            best_state={k:v.detach().clone() for k,v in actor.state_dict().items()}
        history.append({
            "epoch":ep+1,
            "train_loss":float(np.mean(losses)) if losses else None,
            "rescue_loss":rescue_eval["loss"],
            "rescue_top1":rescue_eval["top1"],
            "elite_top1":elite_eval_stats["top1"],
            "parent_kl":elite_eval_stats["parent_kl"],
        })
        print("V21_BUILD_BC_EPOCH",json.dumps(history[-1],sort_keys=True),flush=True)

    actor.load_state_dict(best_state); actor.eval()
    after_rescue=_eval_rescue(actor,torch,rescues)
    after_elite=_eval_elite(actor,parent,torch,elite,elite_eval)
    if after_rescue["loss"]>=before_rescue["loss"]:
        raise RuntimeError("verified rescue loss did not improve")
    if after_elite["parent_kl"]>a.max_parent_kl:
        raise RuntimeError("candidate drift exceeds parent KL guard")
    if after_elite["top1"]+0.10<before_elite["top1"]:
        raise RuntimeError("candidate lost too much proven-win imitation accuracy")

    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(actor.state_dict(),a.output)
    report={
        "schema_version":"sts1-build-rescue-bc-v21",
        "verified_rescue_examples":len(rescues),
        "rescue_by_kind":{
            k:sum(str(r.get("kind"))==k for r in rescues)
            for k in sorted({str(r.get("kind")) for r in rescues})
        },
        "elite_decisions_total":len(elite["action"]),
        "elite_decisions_sampled":elite_count,
        "rescue_repeats":a.rescue_repeats,
        "before":{"rescue":before_rescue,"elite":before_elite},
        "after":{"rescue":after_rescue,"elite":after_elite},
        "history":history,
        "base_weight":str(a.base_weight),
        "candidate_output":str(a.output),
    }
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("V21_BUILD_BC_PASS",json.dumps({
        "verified_rescue_examples":len(rescues),
        "rescue_loss_before":before_rescue["loss"],
        "rescue_loss_after":after_rescue["loss"],
        "rescue_top1_after":after_rescue["top1"],
        "elite_top1_after":after_elite["top1"],
        "parent_kl":after_elite["parent_kl"],
    },sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
