#!/usr/bin/env python3
"""v3.5 frozen-G7 residual adapter with negative evidence and confidence gating."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import sts1_build_rescue_adapter_v34 as v34
import sts1_build_rescue_bc_v28 as base


SCHEMA_V1 = "sts1-armg-residual-adapter-v1"
SCHEMA_V2 = "sts1-armg-residual-adapter-v2"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        row=json.loads(line)
        if not isinstance(row,dict):
            raise RuntimeError(f"{path}:{n} is not an object")
        rows.append(row)
    return rows


def _negative_loss(parent, adapter, torch, row, margin: float):
    x=v34._x(torch,row)
    with torch.no_grad():
        parent_logits=parent.net(x).squeeze(1)
    logits=parent_logits+adapter(x).squeeze(1)
    current=int(row["current_armg_index"])
    rejected=int(row["rejected_index"])
    return torch.relu(logits[rejected]-logits[current]+margin)


def _negative_eval(parent, adapter, torch, rows):
    if not rows:
        return {
            "examples":0,"rejected_top1_rate":0.0,"current_top1_rate":1.0,
            "mean_current_minus_rejected":0.0,
        }
    rejected_top1=0
    current_top1=0
    margins=[]
    with torch.no_grad():
        for row in rows:
            x=v34._x(torch,row)
            logits=parent.net(x).squeeze(1)+adapter(x).squeeze(1)
            current=int(row["current_armg_index"])
            rejected=int(row["rejected_index"])
            pred=int(torch.argmax(logits))
            rejected_top1+=int(pred==rejected)
            current_top1+=int(pred==current)
            margins.append(float(logits[current]-logits[rejected]))
    return {
        "examples":len(rows),
        "rejected_top1_rate":rejected_top1/len(rows),
        "current_top1_rate":current_top1/len(rows),
        "mean_current_minus_rejected":float(np.mean(margins)),
        "min_current_minus_rejected":float(np.min(margins)),
    }


def _state_key(row: dict[str,Any]) -> str:
    return json.dumps(
        {"kind":str(row["kind"]),"obs":row["obs"],"descs":row["descs"]},
        sort_keys=True,separators=(",",":"),
    )


def _build_gate(torch, positives, negatives, *, ratio: float):
    # Exact duplicated positive states must never also act as negative centers.
    pos_keys={_state_key(r) for r in positives}
    negatives=[r for r in negatives if _state_key(r) not in pos_keys]
    all_obs=np.asarray(
        [r["obs"] for r in positives]+[r["obs"] for r in negatives],
        dtype=np.float32,
    )
    if len(all_obs)<2:
        raise RuntimeError("not enough states to build confidence gate")
    mean=all_obs.mean(axis=0)
    std=all_obs.std(axis=0)
    std=np.where(std<1e-3,1.0,std).astype(np.float32)

    pos_by=defaultdict(list)
    neg_by=defaultdict(list)
    for r in positives:
        pos_by[str(r["kind"])].append((np.asarray(r["obs"],dtype=np.float32)-mean)/std)
    for r in negatives:
        neg_by[str(r["kind"])].append((np.asarray(r["obs"],dtype=np.float32)-mean)/std)

    # Conservative fallback radius comes only from within-kind positive neighbors.
    global_neighbor=[]
    per_kind_neighbor={}
    for kind,rows in pos_by.items():
        arr=np.asarray(rows,dtype=np.float32)
        vals=[]
        if len(arr)>=2:
            for i in range(len(arr)):
                d=np.mean((arr-arr[i])**2,axis=1)
                d[i]=np.inf
                vals.append(float(np.min(d)))
                global_neighbor.append(vals[-1])
        per_kind_neighbor[kind]=vals
    fallback=float(np.quantile(global_neighbor,0.50)*1.25) if global_neighbor else 0.05
    fallback=max(fallback,1e-4)

    max_distance={}
    positive_centers={}
    negative_centers={}
    for kind,rows in pos_by.items():
        parr=np.asarray(rows,dtype=np.float32)
        narr=np.asarray(neg_by.get(kind,[]),dtype=np.float32)
        positive_centers[kind]=torch.tensor(parr,dtype=torch.float32)
        negative_centers[kind]=(
            torch.tensor(narr,dtype=torch.float32)
            if len(narr)
            else torch.empty((0,len(mean)),dtype=torch.float32)
        )
        vals=per_kind_neighbor.get(kind) or []
        if vals:
            radius=float(np.quantile(vals,0.75)*1.35)+1e-6
        elif len(narr):
            d=float(np.min(np.mean((narr-parr[0])**2,axis=1)))
            radius=max(1e-4,min(fallback,d*0.50))
        else:
            radius=fallback
        max_distance[kind]=float(radius)

    # Contract diagnostics: every training positive is admitted; every retained
    # negative center is rejected because its nearest negative distance is zero.
    def allows(kind: str, obs: list[float]) -> bool:
        z=(np.asarray(obs,dtype=np.float32)-mean)/std
        p=np.asarray(pos_by[kind],dtype=np.float32)
        dpos=float(np.min(np.mean((p-z)**2,axis=1)))
        if dpos>max_distance[kind]:
            return False
        n=np.asarray(neg_by.get(kind,[]),dtype=np.float32)
        if len(n):
            dneg=float(np.min(np.mean((n-z)**2,axis=1)))
            if dpos>dneg*ratio:
                return False
        return True

    pos_pass=sum(allows(str(r["kind"]),r["obs"]) for r in positives)
    neg_block=sum(not allows(str(r["kind"]),r["obs"]) for r in negatives if str(r["kind"]) in pos_by)
    neg_check=sum(1 for r in negatives if str(r["kind"]) in pos_by)
    if pos_pass!=len(positives):
        raise RuntimeError(f"confidence gate rejects positive training states: {pos_pass}/{len(positives)}")
    if neg_check and neg_block!=neg_check:
        raise RuntimeError(f"confidence gate admits negative training states: blocked {neg_block}/{neg_check}")

    gate={
        "obs_mean":torch.tensor(mean,dtype=torch.float32),
        "obs_std":torch.tensor(std,dtype=torch.float32),
        "positive_centers":positive_centers,
        "negative_centers":negative_centers,
        "positive_to_negative_ratio":float(ratio),
        "max_positive_distance":max_distance,
    }
    diag={
        "positive_examples":len(positives),
        "negative_examples":len(negatives),
        "positive_admitted":pos_pass,
        "negative_checked":neg_check,
        "negative_blocked":neg_block,
        "ratio":ratio,
        "max_positive_distance":max_distance,
        "positive_centers_by_kind":{k:len(v) for k,v in pos_by.items()},
        "negative_centers_by_kind":{k:len(v) for k,v in neg_by.items()},
    }
    return gate,diag


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--base-weight",type=Path,required=True)
    p.add_argument("--elite-replay",type=Path,required=True)
    p.add_argument("--new-win-replay",type=Path,required=True)
    p.add_argument("--negative-replay",type=Path,required=True)
    p.add_argument("--preservation-replay",type=Path,required=True)
    p.add_argument("--init-adapter-sidecar",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--hidden-dim",type=int,default=48)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--elite-train-max",type=int,default=8192)
    p.add_argument("--retention-eval-max",type=int,default=2048)
    p.add_argument("--teacher-margin",type=float,default=0.05)
    p.add_argument("--teacher-coef",type=float,default=4.0)
    p.add_argument("--negative-margin",type=float,default=0.02)
    p.add_argument("--negative-coef",type=float,default=3.0)
    p.add_argument("--distill-coef",type=float,default=14.0)
    p.add_argument("--zero-residual-coef",type=float,default=0.85)
    p.add_argument("--preservation-coef",type=float,default=20.0)
    p.add_argument("--preservation-margin",type=float,default=0.03)
    p.add_argument("--min-preservation-margin",type=float,default=0.01)
    p.add_argument("--weight-decay",type=float,default=1e-5)
    p.add_argument("--min-fully-learned-seeds",type=int,default=10)
    p.add_argument("--min-parent-agreement",type=float,default=0.995)
    p.add_argument("--max-parent-kl",type=float,default=0.002)
    p.add_argument("--max-kl-increase",type=float,default=0.001)
    p.add_argument("--max-winner-drop",type=float,default=0.01)
    p.add_argument("--gate-ratio",type=float,default=0.80)
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()

    if not 4<=a.hidden_dim<=128: raise RuntimeError("hidden-dim outside safe bound")
    if not 1<=a.epochs<=50: raise RuntimeError("epochs outside safe bound")
    if not 1e-6<=a.lr<=1e-3: raise RuntimeError("lr outside safe bound")
    if not 0<a.negative_margin<=0.5: raise RuntimeError("negative-margin outside safe bound")
    if not 0<a.gate_ratio<=1: raise RuntimeError("gate-ratio outside safe bound")
    if not 0<=a.max_kl_increase<=0.01: raise RuntimeError("max-kl-increase outside safe bound")

    os.environ["STS_BOT_DIR"]=str(a.armg_root)
    sys.path.insert(0,str(a.armg_root))
    torch=importlib.import_module("torch")
    torch.set_num_threads(a.threads)
    torch.manual_seed(20261006)
    m=importlib.import_module("armG_train")
    m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)

    parent_state=torch.load(a.base_weight,weights_only=True,map_location="cpu")
    parent=m.Scorer((128,128)); parent.load_state_dict(parent_state); parent.eval()
    for param in parent.parameters(): param.requires_grad_(False)

    positives=base._load_rescues(a.new_win_replay)
    negatives=_load_jsonl(a.negative_replay)
    preservation=base._load_preservation(a.preservation_replay)
    elite=base._load_elite(a.elite_replay)
    kinds=sorted({str(r["kind"]) for r in positives})
    if not positives or not negatives:
        raise RuntimeError("v3.5 requires both positive and negative evidence")
    if any(str(r["kind"]) not in kinds for r in negatives):
        # Negative-only kinds cannot activate an adapter without positive rescue evidence.
        negatives=[r for r in negatives if str(r["kind"]) in kinds]

    expected_obs=int(m.OBS_DIM)
    first_linear=next(layer for layer in parent.net if hasattr(layer,"in_features"))
    expected_desc=int(first_linear.in_features)-expected_obs
    input_dim=expected_obs+expected_desc

    adapter=torch.nn.Sequential(
        torch.nn.Linear(input_dim,a.hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(a.hidden_dim,1),
    )
    payload=torch.load(a.init_adapter_sidecar,weights_only=True,map_location="cpu")
    if payload.get("schema_version") not in {SCHEMA_V1,SCHEMA_V2}:
        raise RuntimeError("unsupported warm-start adapter schema")
    if int(payload.get("input_dim",-1))!=input_dim or int(payload.get("hidden_dim",-1))!=a.hidden_dim:
        raise RuntimeError("warm-start adapter dimension mismatch")
    adapter.load_state_dict(payload["state_dict"])

    rng=np.random.default_rng(20261006)
    perm=rng.permutation(len(elite["action"]))
    eval_count=min(a.retention_eval_max,max(64,len(perm)//5))
    train_pool=np.asarray(perm[eval_count:],dtype=np.int64)
    retention_eval=np.asarray(perm[:eval_count],dtype=np.int64)
    elite_train=train_pool[:min(a.elite_train_max,len(train_pool))]
    if len(elite_train)<128: raise RuntimeError("not enough elite replay")

    before_new=v34._teacher_eval(parent,adapter,torch,positives)
    before_neg=_negative_eval(parent,adapter,torch,negatives)
    before_pres=v34._preservation_eval(parent,adapter,torch,preservation,set(kinds))
    before_ret=v34._retention_eval(parent,adapter,torch,elite,retention_eval)
    allowed_parent_kl=max(
        float(a.max_parent_kl),
        float(before_ret["parent_kl"])+float(a.max_kl_increase),
    )
    def retention_ok(stats):
        return (
            int(stats["parent_high_conf_decisions"]) >= base.MIN_HIGH_CONFIDENCE_DECISIONS
            and float(stats["parent_high_conf_top1_agreement"]) >= a.min_parent_agreement
            and float(stats["parent_kl"]) <= allowed_parent_kl
            and float(stats["winner_top1"]) + a.max_winner_drop >= float(before_ret["winner_top1"])
        )

    opt=torch.optim.AdamW(adapter.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    history=[]; best=None; batch=64

    # Epoch 0 is a first-class candidate: keep the already useful v3.4 weights
    # exactly unchanged and add only the v3.5 confidence gate.  This prevents
    # unnecessary gradient updates from destroying a candidate that already
    # passed Gate30/Gate50 on prior unseen data.
    learned0=int(before_new["fully_learned_seeds"])>=a.min_fully_learned_seeds
    pres0=(float(before_pres["top1"])>=1.0 and float(before_pres.get("min_margin",0.0))>=a.min_preservation_margin)
    ret0=retention_ok(before_ret)
    neg0=float(before_neg["rejected_top1_rate"])<=0.01
    epoch0={
        "epoch":0,"train_loss":0.0,"new_win":before_new,"negative":before_neg,
        "preservation":before_pres,"retention":before_ret,
        "learned_enough":learned0,"negative_pass":neg0,
        "preservation_pass":pres0,"retention_pass":ret0,
        "pass":bool(learned0 and pres0 and ret0 and neg0),
    }
    history.append(epoch0)
    print("V35_ADAPTER_EPOCH",json.dumps(epoch0,sort_keys=True),flush=True)
    if epoch0["pass"]:
        score=(
            int(before_new["fully_learned_seeds"]),
            -float(before_neg["rejected_top1_rate"]),
            float(before_neg["current_top1_rate"]),
            -float(before_ret["parent_kl"]),
            float(before_ret["winner_top1"]),
        )
        best=(
            score,
            {k:v.detach().clone() for k,v in adapter.state_dict().items()},
            epoch0,
        )

    for epoch in range(1,a.epochs+1):
        adapter.train()
        order=elite_train.copy(); rng.shuffle(order)
        losses=[]
        for start in range(0,len(order),batch):
            ids=order[start:start+batch]
            distill=[]; zero=[]
            for raw in ids:
                i=int(raw)
                pl,logits,res=v34._elite_logits(parent,adapter,torch,elite,i,True)
                with torch.no_grad():
                    pp=torch.softmax(pl,0); plog=torch.log_softmax(pl,0)
                clog=torch.log_softmax(logits,0)
                distill.append((pp*(plog-clog)).sum())
                zero.append((res**2).mean())
            teacher=[v34._teacher_margin_loss(parent,adapter,torch,r,a.teacher_margin) for r in positives]
            neg=[_negative_loss(parent,adapter,torch,r,a.negative_margin) for r in negatives]
            pres=[]
            for r in preservation:
                t=v34._preservation_loss(parent,adapter,torch,r,set(kinds),a.preservation_margin)
                if t is not None: pres.append(t)
            parts=[
                a.distill_coef*torch.stack(distill).mean(),
                a.zero_residual_coef*torch.stack(zero).mean(),
                a.teacher_coef*torch.stack(teacher).mean(),
                a.negative_coef*torch.stack(neg).mean(),
            ]
            if pres: parts.append(a.preservation_coef*torch.stack(pres).mean())
            loss=torch.stack(parts).sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.0)
            opt.step(); losses.append(float(loss.detach()))

        adapter.eval()
        new=v34._teacher_eval(parent,adapter,torch,positives)
        neg=_negative_eval(parent,adapter,torch,negatives)
        pres=v34._preservation_eval(parent,adapter,torch,preservation,set(kinds))
        ret=v34._retention_eval(parent,adapter,torch,elite,retention_eval)
        retention_pass=retention_ok(ret)
        learned=int(new["fully_learned_seeds"])>=a.min_fully_learned_seeds
        pres_ok=(float(pres["top1"])>=1.0 and float(pres.get("min_margin",0.0))>=a.min_preservation_margin)
        neg_ok=(
            float(neg["rejected_top1_rate"])<=float(before_neg["rejected_top1_rate"])+0.01
            and float(neg["current_top1_rate"])>=float(before_neg["current_top1_rate"])-0.01
        )
        passed=learned and pres_ok and retention_pass and neg_ok
        row={
            "epoch":epoch,"train_loss":float(np.mean(losses)),
            "new_win":new,"negative":neg,"preservation":pres,"retention":ret,
            "learned_enough":learned,"negative_pass":neg_ok,
            "preservation_pass":pres_ok,"retention_pass":retention_pass,"pass":passed,
        }
        history.append(row)
        print("V35_ADAPTER_EPOCH",json.dumps(row,sort_keys=True),flush=True)
        if passed:
            score=(
                int(new["fully_learned_seeds"]),
                -float(neg["rejected_top1_rate"]),
                float(neg["current_top1_rate"]),
                -float(ret["parent_kl"]),
                float(ret["winner_top1"]),
            )
            state={k:v.detach().clone() for k,v in adapter.state_dict().items()}
            if best is None or score>best[0]: best=(score,state,row)

    if best is None:
        raise RuntimeError("no v3.5 adapter passed positive/negative/preservation/retention guards")
    adapter.load_state_dict(best[1]); adapter.eval()

    after_new=v34._teacher_eval(parent,adapter,torch,positives)
    after_neg=_negative_eval(parent,adapter,torch,negatives)
    after_pres=v34._preservation_eval(parent,adapter,torch,preservation,set(kinds))
    after_ret=v34._retention_eval(parent,adapter,torch,elite,retention_eval)
    gate,gate_diag=_build_gate(torch,positives,negatives,ratio=a.gate_ratio)

    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(parent_state,a.output)
    sidecar=Path(str(a.output)+".adapter.pt")
    torch.save({
        "schema_version":SCHEMA_V2,
        "input_dim":int(input_dim),"hidden_dim":int(a.hidden_dim),
        "kinds":kinds,
        "state_dict":{k:v.detach().clone() for k,v in adapter.state_dict().items()},
        "gate":gate,
    },sidecar)

    report={
        "schema_version":"sts1-build-rescue-v35-generalization-adapter",
        "teacher_examples":len(positives),
        "teacher_seeds":sorted({int(r["seed"]) for r in positives}),
        "negative_examples":len(negatives),
        "negative_seeds":sorted({int(r["seed"]) for r in negatives}),
        "enabled_kinds":kinds,
        "selected_epoch":int(best[2]["epoch"]),
        "retention_policy":{"base_max_parent_kl":float(a.max_parent_kl),"max_kl_increase":float(a.max_kl_increase),"allowed_parent_kl":float(allowed_parent_kl)},
        "before":{"new_win":before_new,"negative":before_neg,"preservation":before_pres,"retention":before_ret},
        "after":{"new_win":after_new,"negative":after_neg,"preservation":after_pres,"retention":after_ret},
        "gate":gate_diag,
        "history":history,
    }
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("V35_ADAPTER_TRAIN_PASS",json.dumps({
        "selected_epoch":int(best[2]["epoch"]),
        "teacher_seeds":len({int(r["seed"]) for r in positives}),
        "negative_examples":len(negatives),
        "fully_learned_seeds":after_new["fully_learned_seeds"],
        "negative_rejected_top1_rate":after_neg["rejected_top1_rate"],
        "preservation_top1":after_pres["top1"],
        "parent_kl":after_ret["parent_kl"],
        "winner_top1":after_ret["winner_top1"],
        "gate_positive_admitted":gate_diag["positive_admitted"],
        "gate_negative_blocked":gate_diag["negative_blocked"],
        "gate_negative_checked":gate_diag["negative_checked"],
        "sidecar":str(sidecar),
    },sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
