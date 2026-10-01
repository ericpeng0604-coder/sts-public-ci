#!/usr/bin/env python3
"""Gradient-projected Tail2 update for one verified STS1 New-Win decision.

The early G7 backbone stays frozen.  Only the last hidden Linear layer and the
final scalar scoring head may move.  At each surgical step, the target margin
gradient is projected away from a bank of winner-replay and regression-anchor
margin gradients, then the smallest predicted step toward the teacher action is
applied.  Exact G7 retention gates still decide whether a candidate is valid.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

import sts1_build_rescue_bc_v28 as base


def _tail_params(actor):
    linear_layers = [
        layer for layer in actor.net
        if hasattr(layer, "in_features") and hasattr(layer, "out_features")
    ]
    if len(linear_layers) < 2:
        raise RuntimeError("unable to locate final two ArmG Linear layers")
    layers = linear_layers[-2:]
    params = []
    for layer in layers:
        params.extend(list(layer.parameters()))
    return layers, params


def _flatten_grads(torch, grads, params):
    parts=[]
    for g,p in zip(grads,params):
        if g is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(g.reshape(-1))
    return torch.cat(parts)


def _flatten_params(torch, params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def _assign_flat(torch, params, flat):
    offset=0
    with torch.no_grad():
        for p in params:
            n=p.numel()
            p.copy_(flat[offset:offset+n].reshape_as(p))
            offset+=n
    if offset!=flat.numel():
        raise RuntimeError("flat parameter assignment size mismatch")


def _scores(actor, torch, row):
    obs=torch.tensor(row["obs"],dtype=torch.float32)
    desc=torch.tensor(row["descs"],dtype=torch.float32)
    x=torch.cat([obs.repeat(len(desc),1),desc],1)
    return actor.net(x).squeeze(1)


def _margin_grad(actor, torch, row, target, competitor, params):
    actor.zero_grad(set_to_none=True)
    logits=_scores(actor,torch,row)
    margin=logits[int(target)]-logits[int(competitor)]
    grads=torch.autograd.grad(margin,params,allow_unused=True,retain_graph=False)
    return float(margin.detach()), _flatten_grads(torch,grads,params)


def _elite_row(elite,i):
    lo=int(elite["offsets"][i]); hi=int(elite["offsets"][i+1])
    return {
        "obs":[float(v) for v in elite["obs"][i]],
        "descs":[[float(v) for v in d] for d in elite["desc"][lo:hi]],
    }


def _protected_grad_bank(
    actor,parent,torch,elite,indices,preservation,params,max_rows,
):
    rows=[]
    with torch.no_grad():
        for raw in indices:
            i=int(raw)
            row=_elite_row(elite,i)
            logits=_scores(parent,torch,row)
            if len(logits)<2:
                continue
            order=torch.argsort(logits,descending=True)
            top=int(order[0]); runner=int(order[1])
            probs=torch.softmax(logits,0)
            gap=float(probs[top]-probs[runner])
            rows.append((row,top,runner,4.0 if gap>=base.PARENT_HIGH_CONFIDENCE_PROB_GAP else 1.0))
            if len(rows)>=max_rows:
                break
    for row in preservation:
        with torch.no_grad():
            logits=_scores(parent,torch,row)
            target=int(row["parent_selected_index"])
            order=torch.argsort(logits,descending=True)
            competitor=next(int(x) for x in order if int(x)!=target)
        rows.append((row,target,competitor,25.0))

    bank=[]
    for row,target,competitor,weight in rows:
        _,g=_margin_grad(actor,torch,row,target,competitor,params)
        norm=float(torch.linalg.vector_norm(g))
        if norm<=1e-10:
            continue
        bank.append(g*(float(weight)**0.5)/norm)
    if not bank:
        raise RuntimeError("protected gradient bank is empty")
    return torch.stack(bank,0)


def _project_direction(torch,g,bank,lam):
    # g_proj = g - G^T (G G^T + lam I)^-1 G g
    gram=bank@bank.T
    rhs=bank@g
    eye=torch.eye(gram.shape[0],dtype=gram.dtype)
    coeff=torch.linalg.solve(gram+float(lam)*eye,rhs)
    proj=g-bank.T@coeff
    return proj


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--base-weight",type=Path,required=True)
    p.add_argument("--elite-replay",type=Path,required=True)
    p.add_argument("--new-win-replay",type=Path,required=True)
    p.add_argument("--preservation-replay",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--retention-eval-max",type=int,default=2048)
    p.add_argument("--protect-grad-max",type=int,default=96)
    p.add_argument("--target-margin",type=float,default=0.01)
    p.add_argument("--max-steps",type=int,default=12)
    p.add_argument("--step-fraction",type=float,default=0.50)
    p.add_argument("--raw-kl-stop",type=float,default=0.02)
    p.add_argument("--min-parent-agreement",type=float,default=0.995)
    p.add_argument("--max-parent-kl",type=float,default=0.002)
    p.add_argument("--max-winner-drop",type=float,default=0.01)
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()

    if not 32<=a.protect_grad_max<=256:
        raise RuntimeError("protect-grad-max outside safe bound")
    if not 0<a.target_margin<=0.10:
        raise RuntimeError("target-margin outside safe bound")
    if not 1<=a.max_steps<=30:
        raise RuntimeError("max-steps outside safe bound")
    if not 0<a.step_fraction<=1:
        raise RuntimeError("step-fraction outside safe bound")
    if not a.max_parent_kl<a.raw_kl_stop<=0.10:
        raise RuntimeError("raw-kl-stop invalid")

    os.environ["STS_BOT_DIR"]=str(a.armg_root)
    sys.path.insert(0,str(a.armg_root))
    torch=importlib.import_module("torch")
    torch.set_num_threads(a.threads)
    torch.manual_seed(20261001)
    m=importlib.import_module("armG_train")
    m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)

    parent_state=torch.load(a.base_weight,weights_only=True,map_location="cpu")
    actor=m.Scorer((128,128)); actor.load_state_dict(parent_state)
    parent=m.Scorer((128,128)); parent.load_state_dict(parent_state); parent.eval()
    for p0 in parent.parameters(): p0.requires_grad_(False)

    teachers=base._load_rescues(a.new_win_replay)
    if len(teachers)!=1:
        raise RuntimeError(f"v2.10 requires exactly one focused Teacher; got {len(teachers)}")
    teacher=teachers[0]
    preservation=base._load_preservation(a.preservation_replay)
    elite=base._load_elite(a.elite_replay)

    # Freeze early backbone, enable only Tail2.
    for p0 in actor.parameters(): p0.requires_grad_(False)
    tail_layers,params=_tail_params(actor)
    for layer in tail_layers:
        for p0 in layer.parameters(): p0.requires_grad_(True)

    rng=np.random.default_rng(20261001)
    perm=rng.permutation(len(elite["action"]))
    eval_count=min(a.retention_eval_max,max(64,len(perm)//5))
    if len(perm)-eval_count<128:
        eval_count=max(1,len(perm)//4)
    retention_eval=np.asarray(perm[:eval_count],dtype=np.int64)
    protection_pool=np.asarray(perm[eval_count:],dtype=np.int64)

    before_new=base._eval_new_win(actor,torch,teachers)
    before_pres=base._eval_preservation(actor,torch,preservation)
    before_ret=base._eval_retention(actor,parent,torch,elite,retention_eval)
    if before_new["top1"]!=0.0:
        raise RuntimeError("focused Teacher already top1")
    if before_pres["top1"]<1.0:
        raise RuntimeError("G7 fails frozen preservation anchors")

    base_flat=_flatten_params(torch,params).clone()
    lambdas=(1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0)
    trials=[]
    best=None

    for lam in lambdas:
        actor.load_state_dict(parent_state)
        for p0 in actor.parameters(): p0.requires_grad_(False)
        tail_layers,params=_tail_params(actor)
        for layer in tail_layers:
            for p0 in layer.parameters(): p0.requires_grad_(True)
        base_flat=_flatten_params(torch,params).clone()

        bank=_protected_grad_bank(
            actor,parent,torch,elite,protection_pool,preservation,params,a.protect_grad_max
        )
        steps=[]
        stopped=None
        for step_idx in range(a.max_steps):
            logits=_scores(actor,torch,teacher)
            t=int(teacher["teacher_best_index"])
            others=[i for i in range(len(logits)) if i!=t]
            comp=max(others,key=lambda i:float(logits[i].detach()))
            raw=float((logits[t]-logits[comp]).detach())
            if int(torch.argmax(logits))==t and raw>=a.target_margin:
                stopped="target_flipped"
                break

            margin,g=_margin_grad(actor,torch,teacher,t,comp,params)
            direction=_project_direction(torch,g,bank,lam)
            gain=float(torch.dot(g,direction))
            dnorm=float(torch.linalg.vector_norm(direction))
            if not np.isfinite(gain) or gain<=1e-12 or dnorm<=1e-12:
                stopped="projected_gradient_degenerate"
                break
            need=max(0.0,a.target_margin-margin)
            scale=a.step_fraction*need/gain
            flat=_flatten_params(torch,params)
            _assign_flat(torch,params,flat+scale*direction)
            actor.eval()
            ret=base._eval_retention(actor,parent,torch,elite,retention_eval)
            new=base._eval_new_win(actor,torch,teachers)
            steps.append({
                "step":step_idx+1,
                "margin_before":margin,
                "competitor":int(comp),
                "projected_gain":gain,
                "direction_norm":dnorm,
                "scale":float(scale),
                "parent_kl":ret["parent_kl"],
                "new_win_top1":new["top1"],
            })
            if float(ret["parent_kl"])>=a.raw_kl_stop and new["top1"]<1.0:
                stopped="raw_kl_stop"
                break

        # Search interpolation from parent to the surgical raw update.
        raw_flat=_flatten_params(torch,params).detach().clone()
        delta=raw_flat-base_flat
        for alpha in (1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.125):
            _assign_flat(torch,params,base_flat+float(alpha)*delta)
            actor.eval()
            new=base._eval_new_win(actor,torch,teachers)
            pres=base._eval_preservation(actor,torch,preservation)
            ret=base._eval_retention(actor,parent,torch,elite,retention_eval)
            target_ok=int(new["fully_learned_seeds"])==1
            pres_ok=pres["top1"]>=1.0
            ret_ok=base._retention_ok(
                ret,before_ret,
                min_parent_agreement=a.min_parent_agreement,
                max_parent_kl=a.max_parent_kl,
                max_winner_drop=a.max_winner_drop,
            )
            ok=target_ok and pres_ok and ret_ok
            trial={
                "lambda":lam,
                "alpha":alpha,
                "stopped":stopped,
                "steps":steps,
                "new_win":new,
                "preservation":pres,
                "retention":ret,
                "target_pass":bool(target_ok),
                "preservation_pass":bool(pres_ok),
                "retention_pass":bool(ret_ok),
                "pass":bool(ok),
                "delta_norm":float(torch.linalg.vector_norm(delta)),
            }
            trials.append(trial)
            print("V210_GRADPROJ_TRIAL",json.dumps(trial,sort_keys=True),flush=True)
            if ok:
                score=(
                    float(ret["parent_high_conf_top1_agreement"]),
                    -float(ret["parent_kl"]),
                    float(ret["winner_top1"]),
                    -float(trial["delta_norm"]*alpha),
                )
                if best is None or score>best[0]:
                    best=(score,{k:v.detach().clone() for k,v in actor.state_dict().items()},trial)

    if best is None:
        raise RuntimeError("no gradient-projected Tail2 candidate flipped target while passing G7 guards")

    actor.load_state_dict(best[1]); actor.eval()
    after_new=base._eval_new_win(actor,torch,teachers)
    after_pres=base._eval_preservation(actor,torch,preservation)
    after_ret=base._eval_retention(actor,parent,torch,elite,retention_eval)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(actor.state_dict(),a.output)
    report={
        "schema_version":"sts1-build-rescue-v210-gradproj-one",
        "teacher_seed":int(teacher["seed"]),
        "teacher_kind":str(teacher["kind"]),
        "teacher_floor":int(teacher["floor"]),
        "before":{"new_win":before_new,"preservation":before_pres,"retention":before_ret},
        "after":{"new_win":after_new,"preservation":after_pres,"retention":after_ret},
        "selected_trial":best[2],
        "trials":trials,
        "trainable_parameter_tensors":len(params),
    }
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("V210_GRADPROJ_ONE_PASS",json.dumps({
        "teacher_seed":int(teacher["seed"]),
        "teacher_kind":str(teacher["kind"]),
        "teacher_floor":int(teacher["floor"]),
        "new_win_top1_after":after_new["top1"],
        "preservation_top1_after":after_pres["top1"],
        "parent_high_conf_top1_agreement":after_ret["parent_high_conf_top1_agreement"],
        "parent_top1_agreement":after_ret["parent_top1_agreement"],
        "parent_kl":after_ret["parent_kl"],
        "winner_top1_after":after_ret["winner_top1"],
        "selected_lambda":best[2]["lambda"],
        "selected_alpha":best[2]["alpha"],
    },sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
