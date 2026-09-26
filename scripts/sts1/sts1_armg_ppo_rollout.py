#!/usr/bin/env python3
"""Collect on-policy ArmG non-combat rollouts while combat remains MCTS-2000."""
from __future__ import annotations
import argparse, importlib, json, math, os, random, sys
from pathlib import Path
import numpy as np
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, run_simulator_game

class SamplingArmG(ArmGNoncombatPolicy):
    def __init__(self,*a,temperature=1.0,**kw):
        super().__init__(*a,**kw); self.temperature=temperature
    def decide(self,gc,sts):
        kind,descs,execs=self.choices(gc)
        if not descs:
            if gc.screen_state==sts.ScreenState.REWARDS:return "reward_empty",-1,[],[],[]
            raise RuntimeError("no legal ArmG choice")
        if len(descs)==1:return kind,0,descs,execs,[0.0]
        _,_,scores=self.score_choices(gc); raw=[float(x) for x in scores.tolist()]
        probs=self.torch.softmax(scores/self.temperature,dim=0)
        idx=int(self.torch.multinomial(probs,1).item())
        return kind,idx,descs,execs,raw

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True);p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--weight",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed-start",type=int,required=True);p.add_argument("--games",type=int,default=25)
    p.add_argument("--temperature",type=float,default=1.0);p.add_argument("--checkpoint-id",required=True)
    a=p.parse_args();sys.path.insert(0,str(a.module_dir));sts=importlib.import_module("slaythespire")
    a.out.mkdir(parents=True,exist_ok=True); rng=random.Random(a.seed_start)
    seeds=[]; seen=set()
    while len(seeds)<a.games:
        s=rng.randrange(1,2**31-1)
        if s not in seen:seen.add(s);seeds.append(s)
    policy=SamplingArmG(root=a.armg_root,weight_path=a.weight,temperature=a.temperature)
    all_rows=[]; wins=0
    for s in seeds:
        ev=a.out/f"seed-{s}.ndjson"
        result=run_simulator_game(student=None,sts=sts,seed=s,evidence_path=ev,armg_policy=policy,combat_mcts_sims=2000,training_seeds=seeds)
        rows=[json.loads(x) for x in ev.read_text().splitlines() if x.strip()]
        dec=[x for x in rows if x.get("type")=="armg_noncombat_decision_v3" and x.get("selected_index",-1)>=0 and len(x.get("candidate_desc_368",[]))>1]
        outcome=result.get("outcome",""); victory=str(outcome).lower()=="victory"; wins+=int(victory)
        for i,r in enumerate(dec):
            scores=np.asarray(r["choice_scores"],dtype=np.float64)/a.temperature
            scores-=scores.max(); probs=np.exp(scores);probs/=probs.sum()
            floor=int(r.get("floor") or 0); nf=int(dec[i+1].get("floor") or floor) if i+1<len(dec) else int(result.get("max_floor") or floor)
            hp=float(r.get("hp_before") or 0); nhp=float(dec[i+1].get("hp_before") or hp) if i+1<len(dec) else hp
            reward=0.10*max(0,nf-floor)+0.01*(nhp-hp)
            if i==len(dec)-1: reward += 10.0 if victory else -2.0
            all_rows.append((s,r,reward,float(math.log(max(1e-12,probs[int(r["selected_index"])])))))
    np.savez_compressed(a.out/"rollout.npz",
      seed=np.array([x[0] for x in all_rows]),obs=np.array([x[1]["obs_412"] for x in all_rows],dtype=np.float32),
      desc=np.array([x[1]["candidate_desc_368"] for x in all_rows],dtype=object),action=np.array([x[1]["selected_index"] for x in all_rows]),
      reward=np.array([x[2] for x in all_rows],dtype=np.float32),old_logp=np.array([x[3] for x in all_rows],dtype=np.float32),
      checkpoint_id=np.array([a.checkpoint_id]*len(all_rows)),game_seed=np.array([x[0] for x in all_rows]))
    print("ARMG_PPO_ROLLOUT",json.dumps({"games":len(seeds),"wins":wins,"decisions":len(all_rows),"checkpoint_id":a.checkpoint_id}))

if __name__=="__main__":main()
