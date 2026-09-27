#!/usr/bin/env python3
"""Collect ArmG PPO v1.1 episodes; combat stays MCTS-2000."""
from __future__ import annotations
import argparse,importlib,json,math,random,sys
from pathlib import Path
import numpy as np
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,run_simulator_game
class SamplingArmG(ArmGNoncombatPolicy):
 def __init__(self,*a,temperature=1.,torch_seed=0,**kw):
  super().__init__(*a,**kw);self.temperature=temperature;self.torch.manual_seed(torch_seed)
 def decide(self,gc,sts):
  kind,descs,execs=self.choices(gc)
  if not descs:
   if gc.screen_state==sts.ScreenState.REWARDS:return "reward_empty",-1,[],[],[]
   raise RuntimeError("no legal ArmG choice")
  if len(descs)==1:return kind,0,descs,execs,[0.]
  _,_,scores=self.score_choices(gc);raw=[float(x) for x in scores.tolist()]
  probs=self.torch.softmax(scores/self.temperature,0);idx=int(self.torch.multinomial(probs,1).item())
  return kind,idx,descs,execs,raw
def main():
 p=argparse.ArgumentParser()
 for n in ("module-dir","armg-root","weight","out","formal-seed-file"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--seed-start",type=int,required=True);p.add_argument("--games",type=int,default=50);p.add_argument("--temperature",type=float,default=1.);p.add_argument("--checkpoint-id",required=True);p.add_argument("--worker",type=int,default=0)
 a=p.parse_args();sys.path.insert(0,str(a.module_dir));sts=importlib.import_module("slaythespire");a.out.mkdir(parents=True,exist_ok=True)
 formal={int(x) for x in a.formal_seed_file.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")}
 rng=random.Random(a.seed_start);seeds=[];seen=set(formal)
 while len(seeds)<a.games:
  s=rng.randrange(1,2**31-1)
  if s not in seen:seen.add(s);seeds.append(s)
 policy=SamplingArmG(root=a.armg_root,weight_path=a.weight,temperature=a.temperature,torch_seed=a.seed_start+a.worker)
 rows=[];wins=0
 for s in seeds:
  ev=a.out/f"seed-{s}.ndjson";res=run_simulator_game(student=None,sts=sts,seed=s,evidence_path=ev,armg_policy=policy,combat_mcts_sims=2000,training_seeds=seeds)
  js=[json.loads(x) for x in ev.read_text().splitlines() if x.strip()];dec=[x for x in js if x.get("type")=="armg_noncombat_decision_v3" and x.get("selected_index",-1)>=0 and len(x.get("candidate_desc_368",[]))>1]
  victory=str(res.get("outcome","")).lower()=="victory";wins+=int(victory)
  for i,r in enumerate(dec):
   sc=np.asarray(r["choice_scores"],dtype=np.float64)/a.temperature;sc-=sc.max();pr=np.exp(sc);pr/=pr.sum();fl=int(r.get("floor") or 0);nf=int(dec[i+1].get("floor") or fl) if i+1<len(dec) else int(res.get("max_floor") or res.get("final_floor") or fl);hp=float(r.get("hp_before") or 0);nh=float(dec[i+1].get("hp_before") or hp) if i+1<len(dec) else hp
   rew=.10*max(0,nf-fl)+.01*(nh-hp)+(10. if victory and i==len(dec)-1 else -2. if (not victory and i==len(dec)-1) else 0.)
   rows.append((s,r,rew,float(math.log(max(1e-12,pr[int(r["selected_index"])]))),i==len(dec)-1))
 np.savez_compressed(a.out/"rollout.npz",obs=np.array([x[1]["obs_412"] for x in rows],np.float32),desc=np.array([x[1]["candidate_desc_368"] for x in rows],dtype=object),action=np.array([x[1]["selected_index"] for x in rows]),reward=np.array([x[2] for x in rows],np.float32),old_logp=np.array([x[3] for x in rows],np.float32),done=np.array([x[4] for x in rows],np.bool_),checkpoint_id=np.array([a.checkpoint_id]*len(rows)),game_seed=np.array([x[0] for x in rows]))
 print("ARMG_PPO_V11_ROLLOUT",json.dumps({"games":len(seeds),"wins":wins,"decisions":len(rows),"checkpoint_id":a.checkpoint_id}))
if __name__=="__main__":main()
