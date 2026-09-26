#!/usr/bin/env python3
"""Offline clipped-PPO update for ArmG from checkpoint-tagged rollout NPZ files."""
from __future__ import annotations
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument("--rollouts",type=Path,required=True);p.add_argument("--armg-root",type=Path,required=True);p.add_argument("--base-weight",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--checkpoint-id",required=True);p.add_argument("--epochs",type=int,default=4);p.add_argument("--lr",type=float,default=2e-5);p.add_argument("--clip",type=float,default=.2);p.add_argument("--gamma",type=float,default=.99);p.add_argument("--entropy",type=float,default=.01);a=p.parse_args()
 os.environ["STS_BOT_DIR"]=str(a.armg_root);sys.path.insert(0,str(a.armg_root));torch=importlib.import_module("torch");m=importlib.import_module("armG_train");m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)
 files=sorted(a.rollouts.rglob("rollout.npz")); rows=[]
 for f in files:
  d=np.load(f,allow_pickle=True); ids=set(map(str,d["checkpoint_id"]))
  if ids!={a.checkpoint_id}: print("STALE_ROLLOUT_REJECT",f,ids);continue
  for i in range(len(d["action"])):rows.append((int(d["game_seed"][i]),d["obs"][i],d["desc"][i],int(d["action"][i]),float(d["reward"][i]),float(d["old_logp"][i])))
 if not rows:raise SystemExit("no current-checkpoint rollouts")
 net=m.Scorer((128,128));net.load_state_dict(torch.load(a.base_weight,weights_only=True,map_location="cpu"));opt=torch.optim.Adam(net.parameters(),lr=a.lr)
 # Monte-Carlo returns per game; standardized advantage is a simple baseline for PPO v1.
 adv=np.zeros(len(rows),dtype=np.float32)
 by={}
 for i,r in enumerate(rows):by.setdefault(r[0],[]).append(i)
 for inds in by.values():
  ret=0.
  for i in reversed(inds):ret=rows[i][4]+a.gamma*ret;adv[i]=ret
 adv=(adv-adv.mean())/(adv.std()+1e-8)
 history=[]
 for ep in range(1,a.epochs+1):
  total=0.;kl=0.
  for i,r in enumerate(rows):
   _,obs,descs,act,_,old=r;obs_t=torch.tensor(obs,dtype=torch.float32); logits=net.score(obs_t,list(descs)); lp=torch.log_softmax(logits,0); new=lp[act]; ratio=torch.exp(new-old); A=torch.tensor(float(adv[i])); clipped=torch.clamp(ratio,1-a.clip,1+a.clip)*A; policy=-torch.minimum(ratio*A,clipped); entropy=-(torch.softmax(logits,0)*lp).sum();loss=policy-a.entropy*entropy
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();total+=float(loss.detach());kl+=old-float(new.detach())
  rec={"epoch":ep,"loss":total/len(rows),"approx_kl":kl/len(rows)};history.append(rec);print(json.dumps(rec))
  if abs(rec["approx_kl"])>.03:break
 a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),a.output);report={"schema":"sts1-armg-ppo-v1","rollouts":len(files),"games":len(by),"decisions":len(rows),"checkpoint_id":a.checkpoint_id,"history":history};a.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n");print("ARMG_PPO_TRAIN",json.dumps(report))
if __name__=="__main__":main()
