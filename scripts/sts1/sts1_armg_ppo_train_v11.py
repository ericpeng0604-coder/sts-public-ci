#!/usr/bin/env python3
"""ArmG PPO v1.1: actor + critic, GAE, minibatches and KL stop."""
from __future__ import annotations
import argparse,importlib,json,os,random,sys
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser()
 for n in ("rollouts","armg-root","base-weight","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--checkpoint-id",required=True);p.add_argument("--epochs",type=int,default=4);p.add_argument("--lr",type=float,default=1e-5);p.add_argument("--clip",type=float,default=.15);p.add_argument("--gamma",type=float,default=.99);p.add_argument("--lam",type=float,default=.95);p.add_argument("--entropy",type=float,default=.01);p.add_argument("--value-coef",type=float,default=.5);p.add_argument("--target-kl",type=float,default=.02);p.add_argument("--batch-size",type=int,default=128)
 a=p.parse_args();os.environ["STS_BOT_DIR"]=str(a.armg_root);sys.path.insert(0,str(a.armg_root));torch=importlib.import_module("torch");m=importlib.import_module("armG_train");m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)
 rows=[];files=sorted(a.rollouts.rglob("rollout.npz"))
 for f in files:
  d=np.load(f,allow_pickle=True);ids=set(map(str,d["checkpoint_id"]))
  if ids!={a.checkpoint_id}:print("STALE_ROLLOUT_REJECT",f,ids);continue
  for i in range(len(d["action"])):rows.append((int(d["game_seed"][i]),d["obs"][i],d["desc"][i],int(d["action"][i]),float(d["reward"][i]),float(d["old_logp"][i]),bool(d["done"][i])))
 if not rows:raise SystemExit("no current-checkpoint rollouts")
 actor=m.Scorer((128,128));actor.load_state_dict(torch.load(a.base_weight,weights_only=True,map_location="cpu"))
 critic=torch.nn.Sequential(torch.nn.Linear(412,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,1))
 opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=a.lr)
 obs=torch.tensor(np.stack([r[1] for r in rows]),dtype=torch.float32)
 with torch.no_grad():v0=critic(obs).squeeze(-1).numpy()
 adv=np.zeros(len(rows),np.float32);ret=np.zeros(len(rows),np.float32);by={}
 for i,r in enumerate(rows):by.setdefault(r[0],[]).append(i)
 for inds in by.values():
  gae=0.
  for pos in range(len(inds)-1,-1,-1):
   i=inds[pos];done=rows[i][6];nv=0. if done else v0[inds[pos+1]];delta=rows[i][4]+a.gamma*nv-v0[i];gae=delta+a.gamma*a.lam*(0. if done else gae);adv[i]=gae;ret[i]=gae+v0[i]
 adv=(adv-adv.mean())/(adv.std()+1e-8);history=[];idx=list(range(len(rows)));random.Random(20260927).shuffle(idx)
 for ep in range(1,a.epochs+1):
  random.Random(20260927+ep).shuffle(idx);losses=[];kls=[]
  for st in range(0,len(idx),a.batch_size):
   batch=idx[st:st+a.batch_size];pl=[];vl=[];ent=[];kl=[]
   for i in batch:
    _,ob,descs,act,_,old,_=rows[i];logits=actor.score(torch.tensor(ob,dtype=torch.float32),list(descs));lp=torch.log_softmax(logits,0);new=lp[act];ratio=torch.exp(new-old);A=torch.tensor(float(adv[i]));pl.append(-torch.minimum(ratio*A,torch.clamp(ratio,1-a.clip,1+a.clip)*A));ent.append(-(torch.softmax(logits,0)*lp).sum());kl.append(old-new)
   bobs=obs[batch];pred=critic(bobs).squeeze(-1);target=torch.tensor(ret[batch]);loss=torch.stack(pl).mean()+a.value_coef*torch.nn.functional.mse_loss(pred,target)-a.entropy*torch.stack(ent).mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),1.0);opt.step();losses.append(float(loss.detach()));kls.extend(float(x.detach()) for x in kl)
  rec={"epoch":ep,"loss":float(np.mean(losses)),"approx_kl":float(np.mean(kls))};history.append(rec);print(json.dumps(rec))
  if rec["approx_kl"]>a.target_kl:print("TARGET_KL_STOP",rec["approx_kl"]);break
 a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(actor.state_dict(),a.output);torch.save(critic.state_dict(),a.output.with_name(a.output.stem+"_critic.pt"));report={"schema":"sts1-armg-ppo-v1.1","rollouts":len(files),"games":len(by),"decisions":len(rows),"checkpoint_id":a.checkpoint_id,"history":history};a.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n");print("ARMG_PPO_V11_TRAIN",json.dumps(report))
if __name__=="__main__":main()
