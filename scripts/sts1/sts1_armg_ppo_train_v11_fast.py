#!/usr/bin/env python3
"""PPO v1.1 fast learner: vectorized critic/GAE, cached actor graphs, checkpoint each epoch."""
from __future__ import annotations
import argparse,importlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser()
 for n in ("rollouts","armg-root","base-weight","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--checkpoint-id",required=True);p.add_argument("--epochs",type=int,default=2);p.add_argument("--lr",type=float,default=1e-5);p.add_argument("--clip",type=float,default=.15);p.add_argument("--gamma",type=float,default=.99);p.add_argument("--lam",type=float,default=.95);p.add_argument("--entropy",type=float,default=.01);p.add_argument("--value-coef",type=float,default=.5);p.add_argument("--target-kl",type=float,default=.02);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--threads",type=int,default=4)
 a=p.parse_args();os.environ["STS_BOT_DIR"]=str(a.armg_root);sys.path.insert(0,str(a.armg_root));torch=importlib.import_module("torch");torch.set_num_threads(a.threads);m=importlib.import_module("armG_train");m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)
 rows=[];files=sorted(a.rollouts.rglob("rollout.npz"))
 for f in files:
  d=np.load(f,allow_pickle=True);ids=set(map(str,d["checkpoint_id"]))
  if ids!={a.checkpoint_id}:print("STALE_ROLLOUT_REJECT",f,ids);continue
  rows.extend((int(d["game_seed"][i]),d["obs"][i],d["desc"][i],int(d["action"][i]),float(d["reward"][i]),float(d["old_logp"][i]),bool(d["done"][i])) for i in range(len(d["action"])))
 if not rows:raise SystemExit("no current-checkpoint rollouts")
 actor=m.Scorer((128,128));actor.load_state_dict(torch.load(a.base_weight,weights_only=True,map_location="cpu"))
 critic=torch.nn.Sequential(torch.nn.Linear(412,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,1))
 obs=torch.from_numpy(np.stack([r[1] for r in rows]).astype(np.float32))
 with torch.no_grad():v0=critic(obs).squeeze(1).numpy()
 adv=np.zeros(len(rows),np.float32);ret=np.zeros(len(rows),np.float32);by={}
 for i,r in enumerate(rows):by.setdefault(r[0],[]).append(i)
 for inds in by.values():
  gae=0.
  for pos in range(len(inds)-1,-1,-1):
   i=inds[pos];done=rows[i][6];nv=0. if done else v0[inds[pos+1]];delta=rows[i][4]+a.gamma*nv-v0[i];gae=delta+a.gamma*a.lam*(0. if done else gae);adv[i]=gae;ret[i]=gae+v0[i]
 adv=(adv-adv.mean())/(adv.std()+1e-8);adv_t=torch.from_numpy(adv);ret_t=torch.from_numpy(ret)
 opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=a.lr);idx=list(range(len(rows)));history=[];a.output.parent.mkdir(parents=True,exist_ok=True)
 for ep in range(1,a.epochs+1):
  t=time.time();random.Random(20260927+ep).shuffle(idx);losses=[];kls=[]
  for st in range(0,len(idx),a.batch_size):
   batch=idx[st:st+a.batch_size];pol=[];ents=[];oldv=[]
   # ArmG choices have variable candidate counts; one graph per state is unavoidable,
   # but backward is done once per minibatch instead of once per decision.
   for i in batch:
    _,ob,descs,act,_,old,_=rows[i];logits=actor.score(torch.from_numpy(np.asarray(ob,dtype=np.float32)),list(descs));lp=torch.log_softmax(logits,0);pol.append(lp[act]);ents.append(-(torch.softmax(logits,0)*lp).sum());oldv.append(old)
   new=torch.stack(pol);old=torch.tensor(oldv);A=adv_t[batch];ratio=torch.exp(new-old);pg=-torch.minimum(ratio*A,torch.clamp(ratio,1-a.clip,1+a.clip)*A).mean()
   pred=critic(obs[batch]).squeeze(1);vl=torch.nn.functional.mse_loss(pred,ret_t[batch]);loss=pg+a.value_coef*vl-a.entropy*torch.stack(ents).mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),1.0);opt.step();losses.append(float(loss.detach()));kls.extend((old-new.detach()).tolist())
  rec={"epoch":ep,"loss":float(np.mean(losses)),"approx_kl":float(np.mean(kls)),"seconds":time.time()-t};history.append(rec);print("V11_FAST_EPOCH",json.dumps(rec),flush=True)
  torch.save(actor.state_dict(),a.output);torch.save(critic.state_dict(),a.output.with_name(a.output.stem+"_critic.pt"))
  (a.output.parent/"checkpoint.json").write_text(json.dumps({"completed_epoch":ep,"history":history},indent=2)+"\n")
  if rec["approx_kl"]>a.target_kl:print("TARGET_KL_STOP",rec["approx_kl"],flush=True);break
 report={"schema":"sts1-armg-ppo-v1.1-fast","rollouts":len(files),"games":len(by),"decisions":len(rows),"checkpoint_id":a.checkpoint_id,"history":history};a.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n");print("ARMG_PPO_V11_FAST_TRAIN",json.dumps(report),flush=True)
if __name__=="__main__":main()
