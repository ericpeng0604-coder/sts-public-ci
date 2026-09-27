#!/usr/bin/env python3
"""Fully vectorized PPO v1.1 learner for ArmG."""
from __future__ import annotations
import argparse,importlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser()
 for n in ("rollouts","armg-root","base-weight","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--checkpoint-id",required=True);p.add_argument("--epochs",type=int,default=2);p.add_argument("--lr",type=float,default=1e-5);p.add_argument("--clip",type=float,default=.15);p.add_argument("--gamma",type=float,default=.99);p.add_argument("--lam",type=float,default=.95);p.add_argument("--entropy",type=float,default=.01);p.add_argument("--value-coef",type=float,default=.5);p.add_argument("--target-kl",type=float,default=.02);p.add_argument("--batch-size",type=int,default=512);p.add_argument("--threads",type=int,default=4)
 a=p.parse_args();os.environ["STS_BOT_DIR"]=str(a.armg_root);sys.path.insert(0,str(a.armg_root));torch=importlib.import_module("torch");torch.set_num_threads(a.threads);m=importlib.import_module("armG_train");m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)
 rows=[];files=sorted(a.rollouts.rglob("rollout.npz"))
 for f in files:
  d=np.load(f,allow_pickle=True);ids=set(map(str,d["checkpoint_id"]))
  if ids!={a.checkpoint_id}:continue
  rows.extend((int(d["game_seed"][i]),np.asarray(d["obs"][i],np.float32),np.asarray(d["desc"][i],np.float32),int(d["action"][i]),float(d["reward"][i]),float(d["old_logp"][i]),bool(d["done"][i])) for i in range(len(d["action"])))
 if not rows:raise SystemExit("no rollouts")
 actor=m.Scorer((128,128));actor.load_state_dict(torch.load(a.base_weight,weights_only=True,map_location="cpu"))
 critic=torch.nn.Sequential(torch.nn.Linear(m.OBS_DIM,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,1))
 obs=torch.from_numpy(np.stack([r[1] for r in rows]));old=torch.tensor([r[5] for r in rows],dtype=torch.float32)
 with torch.no_grad():v0=critic(obs).squeeze(1).numpy()
 adv=np.zeros(len(rows),np.float32);ret=np.zeros(len(rows),np.float32);by={}
 for i,r in enumerate(rows):by.setdefault(r[0],[]).append(i)
 for inds in by.values():
  gae=0.
  for pos in range(len(inds)-1,-1,-1):
   i=inds[pos];done=rows[i][6];nv=0. if done else v0[inds[pos+1]];delta=rows[i][4]+a.gamma*nv-v0[i];gae=delta+a.gamma*a.lam*(0. if done else gae);adv[i]=gae;ret[i]=gae+v0[i]
 adv=(adv-adv.mean())/(adv.std()+1e-8);adv=torch.from_numpy(adv);ret=torch.from_numpy(ret)
 # Cache dense inputs grouped by candidate count. Each bucket forward-scores B*K choices at once.
 buckets={}
 for i,r in enumerate(rows):
  k=len(r[2]);b=buckets.setdefault(k,{"idx":[],"x":[],"act":[]});b["idx"].append(i);b["act"].append(r[3]);b["x"].append(np.concatenate([np.repeat(r[1][None,:],k,0),r[2]],1))
 for k,b in buckets.items():b["x"]=torch.from_numpy(np.stack(b["x"]).astype(np.float32));b["act"]=torch.tensor(b["act"],dtype=torch.long);b["pos"]={v:j for j,v in enumerate(b["idx"])}
 print("VECTORIZED_BUCKETS",json.dumps({k:len(b["idx"]) for k,b in buckets.items()}),flush=True)
 opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=a.lr);allidx=list(range(len(rows)));hist=[];a.output.parent.mkdir(parents=True,exist_ok=True)
 for ep in range(1,a.epochs+1):
  random.Random(20260927+ep).shuffle(allidx);ls=[];ks=[];t=time.time()
  for st in range(0,len(allidx),a.batch_size):
   batch=allidx[st:st+a.batch_size];parts=[];entparts=[];ordered=[]
   # Usually only a handful of K values, so this replaces hundreds of Python actor.score calls with a few big forwards.
   for k,b in buckets.items():
    sel=[i for i in batch if i in b["pos"]]
    if not sel:continue
    pp=torch.tensor([b["pos"][i] for i in sel]);x=b["x"][pp];logits=actor.net(x.reshape(-1,m.INPUT_DIM)).reshape(len(sel),k);lp=torch.log_softmax(logits,1);pr=torch.softmax(logits,1);acts=b["act"][pp];parts.append(lp.gather(1,acts[:,None]).squeeze(1));entparts.append(-(pr*lp).sum(1));ordered.extend(sel)
   new=torch.cat(parts);ent=torch.cat(entparts);order=torch.tensor(ordered);A=adv[order];ov=old[order];ratio=torch.exp(new-ov);pg=-torch.minimum(ratio*A,torch.clamp(ratio,1-a.clip,1+a.clip)*A).mean();vl=torch.nn.functional.mse_loss(critic(obs[order]).squeeze(1),ret[order]);loss=pg+a.value_coef*vl-a.entropy*ent.mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),1.0);opt.step();ls.append(float(loss.detach()));ks.extend((ov-new.detach()).tolist())
  rec={"epoch":ep,"loss":float(np.mean(ls)),"approx_kl":float(np.mean(ks)),"seconds":time.time()-t};hist.append(rec);print("VECTORIZED_EPOCH",json.dumps(rec),flush=True);torch.save(actor.state_dict(),a.output);torch.save(critic.state_dict(),a.output.with_name(a.output.stem+"_critic.pt"));(a.output.parent/"checkpoint.json").write_text(json.dumps({"completed_epoch":ep,"history":hist},indent=2)+"\n")
  if rec["approx_kl"]>a.target_kl:break
 rep={"schema":"sts1-armg-ppo-v1.1-vectorized","rollouts":len(files),"games":len(by),"decisions":len(rows),"history":hist};a.output.with_suffix(".json").write_text(json.dumps(rep,indent=2)+"\n");print("VECTORIZED_TRAIN_PASS",json.dumps(rep),flush=True)
if __name__=="__main__":main()
