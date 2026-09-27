#!/usr/bin/env python3
"""PPO v1.2: stronger but KL-bounded update over preserved v1.1 shards."""
import argparse,importlib,json,os,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for n in ("shards","armg-root","base-weight","output"):p.add_argument("--"+n,type=Path,required=True)
p.add_argument("--checkpoint-id",required=True);p.add_argument("--epochs",type=int,default=4);p.add_argument("--lr",type=float,default=3e-5);p.add_argument("--clip",type=float,default=.20);p.add_argument("--gamma",type=float,default=.99);p.add_argument("--lam",type=float,default=.95);p.add_argument("--entropy",type=float,default=.01);p.add_argument("--value-coef",type=float,default=.5);p.add_argument("--target-kl",type=float,default=.02);p.add_argument("--threads",type=int,default=4)
a=p.parse_args();os.environ["STS_BOT_DIR"]=str(a.armg_root);sys.path.insert(0,str(a.armg_root));torch=importlib.import_module("torch");torch.set_num_threads(a.threads);m=importlib.import_module("armG_train");m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)
actor=m.Scorer((128,128));actor.load_state_dict(torch.load(a.base_weight,weights_only=True,map_location="cpu"));critic=torch.nn.Sequential(torch.nn.Linear(m.OBS_DIM,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,1));opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=a.lr);a.output.parent.mkdir(parents=True,exist_ok=True)
files=sorted(a.shards.glob("shard_*.npz"));assert files,"no shards";hist=[]
for ep in range(1,a.epochs+1):
 losses=[];kls=[];t0=time.time()
 for si,f in enumerate(files):
  t=time.time()
  with np.load(f,allow_pickle=False) as d:
   if str(d["checkpoint_id"][0])!=a.checkpoint_id:raise SystemExit("stale shard")
   obs=torch.from_numpy(d["obs"].astype(np.float32));desc=d["desc"];off=d["offsets"];act=torch.from_numpy(d["action"].astype(np.int64));rew=d["reward"].astype(np.float32);old=torch.from_numpy(d["old_logp"].astype(np.float32));done=d["done"].astype(bool)
   with torch.no_grad():v=critic(obs).squeeze(1).numpy()
   adv=np.zeros(len(rew),np.float32);ret=np.zeros(len(rew),np.float32);gae=0.
   for i in range(len(rew)-1,-1,-1):
    nv=0. if done[i] or i==len(rew)-1 else v[i+1];delta=rew[i]+a.gamma*nv-v[i];gae=delta+a.gamma*a.lam*(0. if done[i] else gae);adv[i]=gae;ret[i]=gae+v[i]
   adv=(adv-adv.mean())/(adv.std()+1e-8);new=[];ents=[]
   # bounded memory: one decision at a time, but candidate scoring within each decision is vectorized.
   for i in range(len(rew)):
    ds=torch.from_numpy(desc[off[i]:off[i+1]].astype(np.float32));o=obs[i].repeat(len(ds),1);logits=actor.net(torch.cat([o,ds],1)).squeeze(1);lp=torch.log_softmax(logits,0);pr=torch.softmax(logits,0);new.append(lp[act[i]]);ents.append(-(pr*lp).sum())
   new=torch.stack(new);ent=torch.stack(ents);A=torch.from_numpy(adv);R=torch.from_numpy(ret);ratio=torch.exp(new-old);pg=-torch.minimum(ratio*A,torch.clamp(ratio,1-a.clip,1+a.clip)*A).mean();vl=torch.nn.functional.mse_loss(critic(obs).squeeze(1),R);loss=pg+a.value_coef*vl-a.entropy*ent.mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),1.0);opt.step();kl=float((old-new.detach()).mean());losses.append(float(loss.detach()));kls.append(kl);print("TRAIN_SHARD",json.dumps({"epoch":ep,"shard":si,"decisions":len(rew),"loss":losses[-1],"kl":kl,"seconds":time.time()-t}),flush=True)
  torch.save(actor.state_dict(),a.output);torch.save(critic.state_dict(),a.output.with_name(a.output.stem+"_critic.pt"));(a.output.parent/"checkpoint.json").write_text(json.dumps({"epoch":ep,"completed_shard":si},indent=2)+"\n")
 rec={"epoch":ep,"loss":float(np.mean(losses)),"approx_kl":float(np.mean(kls)),"seconds":time.time()-t0};hist.append(rec);print("TRAIN_EPOCH",json.dumps(rec),flush=True)
 if rec["approx_kl"]>a.target_kl:break
rep={"schema":"sts1-armg-ppo-v12-sharded-train","shards":len(files),"history":hist};a.output.with_suffix(".json").write_text(json.dumps(rep,indent=2)+"\n");print("SHARDED_TRAIN_PASS",json.dumps(rep),flush=True)
