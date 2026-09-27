#!/usr/bin/env python3
"""Measure ArmG top1/top2 score margins on preserved PPO rollout states."""
import argparse,importlib,json,os,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--shards",type=Path,required=True);p.add_argument("--armg-root",type=Path,required=True);p.add_argument("--weight",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
os.environ["STS_BOT_DIR"]=str(a.armg_root);sys.path.insert(0,str(a.armg_root));torch=importlib.import_module("torch");m=importlib.import_module("armG_train");m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)
net=m.Scorer((128,128));net.load_state_dict(torch.load(a.weight,weights_only=True,map_location="cpu"));net.eval()
margins=[];chosen_top=[];counts=[]
with torch.no_grad():
 for f in sorted(a.shards.glob("shard_*.npz")):
  with np.load(f,allow_pickle=False) as d:
   obs=d["obs"].astype(np.float32);desc=d["desc"].astype(np.float32);off=d["offsets"];act=d["action"]
   for i in range(len(obs)):
    n=int(off[i+1]-off[i]);counts.append(n)
    if n<2: continue
    ds=torch.from_numpy(desc[off[i]:off[i+1]]);o=torch.from_numpy(np.repeat(obs[i:i+1],n,axis=0));s=net.net(torch.cat([o,ds],1)).squeeze(1);v,ix=torch.topk(s,2);margins.append(float(v[0]-v[1]));chosen_top.append(int(ix[0])==int(act[i]))
arr=np.asarray(margins,np.float64)
q={str(x):float(np.quantile(arr,x)) for x in [0,.1,.25,.5,.75,.9,.95,.99,1]}
rep={"schema":"sts1-armg-margin-v1","decisions":len(counts),"multi_choice":len(arr),"chosen_is_top1_rate":float(np.mean(chosen_top)),"margin_quantiles":q,"margin_le_0_05":float(np.mean(arr<=.05)),"margin_le_0_10":float(np.mean(arr<=.10)),"margin_le_0_25":float(np.mean(arr<=.25)),"margin_le_0_50":float(np.mean(arr<=.50))}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rep,indent=2)+"\n");print("MARGIN_DIAG_PASS",json.dumps(rep),flush=True)
