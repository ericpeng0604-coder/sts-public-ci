#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--rollouts",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--checkpoint-id",required=True);a=p.parse_args()
t=time.time();files=sorted(a.rollouts.rglob("rollout.npz"));print("PREP_FILES",len(files),flush=True)
obs=[];desc=[];action=[];reward=[];old_logp=[];done=[];seed=[];offset=[0]
for f in files:
 d=np.load(f,allow_pickle=True);ids=set(map(str,d["checkpoint_id"]));assert ids=={a.checkpoint_id},(f,ids)
 for i in range(len(d["action"])):
  o=np.asarray(d["obs"][i],np.float32);ds=np.asarray(d["desc"][i],np.float32)
  obs.append(o);desc.append(ds);action.append(int(d["action"][i]));reward.append(float(d["reward"][i]));old_logp.append(float(d["old_logp"][i]));done.append(bool(d["done"][i]));seed.append(int(d["game_seed"][i]));offset.append(offset[-1]+len(ds))
print("PREP_LOAD_SECONDS",round(time.time()-t,3),"decisions",len(action),"choices",offset[-1],flush=True)
t2=time.time();flat=np.concatenate(desc,axis=0).astype(np.float32);a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,obs=np.stack(obs).astype(np.float32),desc_flat=flat,desc_offset=np.asarray(offset,np.int64),action=np.asarray(action,np.int64),reward=np.asarray(reward,np.float32),old_logp=np.asarray(old_logp,np.float32),done=np.asarray(done,np.bool_),game_seed=np.asarray(seed,np.int64),checkpoint_id=np.asarray(a.checkpoint_id))
rep={"schema":"sts1-ppo-v11-preprocessed-v1","source_rollouts":len(files),"decisions":len(action),"candidate_rows":int(offset[-1]),"checkpoint_id":a.checkpoint_id,"load_seconds":round(t2-t,3),"save_seconds":round(time.time()-t2,3)}
a.output.with_suffix(".json").write_text(json.dumps(rep,indent=2)+"\n");print("PREPROCESS_PASS",json.dumps(rep),flush=True)
