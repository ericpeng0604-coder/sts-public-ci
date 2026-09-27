#!/usr/bin/env python3
import argparse,gc,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--rollouts",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--checkpoint-id",required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
files=sorted(a.rollouts.rglob("rollout.npz"));manifest={"schema":"sts1-ppo-v11-sharded","checkpoint_id":a.checkpoint_id,"shards":[]};t0=time.time()
for si,f in enumerate(files):
 t=time.time()
 with np.load(f,allow_pickle=True) as d:
  ids=set(map(str,d["checkpoint_id"]))
  if ids!={a.checkpoint_id}:raise SystemExit(f"stale {f}: {ids}")
  n=len(d["action"]);obs=np.asarray([np.asarray(x,np.float32) for x in d["obs"]],np.float32)
  counts=np.fromiter((len(x) for x in d["desc"]),dtype=np.int32,count=n);offsets=np.zeros(n+1,np.int64);np.cumsum(counts,out=offsets[1:])
  # Allocate exactly once, then copy candidates directly instead of keeping duplicate nested lists.
  first=np.asarray(d["desc"][0],np.float32);dd=first.shape[-1];desc=np.empty((int(offsets[-1]),dd),np.float32)
  for i,x in enumerate(d["desc"]):desc[offsets[i]:offsets[i+1]]=np.asarray(x,np.float32)
  out=a.output_dir/f"shard_{si:02d}.npz";np.savez(out,obs=obs,desc=desc,offsets=offsets,counts=counts,action=np.asarray(d["action"],np.int64),reward=np.asarray(d["reward"],np.float32),old_logp=np.asarray(d["old_logp"],np.float32),done=np.asarray(d["done"],np.bool_),game_seed=np.asarray(d["game_seed"],np.int64),checkpoint_id=np.asarray([a.checkpoint_id]))
  rec={"shard":si,"source":f.name,"decisions":n,"candidates":int(offsets[-1]),"bytes":out.stat().st_size,"seconds":time.time()-t};manifest["shards"].append(rec);print("SHARD_PASS",json.dumps(rec),flush=True)
 del obs,counts,offsets,desc,first;gc.collect()
manifest["decisions"]=sum(x["decisions"] for x in manifest["shards"]);manifest["candidates"]=sum(x["candidates"] for x in manifest["shards"]);manifest["seconds"]=time.time()-t0;(a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print("SHARDED_PREPROCESS_PASS",json.dumps(manifest),flush=True)
