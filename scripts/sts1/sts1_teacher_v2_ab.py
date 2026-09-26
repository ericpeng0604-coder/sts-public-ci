#!/usr/bin/env python3
"""Paired fresh-seed A/B: frozen Teacher v1 vs opt-in contextual Teacher v2."""
import argparse,json,os,random
from pathlib import Path
from statistics import mean
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_load_sts,run_simulator_game

def main():
 p=argparse.ArgumentParser()
 for n in ("module-dir","armg-root","armg-weight","formal-seed-file","output-dir"): p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--armg-map-weight",type=Path);p.add_argument("--seed-count",type=int,default=40);p.add_argument("--rng-seed",type=int,default=20262001);p.add_argument("--mcts-sims",type=int,default=2000)
 a=p.parse_args();formal={int(x) for x in a.formal_seed_file.read_text().splitlines() if x.strip() and not x.startswith("#")}
 rng=random.Random(a.rng_seed);seeds=[];seen=set(formal)
 while len(seeds)<a.seed_count:
  x=rng.randint(1,10**9)
  if x not in seen:seen.add(x);seeds.append(x)
 sts=_load_sts(a.module_dir);a.output_dir.mkdir(parents=True,exist_ok=True);rows=[]
 for seed in seeds:
  pair={}
  for label,on in (("v1",False),("v2",True)):
   os.environ["STS1_TEACHER_V2_CONTEXTUAL_RERANK"]="1" if on else "0"
   armg=ArmGNoncombatPolicy(root=a.armg_root,weight_path=a.armg_weight,map_weight_path=a.armg_map_weight if a.armg_map_weight and a.armg_map_weight.is_file() else None)
   out=a.output_dir/label;out.mkdir(exist_ok=True)
   r=run_simulator_game(student=None,sts=sts,seed=seed,evidence_path=out/f"seed-{seed}.ndjson",armg_policy=armg,combat_mcts_sims=a.mcts_sims,training_seeds=seeds,collect_teacher=True)
   if r.get("result")!="PASS_SIMULATOR_COMPLETE_RUN":raise RuntimeError((label,seed,r))
   pair[label]=dict(r)
  rows.append({"seed":seed,**pair});print(f"AB {len(rows)}/{len(seeds)} v1={pair['v1'].get('outcome')} v2={pair['v2'].get('outcome')}",flush=True)
 def stat(k):
  rs=[x[k] for x in rows];w=sum(r.get("outcome")=="victory" for r in rs);flo=[r.get("final_floor") for r in rs if isinstance(r.get("final_floor"),(int,float))]
  return {"wins":w,"win_rate":w/len(rs),"mean_final_floor":mean(flo) if flo else None}
 v1=stat("v1");v2=stat("v2")
 report={"schema_version":"sts1-teacher-v2-paired-ab-v1","seed_count":len(seeds),"rng_seed":a.rng_seed,"mcts_sims":a.mcts_sims,"v1":v1,"v2":v2,
 "paired":{"v2_only_wins":sum(x["v2"].get("outcome")=="victory" and x["v1"].get("outcome")!="victory" for x in rows),"v1_only_wins":sum(x["v1"].get("outcome")=="victory" and x["v2"].get("outcome")!="victory" for x in rows)},"rows":rows}
 (a.output_dir/"teacher-v2-ab-summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print("TEACHER_V2_AB_RESULT",json.dumps({k:report[k] for k in ("seed_count","v1","v2","paired")},sort_keys=True))
if __name__=="__main__":main()
