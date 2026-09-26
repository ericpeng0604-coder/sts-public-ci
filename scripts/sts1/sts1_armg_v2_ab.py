#!/usr/bin/env python3
"""Paired fresh-seed A/B: frozen ArmG v1 vs learned ArmG v2 candidate."""
import argparse,json,random
from pathlib import Path
from statistics import mean
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_load_sts,run_simulator_game
def main():
 p=argparse.ArgumentParser()
 for n in ("module-dir","armg-root","v1-weight","v2-weight","formal-seed-file","output-dir"): p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--seed-count",type=int,default=50);p.add_argument("--rng-seed",type=int,default=20263001);p.add_argument("--mcts-sims",type=int,default=2000)
 a=p.parse_args();formal={int(x) for x in a.formal_seed_file.read_text().splitlines() if x.strip() and not x.startswith("#")}
 rng=random.Random(a.rng_seed);seeds=[];seen=set(formal)
 while len(seeds)<a.seed_count:
  x=rng.randint(1,10**9)
  if x not in seen:seen.add(x);seeds.append(x)
 sts=_load_sts(a.module_dir);a.output_dir.mkdir(parents=True,exist_ok=True);rows=[]
 for seed in seeds:
  pair={}
  for label,w in (("v1",a.v1_weight),("v2",a.v2_weight)):
   armg=ArmGNoncombatPolicy(root=a.armg_root,weight_path=w)
   out=a.output_dir/label;out.mkdir(exist_ok=True)
   r=run_simulator_game(student=None,sts=sts,seed=seed,evidence_path=out/f"seed-{seed}.ndjson",armg_policy=armg,combat_mcts_sims=a.mcts_sims,training_seeds=seeds,collect_teacher=True)
   if r.get("result")!="PASS_SIMULATOR_COMPLETE_RUN": raise RuntimeError((label,seed,r))
   pair[label]=dict(r)
  rows.append({"seed":seed,**pair});print(f"ARMG_AB {len(rows)}/{len(seeds)} v1={pair['v1'].get('outcome')}:{pair['v1'].get('final_floor')} v2={pair['v2'].get('outcome')}:{pair['v2'].get('final_floor')}",flush=True)
 def stat(k):
  rs=[x[k] for x in rows];w=sum(r.get("outcome")=="victory" for r in rs);flo=[r.get("final_floor") for r in rs if isinstance(r.get("final_floor"),(int,float))]
  return {"wins":w,"win_rate":w/len(rs),"mean_final_floor":mean(flo) if flo else None}
 v1=stat("v1");v2=stat("v2")
 paired={"v2_only_wins":sum(x["v2"].get("outcome")=="victory" and x["v1"].get("outcome")!="victory" for x in rows),"v1_only_wins":sum(x["v1"].get("outcome")=="victory" and x["v2"].get("outcome")!="victory" for x in rows)}
 report={"schema_version":"sts1-armg-v2-paired-ab-v1","seed_source":"fresh_random_excluding_formal_50","seed_count":len(seeds),"rng_seed":a.rng_seed,"mcts_sims":a.mcts_sims,"v1":v1,"v2":v2,"paired":paired,"rows":rows}
 (a.output_dir/"armg-v2-ab-summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print("ARMG_V2_AB_RESULT",json.dumps({k:report[k] for k in ("seed_count","v1","v2","paired")},sort_keys=True))
if __name__=="__main__": main()
