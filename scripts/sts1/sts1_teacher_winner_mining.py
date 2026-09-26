#!/usr/bin/env python3
"""Mine winning and near-winning Teacher trajectories on fresh A0 seeds."""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
from statistics import mean
from typing import Any
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, _load_sts, run_simulator_game

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True); p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--armg-weight",type=Path,required=True); p.add_argument("--armg-map-weight",type=Path)
    p.add_argument("--formal-seed-file",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seed-count",type=int,default=100); p.add_argument("--rng-seed",type=int,default=20261001)
    p.add_argument("--mcts-sims",type=int,default=2000); p.add_argument("--shard-index",type=int,default=0); p.add_argument("--shard-count",type=int,default=1)
    a=p.parse_args()
    formal={int(x.strip()) for x in a.formal_seed_file.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")}
    rng=random.Random(a.rng_seed); seeds=[]; seen=set(formal)
    while len(seeds)<a.seed_count:
        s=rng.randint(1,10**9)
        if s not in seen: seen.add(s); seeds.append(s)
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count: raise ValueError("invalid shard")
    all_seeds=list(seeds); seeds=all_seeds[a.shard_index::a.shard_count]
    if not seeds: raise ValueError("empty shard")
    sts=_load_sts(a.module_dir)
    mw=a.armg_map_weight if a.armg_map_weight and a.armg_map_weight.is_file() else None
    armg=ArmGNoncombatPolicy(root=a.armg_root,weight_path=a.armg_weight,map_weight_path=mw)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; winners=[]; near=[]
    for i,s in enumerate(seeds,1):
        ev=a.output_dir/f"seed-{s}.ndjson"
        row=run_simulator_game(student=None,sts=sts,seed=s,evidence_path=ev,armg_policy=armg,
            combat_mcts_sims=a.mcts_sims,training_seeds=seeds,collect_teacher=True)
        if row.get("result")!="PASS_SIMULATOR_COMPLETE_RUN": raise RuntimeError(f"blocked seed={s}: {row}")
        row=dict(row); rows.append(row)
        if row.get("outcome")=="victory": winners.append({"seed":s,"summary":row,"evidence":ev.name})
        elif int(row.get("final_floor") or 0)>=45: near.append({"seed":s,"summary":row,"evidence":ev.name})
        print(f"WINNER_MINING {i}/{len(seeds)} wins={len(winners)} near={len(near)} floor={row.get('final_floor')}",flush=True)
    floors=[float(r["final_floor"]) for r in rows if isinstance(r.get("final_floor"),(int,float))]
    report={"schema_version":"sts1-teacher-winner-mining-v1","seed_source":"fresh_random_excluding_formal_50",
      "seed_count":len(seeds),"requested_seed_count":a.seed_count,"shard_index":a.shard_index,"shard_count":a.shard_count,"mcts_sims":a.mcts_sims,"victories":len(winners),"win_rate":len(winners)/len(seeds),
      "near_wins":len(near),"mean_final_floor":mean(floors) if floors else None,
      "winning_seeds":[x["seed"] for x in winners],"near_win_seeds":[x["seed"] for x in near],
      "winners":winners,"near_wins_detail":near}
    (a.output_dir/"winner-mining-summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print("WINNER_MINING_RESULT",json.dumps({k:report[k] for k in ("seed_count","victories","win_rate","near_wins","mean_final_floor","winning_seeds")},sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
