#!/usr/bin/env python3
"""Replay known winning seeds with exact ArmG-v3 training telemetry."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_load_sts,run_simulator_game

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--module-dir",type=Path,required=True); ap.add_argument("--armg-root",type=Path,required=True)
    ap.add_argument("--armg-weight",type=Path,required=True); ap.add_argument("--seed-file",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--mcts-sims",type=int,default=2000)
    a=ap.parse_args(); seeds=[int(x) for x in a.seed_file.read_text().split()]; sts=_load_sts(a.module_dir)
    armg=ArmGNoncombatPolicy(root=a.armg_root,weight_path=a.armg_weight); a.output_dir.mkdir(parents=True,exist_ok=True)
    kept=[]; failed=[]
    for i,seed in enumerate(seeds,1):
        p=a.output_dir/f"seed-{seed}.ndjson"
        r=dict(run_simulator_game(student=None,sts=sts,seed=seed,evidence_path=p,armg_policy=armg,combat_mcts_sims=a.mcts_sims,training_seeds=seeds,collect_teacher=True))
        if r.get("result")=="PASS_SIMULATOR_COMPLETE_RUN" and r.get("outcome")=="victory": kept.append(seed)
        else: failed.append({"seed":seed,"result":r.get("result"),"outcome":r.get("outcome"),"floor":r.get("final_floor")})
        print(f"REPLAY {i}/{len(seeds)} seed={seed} outcome={r.get('outcome')} floor={r.get('final_floor')}",flush=True)
    report={"requested":len(seeds),"reproduced_winners":len(kept),"winning_seeds":kept,"failed":failed}
    (a.output_dir/"replay-summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    if len(kept)<2: raise SystemExit("too few reproducible winners for train/validation")
if __name__=="__main__": raise SystemExit(main())
