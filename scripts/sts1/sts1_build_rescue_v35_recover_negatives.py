#!/usr/bin/env python3
"""Recover verified one-step failed alternatives from v3.4 miner checkpoints."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import sts1_build_rescue_v34_aggregate as v34


def _fingerprint(record: dict[str, Any]) -> str:
    return json.dumps(
        {"kind":record["kind"],"obs":record["obs"],"descs":record["descs"]},
        sort_keys=True,separators=(",",":"),
    )


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--results-dir",type=Path,required=True)
    p.add_argument("--positive-teachers",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-per-seed",type=int,default=8)
    a=p.parse_args()

    positives=[
        json.loads(x) for x in a.positive_teachers.read_text().splitlines() if x.strip()
    ]
    positive_states={_fingerprint(x) for x in positives}
    rows_by_seed: dict[int,list[dict[str,Any]]]=defaultdict(list)

    for result_path in sorted(a.results_dir.rglob("result-*.json")):
        result=json.loads(result_path.read_text())
        seed=int(result["seed"])
        target=int(result.get("target_boss_floor",0) or 0)
        if target not in {16,33,50}:
            continue
        checkpoints=list(result_path.parent.glob(f"checkpoint-{seed}.json"))
        if len(checkpoints)!=1:
            continue
        cp=json.loads(checkpoints[0].read_text())
        for item in (cp.get("completed_variants") or {}).values():
            sim=item.get("result")
            records=item.get("records") or []
            if not v34._safe_sim(sim,seed) or v34._passes_target(sim,target):
                continue
            forced=[r for r in records if "forced_index" in r]
            # A two-step failed combination does not prove either action is bad alone.
            if len(forced)!=1:
                continue
            r=forced[0]
            current=int(r["unforced_index"])
            rejected=int(r["forced_index"])
            if current==rejected:
                continue
            state={
                "kind":str(r["kind"]),"obs":r["obs"],"descs":r["descs"]
            }
            if _fingerprint(state) in positive_states:
                continue
            rows_by_seed[seed].append({
                "schema_version":"sts1-armg-negative-branch-v1",
                "type":"v35_failed_rescue_alternative",
                "source":"sts1-build-rescue-new-win-miner-v35",
                "seed":seed,
                "floor":int(r.get("floor",0) or 0),
                "act":int(r.get("act",0) or 0),
                "kind":str(r["kind"]),
                "obs":r["obs"],
                "descs":r["descs"],
                "current_armg_index":current,
                "rejected_index":rejected,
                "target_boss_floor":target,
                "failure_stage":"recovered_single_step_failed",
                "combat_policy":"mcts_2000",
                "confirmation_policy":"safe_completed_counterfactual",
                "boss_10k":sim,
                "boss_50k":None,
            })

    selected=[]
    for seed,rows in sorted(rows_by_seed.items()):
        uniq={}
        for r in rows:
            key=(_fingerprint(r),int(r["current_armg_index"]),int(r["rejected_index"]))
            uniq[key]=r
        vals=sorted(
            uniq.values(),
            key=lambda r:(-int(r["floor"]),str(r["kind"]),int(r["rejected_index"])),
        )[:a.max_per_seed]
        selected.extend(vals)

    if not selected:
        raise RuntimeError("no safe single-step negative examples recovered")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(
        "".join(json.dumps(x,sort_keys=True)+"\n" for x in selected),
        encoding="utf-8",
    )
    print("V35_RECOVER_NEGATIVES_PASS",json.dumps({
        "examples":len(selected),
        "seeds":len({int(x["seed"]) for x in selected}),
        "kinds":sorted({str(x["kind"]) for x in selected}),
    },sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
