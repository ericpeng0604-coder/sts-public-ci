#!/usr/bin/env python3
"""Freeze verified v2.4 New-Win teachers into an auditable exact Rescue Memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

WANTED={341684210,469494371,523373928,572162032,620908546}
VALID_STATUS={"NEW_FULL_WIN_ONE_STEP","NEW_FULL_WIN_TWO_STEP"}
SCHEMA="sts1-armg-strategy-branch-dataset-v1"


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    a=p.parse_args()

    results={}
    for path in sorted(a.source_dir.rglob("result-*.json")):
        row=json.loads(path.read_text(encoding="utf-8"))
        seed=int(row["seed"])
        if seed in WANTED:
            if seed in results:
                raise RuntimeError(f"duplicate source result for seed {seed}")
            results[seed]=row
    missing=sorted(WANTED-set(results))
    if missing:
        raise RuntimeError(f"missing verified New-Win source results: {missing}")

    memory=[]
    status_counts=Counter()
    kind_counts=Counter()
    for seed in sorted(WANTED):
        result=results[seed]
        status=str(result.get("status"))
        if status not in VALID_STATUS:
            raise RuntimeError(f"seed {seed} is not a verified full-win rescue: {status}")
        teachers=list(result.get("teachers") or [])
        expected=1 if status=="NEW_FULL_WIN_ONE_STEP" else 2
        if len(teachers)!=expected:
            raise RuntimeError(f"seed {seed} expected {expected} teachers, got {len(teachers)}")
        for t in teachers:
            if t.get("schema_version")!=SCHEMA:
                raise RuntimeError(f"teacher schema mismatch seed {seed}")
            if int(t["seed"])!=seed:
                raise RuntimeError(f"teacher seed mismatch {seed}")
            if t.get("type")!="v24_new_win_full_victory":
                raise RuntimeError(f"teacher type mismatch seed {seed}: {t.get('type')}")
            if float(t.get("confidence_weight",0))!=1.0:
                raise RuntimeError(f"teacher confidence mismatch seed {seed}")
            if float(t.get("teacher_consensus_fraction",0))!=1.0:
                raise RuntimeError(f"teacher consensus mismatch seed {seed}")
            descs=t["descs"]
            teacher=int(t["teacher_best_index"])
            current=int(t["current_armg_index"])
            if not descs or not 0<=teacher<len(descs) or not 0<=current<len(descs):
                raise RuntimeError(f"teacher candidate index invalid seed {seed}")
            if teacher==current:
                raise RuntimeError(f"teacher must disagree with G7 seed {seed}")
            memory.append(t)
            kind_counts[str(t["kind"])]+=1
        status_counts[status]+=1

    keys=set()
    for row in memory:
        key=(
            str(row["kind"]),
            json.dumps(row["obs"],separators=(",",":")),
            json.dumps(row["descs"],separators=(",",":")),
        )
        if key in keys:
            raise RuntimeError("duplicate exact Rescue Memory state")
        keys.add(key)

    if len(memory)!=7:
        raise RuntimeError(f"expected 7 frozen memory rows, got {len(memory)}")

    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(
        "".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in memory),
        encoding="utf-8",
    )
    summary={
        "schema_version":"sts1-rescue-memory-v30",
        "mode":"exact-match-only",
        "source":"v2.4 verified full-win mining",
        "teacher_examples":len(memory),
        "teacher_seeds":len(WANTED),
        "teacher_seed_ids":sorted(WANTED),
        "status_counts":dict(status_counts),
        "kind_counts":dict(kind_counts),
        "training_contaminated_seed_set":sorted(WANTED),
        "formal_generalization_starts_at":"Hidden50",
    }
    a.summary.parent.mkdir(parents=True,exist_ok=True)
    a.summary.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("V30_RESCUE_MEMORY_BUILD_PASS",json.dumps(summary,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
