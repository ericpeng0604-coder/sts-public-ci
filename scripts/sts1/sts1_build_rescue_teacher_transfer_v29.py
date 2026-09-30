#!/usr/bin/env python3
"""Audit whether verified New-Win teachers still win under the formal MCTS-2000 regime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts1_build_rescue_new_win_miner_v24 import (
    _force_spec,
    _isolated_variant,
    _read_seeds,
)


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--weight",type=Path,required=True)
    p.add_argument("--heldout-seeds-file",type=Path,required=True)
    p.add_argument("--source-result",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--timeout-seconds",type=int,default=180)
    a=p.parse_args()

    src=json.loads(a.source_result.read_text(encoding="utf-8"))
    seed=int(src["seed"])
    status=str(src.get("status",""))
    if status not in {"NEW_FULL_WIN_ONE_STEP","NEW_FULL_WIN_TWO_STEP"}:
        raise RuntimeError(f"source seed {seed} is not a verified full-win rescue: {status}")
    teachers=list(src.get("teachers") or [])
    rescue=dict(src.get("rescue") or {})
    expected=1 if status.endswith("ONE_STEP") else 2
    if len(teachers)!=expected:
        raise RuntimeError(f"source teacher count mismatch {len(teachers)} != {expected}")

    forced={}
    first=rescue.get("first")
    if not first:
        raise RuntimeError("source rescue missing first intervention")
    forced[int(first["branch_index"])]=_force_spec(
        {
            "kind":teachers[0]["kind"],
            "descs":teachers[0]["descs"],
        },
        int(teachers[0]["teacher_best_index"]),
    )
    if expected==2:
        second=rescue.get("second")
        if not second:
            raise RuntimeError("source rescue missing second intervention")
        forced[int(second["branch_index"])]=_force_spec(
            {
                "kind":teachers[1]["kind"],
                "descs":teachers[1]["descs"],
            },
            int(teachers[1]["teacher_best_index"]),
        )

    heldout=_read_seeds(a.heldout_seeds_file)
    if seed not in heldout:
        raise RuntimeError(f"seed {seed} is not in frozen Dev30")

    result, records, failures=_isolated_variant(
        seed=seed,
        module_dir=a.module_dir,
        armg_root=a.armg_root,
        weight=a.weight,
        heldout=heldout,
        boss_sims=None,
        forced=forced,
        timeout_seconds=a.timeout_seconds,
        retries=1,
    )
    if result is None:
        raise RuntimeError(f"formal MCTS-2000 forced replay failed: {failures}")

    formal_win=str(result.get("outcome","")).lower()=="victory"
    base=dict(src.get("base") or {})
    payload={
        "schema_version":"sts1-build-rescue-teacher-transfer-v29",
        "seed":seed,
        "source_status":status,
        "interventions":expected,
        "teacher_kinds":[str(x["kind"]) for x in teachers],
        "base_mcts2000_outcome":base.get("outcome"),
        "base_mcts2000_floor":base.get("final_floor"),
        "forced_mcts2000_outcome":result.get("outcome"),
        "forced_mcts2000_floor":result.get("final_floor"),
        "forced_mcts2000_hp":result.get("final_hp"),
        "formal_mcts2000_full_win":bool(formal_win),
        "source_boss10k_outcome":rescue.get("boss_10k",{}).get("outcome"),
        "source_boss50k_outcome":rescue.get("boss_50k",{}).get("outcome"),
        "native_failures":failures,
        "trace_records":len(records),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("V29_TEACHER_TRANSFER_RESULT",json.dumps({
        "seed":seed,
        "source_status":status,
        "interventions":expected,
        "base_floor":payload["base_mcts2000_floor"],
        "forced_floor":payload["forced_mcts2000_floor"],
        "formal_mcts2000_full_win":formal_win,
    },sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
