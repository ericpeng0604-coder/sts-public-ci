#!/usr/bin/env python3
"""Trace Boss-50k-unrescued runs back to earlier non-combat build decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR=Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0,str(_SCRIPT_DIR))

import sts1_armg_ppo_rollout_v14 as rollout
from roguelike_ai.sts1_phase3.simulator import _load_sts, run_simulator_game

ACT_BY_BOSS={16:1,33:2,50:3}


def _read_seeds(path: Path) -> list[int]:
    rows=[
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows)!=50 or len(set(rows))!=50:
        raise RuntimeError("seed file must contain exactly 50 unique seeds")
    return rows


def _result_passes(row: dict[str,Any]) -> bool:
    return bool(row.get("passed_boss"))


def _second_teacher_row(
    *,
    seed:int,
    intervention:dict[str,Any],
    target_floor:int,
) -> dict[str,Any]:
    return {
        "schema_version":"sts1-armg-strategy-branch-dataset-v1",
        "type":"v20_early_two_step_boss_rescue_followup",
        "source":"sts1-ppo-v20-build-trajectory-diagnostic-v2",
        "seed":int(seed),
        "floor":int(intervention["floor"]),
        "act":int(intervention["act"]),
        "kind":str(intervention["kind"]),
        "obs":intervention["obs"],
        "descs":intervention["descs"],
        "current_armg_index":int(intervention["current_armg_index"]),
        "teacher_best_index":int(intervention["teacher_best_index"]),
        "target_probs":intervention["target_probs"],
        "priority":4.25,
        "teacher_margin":9.0,
        "confidence_weight":1.0,
        "teacher_consensus_fraction":1.0,
        "combat_policy":"mcts_2000",
        "confirmation_policy":"boss_mcts_10000_and_50000",
        "target_boss_floor":int(target_floor),
    }


def _teacher_row(
    *,
    seed:int,
    record:dict[str,Any],
    alternative:int,
    target_floor:int,
    boss10:dict[str,Any],
    boss50:dict[str,Any],
) -> dict[str,Any]:
    target=[0.0]*len(record["descs"])
    target[int(alternative)]=1.0
    return {
        "schema_version":"sts1-armg-strategy-branch-dataset-v1",
        "type":"v20_early_build_boss_rescue",
        "source":"sts1-ppo-v20-build-trajectory-diagnostic-v1",
        "seed":int(seed),
        "floor":int(record["floor"]),
        "act":int(record["act"]),
        "kind":str(record["kind"]),
        "obs":record["obs"],
        "descs":record["descs"],
        "current_armg_index":int(record["selected_index"]),
        "teacher_best_index":int(alternative),
        "target_probs":target,
        "priority":4.5,
        "teacher_margin":10.0,
        "confidence_weight":1.0,
        "teacher_consensus_fraction":1.0,
        "combat_policy":"mcts_2000",
        "confirmation_policy":"boss_mcts_10000_and_50000",
        "target_boss_floor":int(target_floor),
        "boss_10k":boss10,
        "boss_50k":boss50,
    }


def diagnose_seed(
    *,
    seed:int,
    module_dir:Path,
    armg_root:Path,
    weight:Path,
    heldout:list[int],
    max_states:int,
    max_alternatives:int,
    max_two_step_states:int,
    max_second_alternatives:int,
) -> dict[str,Any]:
    sts=_load_sts(module_dir)
    policy=rollout.SamplingArmG(
        root=armg_root,
        weight_path=weight,
        temperature=1.0,
        torch_seed=seed,
    )
    base=run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=None,
        armg_policy=policy,
        combat_mcts_sims=2000,
        heldout_seeds=heldout,
    )
    if base.get("result")!="PASS_SIMULATOR_COMPLETE_RUN":
        raise RuntimeError(f"seed {seed} base rerun incomplete: {base}")
    floor=int(base.get("final_floor") or base.get("max_floor") or 0)
    if str(base.get("outcome","")).lower()=="victory" or not rollout._near_boss_failure(floor):
        return {
            "seed":seed,
            "status":"NOT_REPRODUCED_AS_NEAR_BOSS_LOSS",
            "base":base,
            "captured_states":len(policy.conversion_states),
            "captured_state_summary":[
                {
                    "floor":int(row.get("floor",0) or 0),
                    "act":int(row.get("act",0) or 0),
                    "kind":str(row.get("kind","unknown")),
                    "choices":len(row.get("descs",[])),
                }
                for row in policy.conversion_states
            ],
            "attempted_states":0,
            "rescue":None,
        }

    target=rollout._boss_target_floor(floor)
    captured=list(policy.conversion_states)
    captured_summary=[
        {
            "floor":int(row.get("floor",0) or 0),
            "act":int(row.get("act",0) or 0),
            "kind":str(row.get("kind","unknown")),
            "choices":len(row.get("descs",[])),
        }
        for row in captured
    ]
    # Floor is the stable cross-version key. The simulator's act field can
    # transition around Boss boundaries, so filtering on exact act caused
    # valid reversible states to be discarded.
    records=[
        row
        for row in captured
        if int(row.get("floor",0) or 0) <= target
        and len(row.get("descs",[]))>=2
    ]
    records=list(reversed(records[-max_states:]))
    attempted=0
    if not records:
        return {
            "seed":seed,
            "status":"NO_CAPTURED_REVERSIBLE_STATE",
            "base":base,
            "target_boss_floor":target,
            "captured_state_count":len(policy.conversion_states),
            "attempted_states":0,
            "rescue":None,
        }

    for record in records:
        selected=int(record["selected_index"])
        baseline50=rollout._force_choice_and_finish(
            record,
            choice_index=selected,
            sts=sts,
            policy=policy,
            mcts_sims=2000,
            boss_mcts_sims=50000,
            target_floor=target,
        )
        if _result_passes(baseline50):
            continue

        alternatives=[i for i in range(len(record["descs"])) if i!=selected]
        alternatives.sort(
            key=lambda i: float(record["scores"][i]) if i < len(record["scores"]) else -1e30,
            reverse=True,
        )
        for alternative in alternatives[:max_alternatives]:
            attempted += 1
            boss10=rollout._force_choice_and_finish(
                record,
                choice_index=alternative,
                sts=sts,
                policy=policy,
                mcts_sims=2000,
                boss_mcts_sims=10000,
                target_floor=target,
            )
            if not _result_passes(boss10):
                continue
            boss50=rollout._force_choice_and_finish(
                record,
                choice_index=alternative,
                sts=sts,
                policy=policy,
                mcts_sims=2000,
                boss_mcts_sims=50000,
                target_floor=target,
            )
            if not _result_passes(boss50):
                continue

            return {
                "seed":seed,
                "status":"EARLY_BUILD_RESCUE_FOUND",
                "base":base,
                "target_boss_floor":target,
                "captured_states":len(captured),
                "captured_state_summary":captured_summary,
                "candidate_states":len(records),
                "captured_state_count":len(policy.conversion_states),
                "attempted_states":attempted,
                "rescue":{
                    "decision_floor":int(record["floor"]),
                    "decision_act":int(record["act"]),
                    "kind":str(record["kind"]),
                    "current_index":selected,
                    "alternative_index":int(alternative),
                    "boss_10k":boss10,
                    "boss_50k":boss50,
                    "teacher":_teacher_row(
                        seed=seed,
                        record=record,
                        alternative=alternative,
                        target_floor=target,
                        boss10=boss10,
                        boss50=boss50,
                    ),
                },
            }

    two_step_attempts=0
    for record in records[:max_two_step_states]:
        selected=int(record["selected_index"])
        baseline50=rollout._force_choice_and_finish(
            record,
            choice_index=selected,
            sts=sts,
            policy=policy,
            mcts_sims=2000,
            boss_mcts_sims=50000,
            target_floor=target,
        )
        if _result_passes(baseline50):
            continue

        alternatives=[i for i in range(len(record["descs"])) if i!=selected]
        alternatives.sort(
            key=lambda i: float(record["scores"][i]) if i < len(record["scores"]) else -1e30,
            reverse=True,
        )
        for alternative in alternatives[:max_alternatives]:
            for second_rank in range(max_second_alternatives):
                two_step_attempts += 1
                boss10=rollout._force_choice_and_finish(
                    record,
                    choice_index=alternative,
                    sts=sts,
                    policy=policy,
                    mcts_sims=2000,
                    boss_mcts_sims=10000,
                    target_floor=target,
                    second_alternative_rank=second_rank,
                )
                if not _result_passes(boss10):
                    continue
                boss50=rollout._force_choice_and_finish(
                    record,
                    choice_index=alternative,
                    sts=sts,
                    policy=policy,
                    mcts_sims=2000,
                    boss_mcts_sims=50000,
                    target_floor=target,
                    second_alternative_rank=second_rank,
                )
                if not _result_passes(boss50):
                    continue

                second=(
                    boss50.get("second_intervention")
                    or boss10.get("second_intervention")
                )
                first_teacher=_teacher_row(
                    seed=seed,
                    record=record,
                    alternative=alternative,
                    target_floor=target,
                    boss10=boss10,
                    boss50=boss50,
                )
                first_teacher["type"]="v20_early_two_step_boss_rescue"
                first_teacher["source"]="sts1-ppo-v20-build-trajectory-diagnostic-v2"
                teachers=[first_teacher]
                if second is not None:
                    teachers.append(
                        _second_teacher_row(
                            seed=seed,
                            intervention=second,
                            target_floor=target,
                        )
                    )
                return {
                    "seed":seed,
                    "status":"EARLY_TWO_STEP_BUILD_RESCUE_FOUND",
                    "base":base,
                    "target_boss_floor":target,
                    "captured_states":len(captured),
                    "captured_state_summary":captured_summary,
                    "candidate_states":len(records),
                    "captured_state_count":len(policy.conversion_states),
                    "attempted_states":attempted,
                    "two_step_attempts":two_step_attempts,
                    "rescue":{
                        "decision_floor":int(record["floor"]),
                        "decision_act":int(record["act"]),
                        "kind":str(record["kind"]),
                        "current_index":selected,
                        "alternative_index":int(alternative),
                        "second_alternative_rank":int(second_rank),
                        "boss_10k":boss10,
                        "boss_50k":boss50,
                        "teachers":teachers,
                    },
                }

    return {
        "seed":seed,
        "status":"NO_EARLY_BUILD_RESCUE",
        "base":base,
        "target_boss_floor":target,
        "captured_states":len(captured),
        "captured_state_summary":captured_summary,
        "candidate_states":len(records),
        "captured_state_count":len(policy.conversion_states),
        "attempted_states":attempted,
        "two_step_attempts":two_step_attempts,
        "rescue":None,
    }


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--weight",type=Path,required=True)
    p.add_argument("--seeds-file",type=Path,required=True)
    p.add_argument("--boss-summary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--teacher-output",type=Path,required=True)
    p.add_argument("--max-seeds",type=int,default=10)
    p.add_argument("--max-states",type=int,default=20)
    p.add_argument("--max-alternatives",type=int,default=2)
    p.add_argument("--max-two-step-states",type=int,default=8)
    p.add_argument("--max-second-alternatives",type=int,default=2)
    args=p.parse_args()
    if not 1<=args.max_seeds<=20:
        raise RuntimeError("max-seeds must be within 1..20")
    if not 1<=args.max_states<=20:
        raise RuntimeError("max-states must be within 1..20")
    if not 1<=args.max_alternatives<=4:
        raise RuntimeError("max-alternatives must be within 1..4")
    if not 1<=args.max_two_step_states<=12:
        raise RuntimeError("max-two-step-states must be within 1..12")
    if not 1<=args.max_second_alternatives<=3:
        raise RuntimeError("max-second-alternatives must be within 1..3")

    heldout=_read_seeds(args.seeds_file)
    summary=json.loads(args.boss_summary.read_text(encoding="utf-8"))
    candidates=[
        int(v)
        for v in summary["not_rescued_by_boss_50k"]["seeds"]
    ][:args.max_seeds]

    rows=[
        diagnose_seed(
            seed=seed,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            weight=args.weight,
            heldout=heldout,
            max_states=args.max_states,
            max_alternatives=args.max_alternatives,
            max_two_step_states=args.max_two_step_states,
            max_second_alternatives=args.max_second_alternatives,
        )
        for seed in candidates
    ]
    teacher=[]
    for row in rows:
        rescue=row.get("rescue") or {}
        if rescue.get("teacher"):
            teacher.append(rescue["teacher"])
        teacher.extend(rescue.get("teachers") or [])
    payload={
        "schema_version":"sts1-ppo-v20-build-trajectory-diagnostic-v2",
        "input_build_limited":len(summary["not_rescued_by_boss_50k"]["seeds"]),
        "diagnosed_seeds":len(rows),
        "early_build_rescue_seeds":sum(
            row["status"] in {"EARLY_BUILD_RESCUE_FOUND","EARLY_TWO_STEP_BUILD_RESCUE_FOUND"}
            for row in rows
        ),
        "single_step_rescues":sum(row["status"]=="EARLY_BUILD_RESCUE_FOUND" for row in rows),
        "two_step_rescues":sum(row["status"]=="EARLY_TWO_STEP_BUILD_RESCUE_FOUND" for row in rows),
        "teacher_examples":len(teacher),
        "no_early_rescue":sum(row["status"]=="NO_EARLY_BUILD_RESCUE" for row in rows),
        "capture_missing":sum(row["status"]=="NO_CAPTURED_REVERSIBLE_STATE" for row in rows),
        "not_reproduced":sum(row["status"]=="NOT_REPRODUCED_AS_NEAR_BOSS_LOSS" for row in rows),
        "captured_state_total":sum(int(row.get("captured_state_count",0) or 0) for row in rows),
        "rows":rows,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    args.teacher_output.parent.mkdir(parents=True,exist_ok=True)
    args.teacher_output.write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in teacher),
        encoding="utf-8",
    )
    print("PPO_V20_BUILD_TRAJECTORY_RESULT",json.dumps({
        "diagnosed_seeds":len(rows),
        "early_build_rescue_seeds":payload["early_build_rescue_seeds"],
        "single_step_rescues":payload["single_step_rescues"],
        "two_step_rescues":payload["two_step_rescues"],
        "teacher_examples":payload["teacher_examples"],
        "no_early_rescue":payload["no_early_rescue"],
        "capture_missing":payload["capture_missing"],
        "captured_state_total":payload["captured_state_total"],
        "not_reproduced":payload["not_reproduced"],
    },sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
