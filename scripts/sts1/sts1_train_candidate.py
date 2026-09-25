#!/usr/bin/env python3
"""Train one Student v1 Candidate from verified rollout shards."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from roguelike_ai.sts1_phase3.frozen_student import ACTION_SCHEMA_VERSION, PUBLIC_STATE_SCHEMA_VERSION, FrozenStudentV0
from roguelike_ai.sts1_phase3.ppo_rollout import PPORolloutError, read_rollout_shard
from roguelike_ai.sts1_phase3.self_improve_loop import RolloutIdentity
from roguelike_ai.sts1_phase3.student_v1_ppo import StudentV1Error, StudentV1PPO, file_sha256, ppo_update

def discover(root:Path):
    pairs=[]
    for m in sorted(root.rglob("manifest.json")):
        p=m.with_name("rollout.jsonl")
        if p.is_file(): pairs.append((p,m))
    if not pairs: raise PPORolloutError("no rollout shards found")
    return pairs

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--baseline-v0",type=Path,required=True)
    p.add_argument("--source-v1",type=Path,required=True)
    p.add_argument("--rollout-root",type=Path,required=True)
    p.add_argument("--simulator-sha",required=True)
    p.add_argument("--generation",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    if a.output.resolve() in {a.baseline_v0.resolve(),a.source_v1.resolve()}: p.error("Candidate output may not overwrite baseline/source")
    try:
        baseline=FrozenStudentV0.from_path(a.baseline_v0)
        source_sha=file_sha256(a.source_v1)
        identity=RolloutIdentity(generation=a.generation,policy_sha256=source_sha,simulator_sha=a.simulator_sha,public_state_schema=PUBLIC_STATE_SCHEMA_VERSION,action_schema=ACTION_SCHEMA_VERSION)
        episodes=[]; shards=[]
        for payload,manifest in discover(a.rollout_root):
            eps=read_rollout_shard(payload,manifest,expected_identity=identity)
            episodes.extend(eps); shards.append({"payload":str(payload),"episodes":len(eps),"transitions":sum(len(x.transitions) for x in eps)})
        policy=StudentV1PPO.load(a.source_v1,baseline,device=a.device)
        if policy.generation!=a.generation: raise StudentV1Error("source rollout-policy generation mismatch")
        policy.generation=a.generation+1
        stats=ppo_update(policy,episodes)
        policy.save(a.output)
        report={"schema_version":"sts1-train-candidate-report-v1","result":"PASS_CANDIDATE_TRAINED_NOT_PROMOTED","source_generation":a.generation,"candidate_generation":policy.generation,"source_policy_sha256":source_sha,"rollout_identity_hash":identity.identity_hash,"rollout_shards":shards,"episodes":len(episodes),"transitions":sum(len(x.transitions) for x in episodes),"candidate_sha256":file_sha256(a.output),"ppo":stats,"promotion_status":"NOT_EVALUATED"}
        a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,sort_keys=True)); return 0
    except (OSError,ValueError,PPORolloutError,StudentV1Error) as exc:
        err={"result":"BLOCKED_CANDIDATE_TRAINING","error":str(exc),"promotion_status":"NOT_EVALUATED"}
        a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(err,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(err,sort_keys=True),file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
