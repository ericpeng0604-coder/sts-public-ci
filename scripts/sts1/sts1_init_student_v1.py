#!/usr/bin/env python3
"""Initialize generation-0 Student v1 from frozen Student v0."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from roguelike_ai.sts1_phase3.frozen_student import FrozenStudentV0
from roguelike_ai.sts1_phase3.student_v1_ppo import StudentV1Config, StudentV1PPO, file_sha256

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--baseline-v0",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--state-dim",type=int,default=512)
    p.add_argument("--action-dim",type=int,default=256)
    p.add_argument("--hidden-dim",type=int,default=128)
    a=p.parse_args()
    baseline=FrozenStudentV0.from_path(a.baseline_v0)
    policy=StudentV1PPO(baseline,config=StudentV1Config(state_dim=a.state_dim,action_dim=a.action_dim,hidden_dim=a.hidden_dim),generation=0)
    policy.save(a.output)
    report={"schema_version":"sts1-init-student-v1-report-v1","result":"PASS_STUDENT_V1_GENERATION0_INITIALIZED","generation":0,"baseline_artifact_sha256":baseline.artifact_sha256,"rollout_policy_sha256":file_sha256(a.output),"config_hash":policy.config.config_hash,"promotion_status":"NOT_A_PROMOTION"}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
