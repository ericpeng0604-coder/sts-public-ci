#!/usr/bin/env python3
"""End-to-end public CPU smoke for STS1 Student v1 PPO data plumbing.

Uses a synthetic zero-weight v0 baseline only to prove:
sts_lightspeed + ArmG -> stochastic Student v1 -> public evidence -> PPO episode
-> hashed rollout shard -> checksum/stale-safe reload.

It does not claim gameplay quality and does not use the real frozen v0 weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from roguelike_ai.sts1_phase3.frozen_student import (
    ACTION_SCHEMA_VERSION,
    PUBLIC_STATE_SCHEMA_VERSION,
    FrozenStudentV0,
)
from roguelike_ai.sts1_phase3.ppo_rollout import (
    episode_from_simulator_evidence,
    read_rollout_shard,
    write_rollout_shard,
)
from roguelike_ai.sts1_phase3.protocol import FROZEN_SIMULATOR_SHA
from roguelike_ai.sts1_phase3.self_improve_loop import RolloutIdentity
from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)
from roguelike_ai.sts1_phase3.student_v1_ppo import (
    StudentV1Config,
    StudentV1PPO,
    file_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--armg-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="347001")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(347001)

    baseline = FrozenStudentV0(
        weights={},
        artifact_sha256=hashlib.sha256(b"synthetic-public-smoke-v0").hexdigest(),
    )
    policy = StudentV1PPO(
        baseline,
        config=StudentV1Config(
            state_dim=64,
            action_dim=32,
            hidden_dim=16,
            epochs=1,
            batch_size=8,
        ),
        generation=0,
    )
    checkpoint = args.output_dir / "synthetic-student-v1.pt"
    policy.save(checkpoint)
    policy_sha = file_sha256(checkpoint)

    sts = _load_sts(args.module_dir)
    armg = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.armg_weight)
    evidence = args.output_dir / "simulator-evidence.ndjson"
    summary = run_simulator_game(
        student=policy,
        sts=sts,
        seed=args.seed,
        evidence_path=evidence,
        armg_policy=armg,
        collect_ppo=True,
    )
    if summary.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
        raise RuntimeError(f"synthetic PPO simulator smoke did not finish: {summary}")

    episode = episode_from_simulator_evidence(
        evidence,
        episode_id=f"public-smoke-{args.seed}",
    )
    identity = RolloutIdentity(
        generation=0,
        policy_sha256=policy_sha,
        simulator_sha=FROZEN_SIMULATOR_SHA,
        public_state_schema=PUBLIC_STATE_SCHEMA_VERSION,
        action_schema=ACTION_SCHEMA_VERSION,
    )
    shard_dir = args.output_dir / "shard-000"
    payload = shard_dir / "rollout.jsonl"
    manifest = shard_dir / "manifest.json"
    write_rollout_shard(
        payload,
        manifest,
        identity=identity,
        shard_id="public-smoke-000",
        episodes=[episode],
    )
    loaded = read_rollout_shard(
        payload,
        manifest,
        expected_identity=identity,
    )
    if len(loaded) != 1 or len(loaded[0].transitions) < 1:
        raise RuntimeError("PPO rollout round-trip produced no transitions")

    result = {
        "result": "PASS_PUBLIC_STS1_PPO_END_TO_END_SMOKE",
        "seed": args.seed,
        "game_outcome": summary.get("outcome"),
        "final_floor": summary.get("final_floor"),
        "ppo_transitions": len(loaded[0].transitions),
        "armg_action_count": summary.get("armg_action_count"),
        "illegal_action_count": summary.get("illegal_action_count"),
        "timeout_count": summary.get("timeout_count"),
        "crash_count": summary.get("crash_count"),
        "policy_sha256": policy_sha,
        "rollout_identity_hash": identity.identity_hash,
    }
    (args.output_dir / "smoke-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
