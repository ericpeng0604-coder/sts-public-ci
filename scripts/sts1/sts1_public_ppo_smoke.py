#!/usr/bin/env python3
"""Public CPU runner for STS1 Student v1 PPO.

Modes:
- smoke: synthetic end-to-end plumbing check.
- collect: fail-closed PPO rollout shard collection on training-only seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

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


def _read_seed_file(path: Path) -> tuple[int, ...]:
    values = tuple(
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values:
        raise ValueError(f"seed file is empty: {path}")
    if len(set(values)) != len(values):
        raise ValueError(f"seed file contains duplicates: {path}")
    if any(seed < 1 or seed > 10**9 for seed in values):
        raise ValueError(f"seed file contains value outside 1..1e9: {path}")
    return values


def _smoke(args: argparse.Namespace) -> int:
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


def _collect(args: argparse.Namespace) -> int:
    if args.baseline_v0 is None or args.source_v1 is None:
        raise ValueError("collect mode requires --baseline-v0 and --source-v1")
    if args.training_seed_file is None or args.eval_seed_file is None:
        raise ValueError("collect mode requires --training-seed-file and --eval-seed-file")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    training_seeds = _read_seed_file(args.training_seed_file)
    eval_seeds = _read_seed_file(args.eval_seed_file)
    overlap = sorted(set(training_seeds) & set(eval_seeds))
    if overlap:
        raise ValueError(f"training/eval seed overlap detected: {overlap[:8]}")

    shard_seeds = tuple(
        seed
        for index, seed in enumerate(training_seeds)
        if index % args.shard_count == args.shard_index
    )
    if not shard_seeds:
        raise ValueError("selected rollout shard has no training seeds")

    baseline = FrozenStudentV0.from_path(args.baseline_v0)
    policy = StudentV1PPO.load(args.source_v1, baseline, device="cpu")
    policy_sha = file_sha256(args.source_v1)
    identity = RolloutIdentity(
        generation=policy.generation,
        policy_sha256=policy_sha,
        simulator_sha=FROZEN_SIMULATOR_SHA,
        public_state_schema=PUBLIC_STATE_SCHEMA_VERSION,
        action_schema=ACTION_SCHEMA_VERSION,
    )

    sts = _load_sts(args.module_dir)
    armg = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.armg_weight)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    summaries = []
    blocked = []
    for seed in shard_seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        evidence = evidence_dir / f"seed-{seed}.ndjson"
        summary = run_simulator_game(
            student=policy,
            sts=sts,
            seed=seed,
            evidence_path=evidence,
            armg_policy=armg,
            training_seeds=training_seeds,
            collect_ppo=True,
        )
        summaries.append(summary)
        if summary.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
            blocked.append({
                "seed": seed,
                "result": summary.get("result"),
                "error": summary.get("error"),
                "illegal_action_count": summary.get("illegal_action_count"),
                "timeout_count": summary.get("timeout_count"),
                "crash_count": summary.get("crash_count"),
            })
            continue
        episodes.append(
            episode_from_simulator_evidence(
                evidence,
                episode_id=f"g{policy.generation}-s{seed}",
            )
        )

    report = {
        "schema_version": "sts1-ppo-rollout-worker-report-v1",
        "generation": policy.generation,
        "policy_sha256": policy_sha,
        "rollout_identity_hash": identity.identity_hash,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "training_seed_count": len(training_seeds),
        "shard_seed_count": len(shard_seeds),
        "complete_runs": len(episodes),
        "blocked_runs": len(blocked),
        "blocked": blocked,
        "summaries": summaries,
    }

    report_path = args.output_dir / "collector-report.json"
    if blocked:
        report["result"] = "BLOCKED_ROLLOUT_SHARD"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 3

    manifest = write_rollout_shard(
        args.output_dir / "rollout.jsonl",
        args.output_dir / "manifest.json",
        identity=identity,
        shard_id=f"g{policy.generation}-worker-{args.shard_index:03d}",
        episodes=episodes,
    )
    report.update({
        "result": "PASS_ROLLOUT_SHARD",
        "episodes": len(episodes),
        "transitions": sum(len(ep.transitions) for ep in episodes),
        "manifest": manifest,
    })
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "collect"), default="smoke")
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--armg-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="347001")
    parser.add_argument("--baseline-v0", type=Path)
    parser.add_argument("--source-v1", type=Path)
    parser.add_argument("--training-seed-file", type=Path)
    parser.add_argument("--eval-seed-file", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "collect":
        return _collect(args)
    return _smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
