#!/usr/bin/env python3
"""Continuous STS1 self-improvement loop v2.

Per round:
1. collect fresh ArmG + MCTS-2000 Teacher games;
2. quality-weight and distill every valid Teacher decision;
3. compare against the current learner on fresh shadow eval seeds;
4. keep only an improving learner;
5. enable PPO only after Teacher agreement is high enough;
6. rollback PPO if it regresses;
7. periodically run the unchanged strict Champion 30/50 gates.

The production/real-game Champion is never replaced here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

import torch

from roguelike_ai.sts1_phase3.champion_gate import (
    FAST_GATE_POLICY,
    FORMAL_GATE_POLICY,
    evaluate_fixed_seed_gate,
)
from roguelike_ai.sts1_phase3.frozen_student import FrozenStudentV0
from roguelike_ai.sts1_phase3.hybrid_model import (
    DEFAULT_HYBRID_MCTS_BUDGETS,
    HYBRID_RUNTIME_ID,
    write_hybrid_model_manifest,
)
from roguelike_ai.sts1_phase3.ppo_rollout import episode_from_simulator_evidence
from roguelike_ai.sts1_phase3.self_improve_v2 import (
    LearnerGatePolicy,
    LoopV2State,
    evaluate_learner_gate,
    ppo_allowed,
)
from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)
from roguelike_ai.sts1_phase3.student_v1_ppo import (
    StudentV1Config,
    StudentV1PPO,
    file_sha256,
    ppo_update,
)
from roguelike_ai.sts1_phase3.teacher_distill import (
    distill_mcts_teacher,
    read_teacher_evidence,
)


def _read_seeds(path: Path) -> tuple[int, ...]:
    seeds = tuple(
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise RuntimeError(f"invalid seed file: {path}")
    return seeds


def _fresh_seeds(
    *,
    count: int,
    rng_seed: int,
    forbidden: set[int],
) -> tuple[int, ...]:
    rng = random.Random(rng_seed)
    result: list[int] = []
    seen = set(forbidden)
    while len(result) < count:
        value = rng.randint(1, 10**9)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _aggregate(label: str, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "sts1-self-improve-v2-eval-v1",
        "policy": label,
        "seeds": len(runs),
        "runs": list(runs),
    }


def _evaluate(
    *,
    policy: StudentV1PPO,
    seeds: Sequence[int],
    heldout_seeds: Sequence[int],
    sts: Any,
    armg: ArmGNoncombatPolicy,
) -> dict[str, Any]:
    runs = []
    for seed in seeds:
        summary = run_simulator_game(
            student=policy,
            sts=sts,
            seed=seed,
            armg_policy=armg,
            heldout_seeds=heldout_seeds,
            hybrid_mcts_budgets=DEFAULT_HYBRID_MCTS_BUDGETS,
            collect_ppo=False,
        )
        if summary.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
            raise RuntimeError(f"blocked eval seed={seed}: {summary}")
        runs.append(summary)
    return _aggregate("student_v1", runs)


def _collect_teacher(
    *,
    student: StudentV1PPO,
    seeds: Sequence[int],
    sts: Any,
    armg: ArmGNoncombatPolicy,
    output_dir: Path,
    mcts_sims: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples: list[Any] = []
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        evidence = output_dir / f"teacher-{seed}.ndjson"
        summary = run_simulator_game(
            student=student,
            sts=sts,
            seed=seed,
            evidence_path=evidence,
            armg_policy=armg,
            combat_mcts_sims=mcts_sims,
            training_seeds=seeds,
            collect_teacher=True,
        )
        if summary.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
            raise RuntimeError(f"blocked Teacher seed={seed}: {summary}")
        summaries.append(dict(summary))
        examples.extend(read_teacher_evidence(evidence))
    return examples, summaries


def _collect_ppo(
    *,
    policy: StudentV1PPO,
    seeds: Sequence[int],
    sts: Any,
    armg: ArmGNoncombatPolicy,
    output_dir: Path,
) -> tuple[list[Any], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        evidence = output_dir / f"ppo-{seed}.ndjson"
        summary = run_simulator_game(
            student=policy,
            sts=sts,
            seed=seed,
            evidence_path=evidence,
            armg_policy=armg,
            training_seeds=seeds,
            collect_ppo=True,
        )
        if summary.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
            raise RuntimeError(f"blocked PPO seed={seed}: {summary}")
        summaries.append(dict(summary))
        episodes.append(
            episode_from_simulator_evidence(
                evidence,
                episode_id=f"v2-g{policy.generation}-s{seed}",
            )
        )
    return episodes, summaries


def _copy_checkpoint(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if file_sha256(source) != file_sha256(target):
        raise RuntimeError("checkpoint copy checksum mismatch")


def _init_or_resume(
    *,
    baseline: FrozenStudentV0,
    state_dir: Path,
    config: StudentV1Config,
) -> LoopV2State:
    state_path = state_dir / "loop-state.json"
    learner_path = state_dir / "learner.pt"
    champion_path = state_dir / "offline-champion.pt"
    if state_path.exists():
        state = LoopV2State.from_path(state_path)
        if not learner_path.is_file() or not champion_path.is_file():
            raise RuntimeError("resume state is missing learner/offline Champion checkpoint")
        StudentV1PPO.load(learner_path, baseline, device="cpu")
        StudentV1PPO.load(champion_path, baseline, device="cpu")
        return state

    state_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260925)
    policy = StudentV1PPO(
        baseline,
        config=config,
        generation=0,
        device="cpu",
    )
    policy.save(learner_path)
    _copy_checkpoint(learner_path, champion_path)
    state = LoopV2State()
    state.write(state_path)
    return state


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-v0", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--armg-weight", type=Path, required=True)
    parser.add_argument("--armg-map-weight", type=Path)
    parser.add_argument("--eval-seed-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--teacher-seeds-per-round", type=int, default=32)
    parser.add_argument("--ppo-seeds-per-round", type=int, default=64)
    parser.add_argument("--shadow-eval-seeds", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=20260930)
    parser.add_argument("--teacher-mcts-sims", type=int, default=2000)
    parser.add_argument("--distill-epochs", type=int, default=4)
    parser.add_argument("--distill-batch-size", type=int, default=64)
    parser.add_argument("--teacher-top1-before-ppo", type=float, default=0.55)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--baseline-anchor-coef", type=float, default=0.05)
    parser.add_argument("--learner-floor-delta", type=float, default=0.5)
    parser.add_argument("--champion-check-interval", type=int, default=3)
    parser.add_argument("--max-stagnation", type=int, default=5)\n    parser.add_argument("--reset-stagnation", action="store_true")
    args = parser.parse_args()

    if args.rounds < 1 or args.teacher_seeds_per_round < 1 or args.shadow_eval_seeds < 2:
        raise RuntimeError("round/Teacher/shadow counts must be positive")
    if args.ppo_seeds_per_round < 1:
        raise RuntimeError("PPO seed count must be positive")
    if args.champion_check_interval < 1 or args.max_stagnation < 1:
        raise RuntimeError("interval/stagnation bounds must be positive")

    formal_eval = _read_seeds(args.eval_seed_file)
    if len(formal_eval) != 50:
        raise RuntimeError("formal eval seed file must contain exactly 50 seeds")

    baseline = FrozenStudentV0.from_path(args.baseline_v0)
    config = StudentV1Config(
        learning_rate=args.learning_rate,
        baseline_anchor_coef=args.baseline_anchor_coef,
    )
    state = _init_or_resume(
        baseline=baseline,
        state_dir=args.state_dir,
        config=config,
    )

    sts = _load_sts(args.module_dir)
    armg_map_weight = (
        args.armg_map_weight
        if args.armg_map_weight is not None and args.armg_map_weight.is_file()
        else None
    )
    armg = ArmGNoncombatPolicy(
        root=args.armg_root,
        weight_path=args.armg_weight,
        map_weight_path=armg_map_weight,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    learner_path = args.state_dir / "learner.pt"
    champion_path = args.state_dir / "offline-champion.pt"
    learner_manifest_path = args.state_dir / "model-manifest.json"
    champion_manifest_path = args.state_dir / "offline-champion-manifest.json"
    write_hybrid_model_manifest(
        learner_path,
        learner_manifest_path,
        armg_map_weight=armg_map_weight,
        loop_round=state.round_index,
    )
    write_hybrid_model_manifest(
        champion_path,
        champion_manifest_path,
        armg_map_weight=armg_map_weight,
        loop_round=state.round_index,
    )
    run_reports: list[dict[str, Any]] = []

    for _local_round in range(args.rounds):
        if state.stagnation_count >= args.max_stagnation:
            break

        round_no = state.round_index + 1
        round_dir = args.output_dir / f"round-{round_no:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        forbidden = (
            set(formal_eval)
            | set(state.used_teacher_seeds)
            | set(state.used_ppo_seeds)
        )
        teacher_seeds = _fresh_seeds(
            count=args.teacher_seeds_per_round,
            rng_seed=args.rng_seed + round_no * 1000 + 1,
            forbidden=forbidden,
        )
        forbidden.update(teacher_seeds)
        shadow_seeds = _fresh_seeds(
            count=args.shadow_eval_seeds,
            rng_seed=args.rng_seed + round_no * 1000 + 2,
            forbidden=forbidden,
        )

        learner_before = StudentV1PPO.load(learner_path, baseline, device="cpu")
        before_sha = file_sha256(learner_path)

        examples, teacher_summaries = _collect_teacher(
            student=learner_before,
            seeds=teacher_seeds,
            sts=sts,
            armg=armg,
            output_dir=round_dir / "teacher",
            mcts_sims=args.teacher_mcts_sims,
        )

        distill_path = round_dir / "distill-candidate.pt"
        _copy_checkpoint(learner_path, distill_path)
        distill_candidate = StudentV1PPO.load(distill_path, baseline, device="cpu")
        distill_stats = distill_mcts_teacher(
            distill_candidate,
            examples,
            epochs=args.distill_epochs,
            batch_size=args.distill_batch_size,
            seed=args.rng_seed + round_no,
        )
        distill_candidate.save(distill_path)

        current_eval = _evaluate(
            policy=learner_before,
            seeds=shadow_seeds,
            heldout_seeds=shadow_seeds,
            sts=sts,
            armg=armg,
        )
        distill_eval = _evaluate(
            policy=distill_candidate,
            seeds=shadow_seeds,
            heldout_seeds=shadow_seeds,
            sts=sts,
            armg=armg,
        )
        distill_gate = evaluate_learner_gate(
            current_eval,
            distill_eval,
            policy=LearnerGatePolicy(args.learner_floor_delta),
        )

        learner_changed = False
        accepted_stages: list[str] = []
        if distill_gate["status"] == "ACCEPT":
            _copy_checkpoint(distill_path, learner_path)
            learner_changed = True
            accepted_stages.append("distill")

        ppo_readiness = ppo_allowed(
            distill_stats,
            min_teacher_top1_accuracy=args.teacher_top1_before_ppo,
        )
        ppo_report: dict[str, Any] = {
            "status": "SKIPPED",
            "reason": ppo_readiness["reason"],
        }
        ppo_seeds: tuple[int, ...] = ()

        if learner_changed and ppo_readiness["allowed"]:
            forbidden.update(shadow_seeds)
            ppo_seeds = _fresh_seeds(
                count=args.ppo_seeds_per_round,
                rng_seed=args.rng_seed + round_no * 1000 + 3,
                forbidden=forbidden,
            )
            pre_ppo_path = round_dir / "pre-ppo.pt"
            _copy_checkpoint(learner_path, pre_ppo_path)
            pre_ppo = StudentV1PPO.load(pre_ppo_path, baseline, device="cpu")
            episodes, ppo_summaries = _collect_ppo(
                policy=pre_ppo,
                seeds=ppo_seeds,
                sts=sts,
                armg=armg,
                output_dir=round_dir / "ppo",
            )
            ppo_candidate_path = round_dir / "ppo-candidate.pt"
            _copy_checkpoint(pre_ppo_path, ppo_candidate_path)
            ppo_candidate = StudentV1PPO.load(
                ppo_candidate_path,
                baseline,
                device="cpu",
            )
            ppo_stats = ppo_update(ppo_candidate, episodes)
            ppo_candidate.save(ppo_candidate_path)

            pre_ppo_eval = _evaluate(
                policy=pre_ppo,
                seeds=shadow_seeds,
                heldout_seeds=shadow_seeds,
                sts=sts,
                armg=armg,
            )
            post_ppo_eval = _evaluate(
                policy=ppo_candidate,
                seeds=shadow_seeds,
                heldout_seeds=shadow_seeds,
                sts=sts,
                armg=armg,
            )
            ppo_gate = evaluate_learner_gate(
                pre_ppo_eval,
                post_ppo_eval,
                policy=LearnerGatePolicy(args.learner_floor_delta),
            )
            ppo_report = {
                "status": ppo_gate["status"],
                "gate": ppo_gate,
                "stats": ppo_stats,
                "episodes": len(episodes),
                "transitions": sum(len(ep.transitions) for ep in episodes),
                "summaries": ppo_summaries,
            }
            if ppo_gate["status"] == "ACCEPT":
                _copy_checkpoint(ppo_candidate_path, learner_path)
                accepted_stages.append("ppo")

        accepted_increment = 1 if learner_changed else 0
        champion_report: dict[str, Any] = {
            "status": "SKIPPED",
            "reason": "champion_check_interval_not_reached",
        }
        next_accepted_total = state.accepted_rounds + accepted_increment

        if (
            learner_changed
            and next_accepted_total % args.champion_check_interval == 0
        ):
            champion = StudentV1PPO.load(champion_path, baseline, device="cpu")
            learner_now = StudentV1PPO.load(learner_path, baseline, device="cpu")
            champion30 = _evaluate(
                policy=champion,
                seeds=formal_eval[:30],
                heldout_seeds=formal_eval,
                sts=sts,
                armg=armg,
            )
            learner30 = _evaluate(
                policy=learner_now,
                seeds=formal_eval[:30],
                heldout_seeds=formal_eval,
                sts=sts,
                armg=armg,
            )
            gate30 = evaluate_fixed_seed_gate(
                champion30,
                learner30,
                policy=FAST_GATE_POLICY,
            )
            if gate30["status"] == "PASS":
                champion50 = _evaluate(
                    policy=champion,
                    seeds=formal_eval,
                    heldout_seeds=formal_eval,
                    sts=sts,
                    armg=armg,
                )
                learner50 = _evaluate(
                    policy=learner_now,
                    seeds=formal_eval,
                    heldout_seeds=formal_eval,
                    sts=sts,
                    armg=armg,
                )
                gate50 = evaluate_fixed_seed_gate(
                    champion50,
                    learner50,
                    policy=FORMAL_GATE_POLICY,
                )
            else:
                gate50 = {
                    "schema_version": "sts1-champion-gate-v1",
                    "gate": "fixed-seed-50",
                    "status": "SKIPPED",
                    "reasons": ["fast_30_seed_gate_did_not_pass"],
                }

            promoted_offline = gate30["status"] == "PASS" and gate50["status"] == "PASS"
            if promoted_offline:
                _copy_checkpoint(learner_path, champion_path)
            champion_report = {
                "status": "OFFLINE_PROMOTED_AWAIT_REAL_GAME" if promoted_offline else "HOLD",
                "gate_30": gate30,
                "gate_50": gate50,
                "production_champion_replaced": False,
            }

        if learner_changed:
            state = LoopV2State(
                round_index=round_no,
                accepted_rounds=state.accepted_rounds + 1,
                rejected_rounds=state.rejected_rounds,
                stagnation_count=0,
                used_teacher_seeds=state.used_teacher_seeds + teacher_seeds,
                used_ppo_seeds=state.used_ppo_seeds + ppo_seeds,
            )
        else:
            state = LoopV2State(
                round_index=round_no,
                accepted_rounds=state.accepted_rounds,
                rejected_rounds=state.rejected_rounds + 1,
                stagnation_count=state.stagnation_count + 1,
                used_teacher_seeds=state.used_teacher_seeds + teacher_seeds,
                used_ppo_seeds=state.used_ppo_seeds + ppo_seeds,
            )
        state.write(args.state_dir / "loop-state.json")
        learner_manifest = write_hybrid_model_manifest(
            learner_path,
            learner_manifest_path,
            loop_round=state.round_index,
        )
        champion_manifest = write_hybrid_model_manifest(
            champion_path,
            champion_manifest_path,
            loop_round=state.round_index,
        )

        report = {
            "schema_version": "sts1-self-improve-round-v2",
            "round": round_no,
            "base_policy": "armg+mcts2000-base-v1",
            "armg_map_weight_sha256": (
                file_sha256(armg_map_weight)
                if armg_map_weight is not None
                else file_sha256(args.armg_weight)
            ),
            "hybrid_runtime_id": HYBRID_RUNTIME_ID,
            "hybrid_mcts_budgets": list(DEFAULT_HYBRID_MCTS_BUDGETS),
            "learner_model_manifest": learner_manifest,
            "offline_champion_model_manifest": champion_manifest,
            "learner_sha_before": before_sha,
            "learner_sha_after": file_sha256(learner_path),
            "offline_champion_sha": file_sha256(champion_path),
            "teacher_seed_count": len(teacher_seeds),
            "teacher_examples": len(examples),
            "teacher_victories": sum(int(x.get("outcome") == "victory") for x in teacher_summaries),
            "teacher_mean_final_floor": sum(float(x.get("final_floor", 0)) for x in teacher_summaries) / len(teacher_summaries),
            "distillation": distill_stats,
            "distill_gate": distill_gate,
            "ppo_readiness": ppo_readiness,
            "ppo": ppo_report,
            "accepted_stages": accepted_stages,
            "learner_changed": learner_changed,
            "champion": champion_report,
            "loop_state": state.to_dict(),
        }
        _write_json(round_dir / "round-report.json", report)
        print("SELF_IMPROVE_ROUND", json.dumps(report, ensure_ascii=False, sort_keys=True))
        run_reports.append(report)

    summary = {
        "schema_version": "sts1-self-improve-run-v2",
        "base_policy": "armg+mcts2000-base-v1",
        "armg_map_weight_sha256": (
            file_sha256(armg_map_weight)
            if armg_map_weight is not None
            else file_sha256(args.armg_weight)
        ),
        "hybrid_runtime_id": HYBRID_RUNTIME_ID,
        "hybrid_mcts_budgets": list(DEFAULT_HYBRID_MCTS_BUDGETS),
        "rounds_executed": len(run_reports),
        "learner_sha256": file_sha256(learner_path),
        "offline_champion_sha256": file_sha256(champion_path),
        "state": state.to_dict(),
        "paused_for_stagnation": state.stagnation_count >= args.max_stagnation,
        "production_champion_replaced": False,
        "real_game_gate_required_for_production": True,
    }
    _write_json(args.output_dir / "loop-summary.json", summary)
    print("SELF_IMPROVE_SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
