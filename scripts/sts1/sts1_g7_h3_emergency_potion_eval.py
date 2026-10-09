"""Run the pre-registered H3 train pool as a private paired G7 counterfactual."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sts1"))

if __package__:
    from . import sts1_g7_h2_elite_route_eval as shared
else:  # direct script invocation from scripts/sts1
    import sts1_g7_h2_elite_route_eval as shared  # type: ignore[no-redef]

from roguelike_ai.sts1_phase3 import simulator as simulator_module  # noqa: E402
from roguelike_ai.sts1_phase3.simulator import (  # noqa: E402
    ArmGNoncombatPolicy,
    _load_sts,
    _public_action,
    run_simulator_game,
)
from sts1_g7_seed_ledger import MAX_SEED, sha256_json, validate_inventory  # noqa: E402


POOL_ID = "round-003-20261009-train_hypothesis_3"
POOL_FILE = "train_hypothesis_3.json"
EXPECTED_POOL_SHA256 = "3f3d5f8f0e9fd762529bddc7577bd02ea29331e66d72cbdf9b36001d48669f52"
G7_SHA256 = "8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0"
SIMULATOR_COMMIT = "7476a81954020087da31d41d16fddf475746ec2d"
SIMULATOR_BINDING_SHA256 = "dea5e3b88097c7e7b7ffb74f03c227b7244a548b07fc5344c6b195d737c65512"
MCTS_SIMS = 2000
HP_RATIO_THRESHOLD = 0.5


class EvaluationIntegrityError(RuntimeError):
    """The H3 run did not satisfy its frozen manifest or safety contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pool(
    pool_path: Path,
    pools_dir: Path,
    inventory_path: Path,
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    if pool_path.name != POOL_FILE:
        raise EvaluationIntegrityError("only the pre-registered H3 train pool is accepted")
    pool = shared._read_json(pool_path)
    if pool.get("pool_id") != POOL_ID or pool.get("purpose") != "train":
        raise EvaluationIntegrityError("H3 pool identity or purpose mismatch")
    if pool.get("status") != "GENERATED_NOT_RUN":
        raise EvaluationIntegrityError("H3 pool is not marked unused")
    recorded_hash = pool.get("manifest_sha256")
    payload = {key: value for key, value in pool.items() if key != "manifest_sha256"}
    if recorded_hash != EXPECTED_POOL_SHA256 or sha256_json(payload) != recorded_hash:
        raise EvaluationIntegrityError("H3 pool manifest hash mismatch")
    seed_values = pool.get("seed_ids")
    if not isinstance(seed_values, list) or len(seed_values) != 10:
        raise EvaluationIntegrityError("H3 train pool must contain exactly ten seeds")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or not 1 <= seed <= MAX_SEED
        for seed in seed_values
    ):
        raise EvaluationIntegrityError("H3 pool contains an invalid simulator seed")
    seeds = tuple(seed_values)
    if len(set(seeds)) != 10:
        raise EvaluationIntegrityError("H3 train pool contains duplicate seeds")

    inventory = shared._read_json(inventory_path)
    excluded, source_audit = validate_inventory(inventory)
    inventory_hash = sha256_json(inventory)
    if (
        pool.get("inventory_id") != inventory.get("inventory_id")
        or pool.get("inventory_sha256") != inventory_hash
        or pool.get("source_audit_sha256") != sha256_json(source_audit)
        or pool.get("source_audit") != source_audit
    ):
        raise EvaluationIntegrityError("H3 pool does not match the verified exclusion inventory")

    ledger = shared._read_json(pools_dir / "ledger.json")
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if sha256_json(ledger_payload) != ledger.get("ledger_sha256"):
        raise EvaluationIntegrityError("round seed ledger hash mismatch")
    ledger_pool = ledger.get("pools", {}).get("train_hypothesis_3")
    if (
        not isinstance(ledger_pool, dict)
        or ledger_pool.get("manifest_sha256") != recorded_hash
        or ledger_pool.get("seed_ids") != seed_values
    ):
        raise EvaluationIntegrityError("round ledger and H3 pool file differ")

    seen: set[int] = set()
    pool_count = 0
    ledger_pools = ledger.get("pools", {})
    expected_files = {
        "train_hypothesis_1": "train_hypothesis_1.json",
        "train_hypothesis_2": "train_hypothesis_2.json",
        "train_hypothesis_3": "train_hypothesis_3.json",
        "probe": "probe.json",
        "dev": "dev.json",
    }
    for key, filename in expected_files.items():
        path = pools_dir / filename
        other = shared._read_json(path)
        other_payload = {name: value for name, value in other.items() if name != "manifest_sha256"}
        other_hash = other.get("manifest_sha256")
        if sha256_json(other_payload) != other_hash:
            raise EvaluationIntegrityError("a round pool manifest hash is invalid")
        ledger_entry = ledger_pools.get(key)
        other_seeds = other.get("seed_ids")
        if (
            not isinstance(ledger_entry, dict)
            or ledger_entry.get("manifest_sha256") != other_hash
            or ledger_entry.get("seed_ids") != other_seeds
            or not isinstance(other_seeds, list)
        ):
            raise EvaluationIntegrityError("round ledger and pool manifest differ")
        normalized = set(other_seeds)
        if len(normalized) != len(other_seeds) or normalized & seen:
            raise EvaluationIntegrityError("round pool manifests overlap")
        if normalized & excluded:
            raise EvaluationIntegrityError("a round pool overlaps the protected exclusion inventory")
        seen.update(normalized)
        pool_count += 1
    if pool_count != 5 or set(seeds) & excluded or set(seeds) != set(ledger_pool["seed_ids"]):
        raise EvaluationIntegrityError("H3 pool or round inventory is incomplete")

    return pool, seeds, {
        "train_seed_count": len(seeds),
        "exclusion_seed_count": len(excluded),
        "pool_manifest_count": pool_count,
        "inventory_sha256": inventory_hash,
        "disjoint_from_exclusion_inventory": True,
        "round_pools_pairwise_disjoint": True,
    }


def _check_private_paths(output_dir: Path, usage_ledger: Path) -> None:
    repo = REPO_ROOT.resolve()
    for path in (output_dir.resolve(), usage_ledger.resolve()):
        if path == repo or repo in path.parents:
            raise EvaluationIntegrityError("raw H3 evidence must stay outside the repository")
    if output_dir.exists():
        raise EvaluationIntegrityError("private output directory exists; never rerun this pool")
    if usage_ledger.exists():
        if any(event.get("pool_id") == POOL_ID for event in shared._read_jsonl(usage_ledger)):
            raise EvaluationIntegrityError("H3 pool already has a private use record")


def _summary_view(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "result", "outcome", "combat_policy", "noncombat_policy", "final_floor", "final_hp",
        "armg_action_count", "mcts_action_count", "potion_reserve_override_count",
        "use_potion_below_hp_fraction", "emergency_potion_override_count",
        "map_elite_avoidance_override_count", "map_elite_avoidance_fail_closed_count",
        "illegal_action_count", "timeout_count", "crash_count", "game_steps",
    )
    return {key: result.get(key) for key in keys}


def _check_trace(path: Path, *, candidate: bool, expected_overrides: int) -> dict[str, Any]:
    terminal = None
    combat_decisions = 0
    low_hp_potion_opportunities = 0
    overrides = 0
    for event in shared._read_jsonl(path):
        event_type = event.get("type")
        if event_type == "terminal_trace_v1":
            terminal = event
            continue
        if event_type == "noncombat_decision_trace_v1":
            if event.get("legal_choices_complete") is not True:
                raise EvaluationIntegrityError("noncombat legal choices are incomplete")
            if event.get("screen_before") == "MAP_SCREEN":
                route = event.get("route")
                if not isinstance(route, dict) or route.get("choices_complete") is not True:
                    raise EvaluationIntegrityError("map route trace is incomplete")
            continue
        if event_type != "combat_decision_trace_v1":
            continue
        combat_decisions += 1
        if event.get("legal_actions_complete") is not True or event.get("mcts_sims") != MCTS_SIMS:
            raise EvaluationIntegrityError("combat legal-action trace or MCTS budget is invalid")
        legal = event.get("canonical_native_legal_actions")
        selected_index = event.get("selected_native_action_index")
        selected = event.get("selected_action")
        recommended = event.get("mcts_recommended_action")
        if (
            not isinstance(legal, list)
            or not legal
            or not isinstance(selected_index, int)
            or isinstance(selected_index, bool)
            or not 0 <= selected_index < len(legal)
            or selected != legal[selected_index]
            or recommended not in legal
        ):
            raise EvaluationIntegrityError("selected or recommended action is not mapped to native legal actions")
        override = event.get("emergency_potion_override")
        if not isinstance(override, bool):
            raise EvaluationIntegrityError("emergency potion trace flag is missing")
        state = event.get("public_state")
        hp = state.get("hp") if isinstance(state, dict) else None
        max_hp = state.get("max_hp") if isinstance(state, dict) else None
        hp_valid = (
            isinstance(hp, (int, float)) and not isinstance(hp, bool)
            and isinstance(max_hp, (int, float)) and not isinstance(max_hp, bool)
            and math.isfinite(float(hp)) and math.isfinite(float(max_hp))
            and hp > 0 and max_hp > 0
        )
        hp_eligible = hp_valid and hp / max_hp <= HP_RATIO_THRESHOLD
        potion_actions = [
            (action.get("potion_index"), index, action)
            for index, action in enumerate(legal)
            if isinstance(action, dict) and action.get("kind") == "use_potion"
            and isinstance(action.get("potion_index"), int)
            and not isinstance(action.get("potion_index"), bool)
        ]
        recommended_is_potion = isinstance(recommended, dict) and recommended.get("kind") == "use_potion"
        opportunity = bool(candidate and hp_eligible and potion_actions and not recommended_is_potion)
        if opportunity:
            low_hp_potion_opportunities += 1
        expected_override = opportunity
        if override is not expected_override:
            raise EvaluationIntegrityError("emergency potion policy action differs from its frozen rule")
        if override:
            expected_slot, expected_index, _ = min(potion_actions, key=lambda row: (row[0], row[1]))
            if (
                not candidate
                or not isinstance(selected, dict)
                or selected.get("kind") != "use_potion"
                or selected.get("potion_index") != expected_slot
                or selected_index != expected_index
            ):
                raise EvaluationIntegrityError("emergency potion did not choose the first legal slot/action")
            overrides += 1
    if (
        not isinstance(terminal, dict)
        or terminal.get("complete") is not True
        or terminal.get("legal_actions_complete") is not True
        or terminal.get("illegal_action_count") != 0
        or terminal.get("timeout_count") != 0
        or terminal.get("crash_count") != 0
        or combat_decisions < 1
    ):
        raise EvaluationIntegrityError("terminal trace failed completeness or safety guards")
    if candidate and overrides != expected_overrides:
        raise EvaluationIntegrityError("trace override count differs from the run summary")
    if not candidate and overrides:
        raise EvaluationIntegrityError("parent run unexpectedly used the H3 intervention")
    return {
        "combat_decisions": combat_decisions,
        "low_hp_potion_opportunities": low_hp_potion_opportunities,
        "overrides": overrides,
    }


def _action_signature(path: Path) -> str:
    selected_events: list[dict[str, Any]] = []
    for event in shared._read_jsonl(path):
        event_type = event.get("type")
        if event_type == "armg_noncombat_decision_v3":
            selected_events.append({
                "type": event_type,
                "kind": event.get("kind"),
                "recommended_index": event.get("recommended_index"),
                "selected_index": event.get("selected_index"),
                "choice_semantics": event.get("choice_semantics"),
                "choice_scores": event.get("choice_scores"),
                "map_policy_intervention": event.get("map_policy_intervention"),
            })
        elif event_type == "simulator_combat_action":
            selected_events.append({
                "type": event_type,
                "floor": event.get("floor"),
                "policy": event.get("policy"),
                "mcts_sims": event.get("mcts_sims"),
                "action_id": event.get("action_id"),
                "action_index": event.get("action_index"),
                "native_action_index": event.get("native_action_index"),
                "emergency_potion_override": event.get("emergency_potion_override"),
            })
    return sha256_json(selected_events)


def _run_one(
    *, sts: Any, policy: ArmGNoncombatPolicy, seed: int,
    all_training_seeds: tuple[int, ...], output_dir: Path, pair_index: int,
    arm: str, candidate: bool, trace_enabled: bool, metadata: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path | None, dict[str, Any] | None]:
    stem = f"pair-{pair_index:02d}-{arm}{'-trace-off' if not trace_enabled else ''}"
    evidence_path = output_dir / f"{stem}.ndjson"
    trace_path = output_dir / f"{stem}.trace.ndjson" if trace_enabled else None
    result = run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=evidence_path,
        armg_policy=policy,
        combat_mcts_sims=MCTS_SIMS,
        reserve_last_potion_until_floor=None,
        use_potion_below_hp_fraction=HP_RATIO_THRESHOLD if candidate else None,
        avoid_low_hp_elite_routes=False,
        training_seeds=all_training_seeds,
        collect_ppo=False,
        collect_teacher=False,
        diagnostic_trace_path=trace_path,
        diagnostic_metadata=metadata,
    )
    shared._check_evidence(evidence_path, result, candidate=False)
    trace_stats = None
    if trace_path is not None:
        trace_stats = _check_trace(
            trace_path,
            candidate=candidate,
            expected_overrides=int(result.get("emergency_potion_override_count", 0)),
        )
    return result, evidence_path, trace_path, trace_stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--pools-dir", type=Path, required=True)
    parser.add_argument("--exclusion-inventory", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--g7-checkpoint", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--private-usage-ledger", type=Path, required=True)
    parser.add_argument("--candidate-commit")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        pool, seeds, preflight = _validate_pool(
            args.pool_file.resolve(), args.pools_dir.resolve(), args.exclusion_inventory.resolve()
        )
        _check_private_paths(args.private_output_dir, args.private_usage_ledger)
        checkpoint_sha = _sha256(args.g7_checkpoint)
        if checkpoint_sha != G7_SHA256:
            raise EvaluationIntegrityError("checkpoint does not match the pinned G7 parent")
        if Path(str(args.g7_checkpoint) + ".adapter.pt").exists():
            raise EvaluationIntegrityError("H3 requires the unmodified G7 parent")
        binding = args.module_dir / "slaythespire.cp312-win_amd64.pyd"
        if _sha256(binding) != SIMULATOR_BINDING_SHA256:
            raise EvaluationIntegrityError("native simulator binding hash mismatch")
        armg_source = args.armg_root / "armG_train.py"
        if not armg_source.is_file():
            raise EvaluationIntegrityError("pinned ArmG source file is missing")
        if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
            raise EvaluationIntegrityError("contextual reranking must be disabled for fixed H3 comparisons")

        candidate_commit = args.candidate_commit
        if not args.preflight_only:
            if not isinstance(candidate_commit, str) or len(candidate_commit) != 40:
                raise EvaluationIntegrityError("run requires the frozen 40-character candidate commit SHA")
            observed_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
            if observed_commit != candidate_commit:
                raise EvaluationIntegrityError("local HEAD differs from the frozen candidate commit SHA")

        sts = _load_sts(args.module_dir)
        parent_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.g7_checkpoint)
        candidate_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.g7_checkpoint)
        metadata = {
            "stage": "train_hypothesis_3",
            "pool_id": POOL_ID,
            "pool_manifest_sha256": EXPECTED_POOL_SHA256,
            "simulator_commit": SIMULATOR_COMMIT,
            "simulator_python_source_sha256": _sha256(Path(simulator_module.__file__).resolve()),
            "runner_source_sha256": _sha256(Path(__file__).resolve()),
            "candidate_git_commit_sha": candidate_commit or "NOT_COMMITTED_PREFLIGHT",
            "simulator_binding_sha256": SIMULATOR_BINDING_SHA256,
            "armg_source_sha256": _sha256(armg_source),
            "g7_checkpoint_sha256": checkpoint_sha,
            "mcts_sims": MCTS_SIMS,
            "candidate_intervention": "at HP/maxHP <= 0.5, use the lowest-slot legal potion; preserve the parent MCTS action when required data or a legal potion is unavailable",
            "tie_break": "lowest_potion_slot_then_native_legal_action_order",
        }
        if args.preflight_only:
            print(json.dumps({"preflight": "PASS", **preflight, **metadata}, sort_keys=True))
            return 0

        args.private_output_dir.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        shared._append_jsonl(args.private_usage_ledger, {
            "pool_id": POOL_ID,
            "manifest_sha256": EXPECTED_POOL_SHA256,
            "stage": "train_hypothesis_3",
            "status": "RUNNING",
            "started_at_utc": started,
            "expected_paired_episodes": 20,
            "trace_invariance_episodes": 1,
            "candidate_source_sha256": metadata["simulator_python_source_sha256"],
        })
        completed_episodes = 0
        parent_outcomes: list[str] = []
        candidate_outcomes: list[str] = []
        rows: list[dict[str, Any]] = []
        try:
            for pair_index, seed in enumerate(seeds):
                print(f"H3 train pair {pair_index + 1}/10: parent")
                parent_result, _, _, parent_trace_stats = _run_one(
                    sts=sts, policy=parent_policy, seed=seed, all_training_seeds=seeds,
                    output_dir=args.private_output_dir, pair_index=pair_index, arm="parent",
                    candidate=False, trace_enabled=True, metadata=metadata,
                )
                completed_episodes += 1
                print(f"H3 train pair {pair_index + 1}/10: candidate")
                candidate_result, candidate_evidence, _, candidate_trace_stats = _run_one(
                    sts=sts, policy=candidate_policy, seed=seed, all_training_seeds=seeds,
                    output_dir=args.private_output_dir, pair_index=pair_index, arm="candidate",
                    candidate=True, trace_enabled=True, metadata=metadata,
                )
                completed_episodes += 1
                trace_invariance = None
                if pair_index == 0:
                    print("H3 tracing invariance check: candidate with trace disabled")
                    trace_off_result, trace_off_evidence, _, _ = _run_one(
                        sts=sts, policy=candidate_policy, seed=seed, all_training_seeds=seeds,
                        output_dir=args.private_output_dir, pair_index=pair_index, arm="candidate",
                        candidate=True, trace_enabled=False, metadata=metadata,
                    )
                    completed_episodes += 1
                    if (
                        trace_off_result.get("outcome") != candidate_result.get("outcome")
                        or _summary_view(trace_off_result) != _summary_view(candidate_result)
                        or _action_signature(trace_off_evidence) != _action_signature(candidate_evidence)
                    ):
                        raise EvaluationIntegrityError("trace on/off changed the fixed-seed H3 trajectory")
                    trace_invariance = "PASS"

                parent_outcomes.append(str(parent_result["outcome"]))
                candidate_outcomes.append(str(candidate_result["outcome"]))
                rows.append({
                    "pair_index": pair_index,
                    "parent": _summary_view(parent_result),
                    "candidate": _summary_view(candidate_result),
                    "parent_trace": parent_trace_stats,
                    "candidate_trace": candidate_trace_stats,
                    "trace_invariance": trace_invariance,
                })
                shared._append_jsonl(args.private_output_dir / "paired-runs.ndjson", rows[-1])

            paired = shared._paired_summary(parent_outcomes, candidate_outcomes)
            candidate_overrides = sum(int(row["candidate"].get("emergency_potion_override_count") or 0) for row in rows)
            low_hp_opportunities = sum(int(row["candidate_trace"]["low_hp_potion_opportunities"]) for row in rows)
            final = {
                "schema_version": "sts1-g7-h3-train-paired-v1",
                "status": "COMPLETE",
                "pool_id": POOL_ID,
                "pool_manifest_sha256": EXPECTED_POOL_SHA256,
                "paired_seed_count": 10,
                "episode_count": completed_episodes,
                "trace_invariance": "PASS",
                "preflight": preflight,
                "identities": metadata,
                "paired": paired,
                "candidate_combat_decisions": sum(int(row["candidate_trace"]["combat_decisions"]) for row in rows),
                "candidate_low_hp_potion_opportunities": low_hp_opportunities,
                "candidate_emergency_potion_overrides": candidate_overrides,
                "communication_errors": "N/A_LOCAL_SIMULATOR",
            }
            (args.private_output_dir / "paired-summary.json").write_text(
                json.dumps(final, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            shared._append_jsonl(args.private_usage_ledger, {
                "pool_id": POOL_ID,
                "manifest_sha256": EXPECTED_POOL_SHA256,
                "stage": "train_hypothesis_3",
                "status": "COMPLETE",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "episode_count": completed_episodes,
                "trace_invariance": "PASS",
                "candidate_source_sha256": metadata["simulator_python_source_sha256"],
            })
            print(json.dumps({
                "status": final["status"],
                "episode_count": completed_episodes,
                **paired,
                "candidate_combat_decisions": final["candidate_combat_decisions"],
                "candidate_low_hp_potion_opportunities": low_hp_opportunities,
                "candidate_emergency_potion_overrides": candidate_overrides,
            }, sort_keys=True))
            return 0
        except BaseException as exc:
            shared._append_jsonl(args.private_usage_ledger, {
                "pool_id": POOL_ID,
                "manifest_sha256": EXPECTED_POOL_SHA256,
                "stage": "train_hypothesis_3",
                "status": "NOT_VERIFIED",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "episode_count": completed_episodes,
                "error_type": type(exc).__name__,
                "candidate_source_sha256": metadata["simulator_python_source_sha256"],
            })
            print(json.dumps({
                "status": "NOT_VERIFIED",
                "episode_count": completed_episodes,
                "error_type": type(exc).__name__,
            }, sort_keys=True))
            return 2
    except (EvaluationIntegrityError, shared.EvaluationIntegrityError, OSError, ValueError) as exc:
        print(json.dumps({"status": "NOT_VERIFIED", "error_type": type(exc).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
