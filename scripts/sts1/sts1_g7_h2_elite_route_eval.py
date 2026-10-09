"""Run the pre-registered H2 train pool as a private paired G7 counterfactual."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from roguelike_ai.sts1_phase3.simulator import (  # noqa: E402
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)
from sts1_g7_seed_ledger import MAX_SEED, sha256_json, validate_inventory  # noqa: E402


POOL_ID = "round-003-20261009-train_hypothesis_2"
POOL_FILE = "train_hypothesis_2.json"
EXPECTED_POOL_SHA256 = "101477197215c2b53773848f1f11182e3d9e9aa5b1539bcbc8b836b7ec9ba8a0"
G7_SHA256 = "8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0"
SIMULATOR_BINDING_SHA256 = "dea5e3b88097c7e7b7ffb74f03c227b7244a548b07fc5344c6b195d737c65512"
MCTS_SIMS = 2000
FAIL_CLOSED_REASONS = frozenset(
    {
        "player_hp_unavailable_or_invalid",
        "map_route_incomplete",
        "map_choice_count_or_index_invalid",
        "armg_scores_incomplete",
        "known_room_vocabulary_unavailable",
        "map_legal_action_order_mismatch",
        "map_room_or_target_unknown",
        "armg_score_invalid",
        "armg_score_non_finite",
        "no_known_non_elite_route",
    }
)


class EvaluationIntegrityError(RuntimeError):
    """A private H2 run did not satisfy its frozen manifest or safety contract."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationIntegrityError("expected a JSON object")
    return value


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _validate_pool(pool_path: Path, pools_dir: Path, inventory_path: Path) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    if pool_path.name != POOL_FILE:
        raise EvaluationIntegrityError("only the pre-registered H2 train pool is accepted")
    pool = _read_json(pool_path)
    if pool.get("pool_id") != POOL_ID or pool.get("purpose") != "train":
        raise EvaluationIntegrityError("H2 pool identity or purpose mismatch")
    if pool.get("status") != "GENERATED_NOT_RUN":
        raise EvaluationIntegrityError("H2 pool is not marked unused")
    recorded_hash = pool.get("manifest_sha256")
    payload = {key: value for key, value in pool.items() if key != "manifest_sha256"}
    if recorded_hash != EXPECTED_POOL_SHA256 or sha256_json(payload) != recorded_hash:
        raise EvaluationIntegrityError("H2 pool manifest hash mismatch")
    seed_values = pool.get("seed_ids")
    if not isinstance(seed_values, list) or len(seed_values) != 10:
        raise EvaluationIntegrityError("H2 train pool must contain exactly ten seeds")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or not 1 <= seed <= MAX_SEED
        for seed in seed_values
    ):
        raise EvaluationIntegrityError("H2 pool contains an invalid simulator seed")
    seeds = tuple(seed_values)
    if len(set(seeds)) != 10:
        raise EvaluationIntegrityError("H2 train pool contains duplicate seeds")

    inventory = _read_json(inventory_path)
    excluded, source_audit = validate_inventory(inventory)
    inventory_hash = sha256_json(inventory)
    if (
        pool.get("inventory_id") != inventory.get("inventory_id")
        or pool.get("inventory_sha256") != inventory_hash
        or pool.get("source_audit_sha256") != sha256_json(source_audit)
        or pool.get("source_audit") != source_audit
    ):
        raise EvaluationIntegrityError("H2 pool does not match the verified exclusion inventory")
    if set(seeds) & excluded:
        raise EvaluationIntegrityError("H2 train pool overlaps the existing exclusion inventory")

    ledger = _read_json(pools_dir / "ledger.json")
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if sha256_json(ledger_payload) != ledger.get("ledger_sha256"):
        raise EvaluationIntegrityError("H2 round seed ledger hash mismatch")
    ledger_h2 = ledger.get("pools", {}).get("train_hypothesis_2")
    if not isinstance(ledger_h2, dict) or ledger_h2.get("manifest_sha256") != recorded_hash or ledger_h2.get("seed_ids") != seed_values:
        raise EvaluationIntegrityError("round-001 seed ledger and H2 pool file differ")

    seen: set[int] = set()
    pool_count = 0
    for path in sorted(pools_dir.glob("*.json")):
        if path.name in {"ledger.json", "exclusion_inventory_private.json"}:
            continue
        other = _read_json(path)
        other_payload = {key: value for key, value in other.items() if key != "manifest_sha256"}
        if sha256_json(other_payload) != other.get("manifest_sha256"):
            raise EvaluationIntegrityError("an H2 round pool manifest hash is invalid")
        other_seeds = other.get("seed_ids")
        if not isinstance(other_seeds, list):
            raise EvaluationIntegrityError("an H2 round pool is missing its seed list")
        normalized = set(other_seeds)
        if len(normalized) != len(other_seeds) or normalized & seen:
            raise EvaluationIntegrityError("H2 round pool manifests overlap")
        seen.update(normalized)
        pool_count += 1
    if pool_count != 5 or not set(seeds).issubset(seen):
        raise EvaluationIntegrityError("H2 round pool inventory is incomplete")

    return pool, seeds, {
        "h2_seed_count": len(seeds),
        "exclusion_seed_count": len(excluded),
        "pool_manifest_count": pool_count,
        "inventory_sha256": inventory_hash,
        "disjoint_from_exclusion_inventory": True,
        "h2_round_pools_pairwise_disjoint": True,
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _check_private_paths(output_dir: Path, usage_ledger: Path) -> None:
    repo = REPO_ROOT.resolve()
    for path in (output_dir.resolve(), usage_ledger.resolve()):
        if path == repo or repo in path.parents:
            raise EvaluationIntegrityError("raw H2 evidence and usage records must stay outside the repository")
    if output_dir.exists():
        raise EvaluationIntegrityError("private output directory already exists; do not rerun this pool")
    if usage_ledger.exists():
        for event in _read_jsonl(usage_ledger):
            if event.get("pool_id") == POOL_ID:
                raise EvaluationIntegrityError("H2 pool already has a private use record")


def _summary_view(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "result",
        "outcome",
        "combat_policy",
        "noncombat_policy",
        "final_floor",
        "final_hp",
        "armg_action_count",
        "mcts_action_count",
        "map_elite_avoidance_override_count",
        "map_elite_avoidance_fail_closed_count",
        "illegal_action_count",
        "timeout_count",
        "crash_count",
        "game_steps",
    )
    return {key: result.get(key) for key in keys}


def _check_terminal_trace(
    path: Path,
    *,
    candidate: bool,
    expected_overrides: int,
    expected_fail_closed: int,
) -> dict[str, Any]:
    terminal = None
    map_decisions = 0
    low_hp_elite_recommendations = 0
    overrides = 0
    fail_closed_reasons: dict[str, int] = {}
    for event in _read_jsonl(path):
        if event.get("type") == "terminal_trace_v1":
            terminal = event
        if event.get("type") != "noncombat_decision_trace_v1":
            continue
        if event.get("legal_choices_complete") is not True:
            raise EvaluationIntegrityError("noncombat legal-choice trace is incomplete")
        if event.get("screen_before") == "MAP_SCREEN":
            route = event.get("route")
            if not isinstance(route, dict) or route.get("choices_complete") is not True:
                raise EvaluationIntegrityError("map route trace is incomplete")
            if candidate:
                map_decisions += 1
                detail = event.get("map_policy_intervention")
                actual_index = event.get("selected_legal_action_index")
                recommended_index = event.get("recommended_legal_action_index")
                if not isinstance(detail, dict):
                    raise EvaluationIntegrityError("candidate map intervention trace is missing")
                for value in (detail.get("recommended_index"), detail.get("actual_index")):
                    if not isinstance(value, int) or isinstance(value, bool):
                        raise EvaluationIntegrityError("map intervention contains a non-integer legal-action index")
                if detail.get("recommended_index") != recommended_index:
                    raise EvaluationIntegrityError("map recommended index differs across trace records")
                if detail.get("actual_index") != actual_index:
                    raise EvaluationIntegrityError("map actual index differs across trace records")
                if route.get("actual_selected_index") != actual_index:
                    raise EvaluationIntegrityError("map route and executed legal-action index differ")
                if route.get("recommended_selected_index") != recommended_index:
                    raise EvaluationIntegrityError("map route recommendation index is inconsistent")
                if (
                    not isinstance(actual_index, int)
                    or isinstance(actual_index, bool)
                    or not isinstance(recommended_index, int)
                    or isinstance(recommended_index, bool)
                ):
                    raise EvaluationIntegrityError("map trace contains a non-integer legal-action index")
                route_choices = route.get("choices")
                legal_choices = event.get("legal_choices")
                if (
                    not isinstance(route_choices, list)
                    or not isinstance(legal_choices, list)
                    or len(route_choices) != len(legal_choices)
                    or route.get("choice_count") != len(route_choices)
                    or not 0 <= actual_index < len(route_choices)
                    or not 0 <= recommended_index < len(route_choices)
                ):
                    raise EvaluationIntegrityError("map route choices do not cover the complete legal-action list")
                if any(
                    not isinstance(choice, dict) or choice.get("legal_action_index") != index
                    for index, choice in enumerate(route_choices)
                ):
                    raise EvaluationIntegrityError("map route indices do not match the legal-action order")
                actual_route = route.get("selected_route")
                recommended_route = route.get("recommended_route")
                if (
                    not isinstance(actual_route, dict)
                    or not isinstance(recommended_route, dict)
                    or actual_route.get("legal_action_index") != actual_index
                    or recommended_route.get("legal_action_index") != recommended_index
                    or actual_route.get("room") != route_choices[actual_index].get("target_room")
                    or recommended_route.get("room") != route_choices[recommended_index].get("target_room")
                ):
                    raise EvaluationIntegrityError("map route details do not match the legal-action choices")
                status = detail.get("status")
                if status not in {"unchanged", "overridden", "fail_closed"}:
                    raise EvaluationIntegrityError("candidate map intervention has an unknown status")
                hp_ratio = detail.get("hp_ratio")
                reason = detail.get("reason")
                if hp_ratio is None:
                    if status != "fail_closed" or reason != "player_hp_unavailable_or_invalid":
                        raise EvaluationIntegrityError("candidate map intervention has an invalid HP ratio")
                elif (
                    not isinstance(hp_ratio, (int, float))
                    or isinstance(hp_ratio, bool)
                    or not math.isfinite(float(hp_ratio))
                    or float(hp_ratio) < 0
                ):
                    raise EvaluationIntegrityError("candidate map intervention has an invalid HP ratio")
                if isinstance(hp_ratio, (int, float)) and not isinstance(hp_ratio, bool):
                    if recommended_route.get("room") == "ELITE" and float(hp_ratio) < 0.5:
                        low_hp_elite_recommendations += 1
                if detail.get("recommended_room") is not None and detail.get("recommended_room") != recommended_route.get("room"):
                    raise EvaluationIntegrityError("map recommended room differs across trace records")
                if detail.get("actual_room") is not None and detail.get("actual_room") != actual_route.get("room"):
                    raise EvaluationIntegrityError("map actual room differs across trace records")
                if status == "overridden" and (
                    detail.get("recommended_room") is None
                    or detail.get("actual_room") is None
                    or not isinstance(hp_ratio, (int, float))
                    or float(hp_ratio) >= 0.5
                    or recommended_route.get("room") != "ELITE"
                    or actual_route.get("room") == "ELITE"
                ):
                    raise EvaluationIntegrityError("H2 override did not replace a low-HP Elite route")
                if status == "fail_closed":
                    if detail.get("overridden") is not False or actual_index != recommended_index:
                        raise EvaluationIntegrityError("fail-closed map decision changed the ArmG action")
                    if reason not in FAIL_CLOSED_REASONS:
                        raise EvaluationIntegrityError("fail-closed map decision has an unknown reason")
                    fail_closed_reasons[reason] = fail_closed_reasons.get(reason, 0) + 1
                if detail.get("overridden") is not (status == "overridden"):
                    raise EvaluationIntegrityError("map override flag and status disagree")
                if actual_index != recommended_index and detail.get("overridden") is not True:
                    raise EvaluationIntegrityError("selected route changed without an intervention")
                if status == "overridden" and actual_index == recommended_index:
                    raise EvaluationIntegrityError("map intervention reports an override without changing the route")
                if status == "overridden":
                    overrides += 1
    if (
        not isinstance(terminal, dict)
        or terminal.get("complete") is not True
        or terminal.get("legal_actions_complete") is not True
        or terminal.get("illegal_action_count") != 0
        or terminal.get("timeout_count") != 0
        or terminal.get("crash_count") != 0
    ):
        raise EvaluationIntegrityError("terminal trace failed completeness or safety guards")
    if candidate and map_decisions == 0:
        raise EvaluationIntegrityError("candidate map-decision trace is missing")
    fail_closed_count = sum(fail_closed_reasons.values())
    if candidate and overrides != expected_overrides:
        raise EvaluationIntegrityError("candidate map-intervention trace counts do not match its summary")
    if candidate and fail_closed_count != expected_fail_closed:
        raise EvaluationIntegrityError("candidate fail-closed trace counts do not match its summary")
    return {
        "map_decisions": map_decisions,
        "low_hp_elite_recommendations": low_hp_elite_recommendations,
        "overrides": overrides,
        "fail_closed_count": fail_closed_count,
        "fail_closed_reasons": fail_closed_reasons,
    }


def _check_evidence(path: Path, result: dict[str, Any], *, candidate: bool) -> None:
    combat_actions = 0
    summary_event = None
    for event in _read_jsonl(path):
        if event.get("type") == "simulator_combat_action":
            combat_actions += 1
            if event.get("policy") != "mcts" or event.get("mcts_sims") != MCTS_SIMS:
                raise EvaluationIntegrityError("a combat action did not use the fixed MCTS-2000 budget")
        elif event.get("type") == "summary":
            summary_event = event
    if (
        result.get("result") != "PASS_SIMULATOR_COMPLETE_RUN"
        or result.get("outcome") not in {"victory", "defeat"}
        or result.get("error") is not None
        or result.get("combat_policy") != "mcts_2000"
        or result.get("illegal_action_count") != 0
        or result.get("timeout_count") != 0
        or result.get("crash_count") != 0
        or combat_actions < 1
        or combat_actions != result.get("mcts_action_count")
    ):
        raise EvaluationIntegrityError("run failed terminal, safety, or MCTS-2000 validation")
    if not isinstance(summary_event, dict) or summary_event.get("outcome") != result.get("outcome"):
        raise EvaluationIntegrityError("private evidence summary does not match the returned result")
    if candidate:
        fail_closed_count = result.get("map_elite_avoidance_fail_closed_count")
        if not isinstance(fail_closed_count, int) or isinstance(fail_closed_count, bool) or fail_closed_count < 0:
            raise EvaluationIntegrityError("candidate fail-closed count is invalid")


def _action_signature(path: Path) -> str:
    selected_events: list[dict[str, Any]] = []
    for event in _read_jsonl(path):
        event_type = event.get("type")
        if event_type == "armg_noncombat_decision_v3":
            selected_events.append(
                {
                    "type": event_type,
                    "kind": event.get("kind"),
                    "recommended_index": event.get("recommended_index"),
                    "selected_index": event.get("selected_index"),
                    "choice_semantics": event.get("choice_semantics"),
                    "choice_scores": event.get("choice_scores"),
                    "map_policy_intervention": event.get("map_policy_intervention"),
                }
            )
        elif event_type == "simulator_combat_action":
            selected_events.append(
                {
                    "type": event_type,
                    "floor": event.get("floor"),
                    "policy": event.get("policy"),
                    "mcts_sims": event.get("mcts_sims"),
                    "chosen_bits": event.get("chosen_bits"),
                    "action_id": event.get("action_id"),
                    "action_index": event.get("action_index"),
                    "native_action_index": event.get("native_action_index"),
                }
            )
    return sha256_json(selected_events)


def _paired_summary(parent: list[str], candidate: list[str]) -> dict[str, Any]:
    if len(parent) != 10 or len(candidate) != 10:
        raise EvaluationIntegrityError("paired H2 training requires ten complete pairs")
    if any(value not in {"victory", "defeat"} for value in parent + candidate):
        raise EvaluationIntegrityError("unknown outcomes cannot be counted as losses")
    parent_wins = sum(value == "victory" for value in parent)
    candidate_wins = sum(value == "victory" for value in candidate)
    candidate_only = sum(p != "victory" and c == "victory" for p, c in zip(parent, candidate, strict=True))
    parent_only = sum(p == "victory" and c != "victory" for p, c in zip(parent, candidate, strict=True))
    discordant = candidate_only + parent_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = sum(math.comb(discordant, value) for value in range(candidate_only, discordant + 1)) / (2**discordant)
    return {
        "parent_wins": parent_wins,
        "candidate_wins": candidate_wins,
        "candidate_only_wins": candidate_only,
        "parent_only_wins": parent_only,
        "net_wins": candidate_wins - parent_wins,
        "discordant_pairs": discordant,
        "exact_one_sided_sign_p_candidate_positive": p_value,
    }


def _run_one(
    *,
    sts: Any,
    policy: ArmGNoncombatPolicy,
    seed: int,
    all_training_seeds: tuple[int, ...],
    output_dir: Path,
    pair_index: int,
    arm: str,
    candidate: bool,
    trace_enabled: bool,
    diagnostic_metadata: dict[str, Any],
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
        avoid_low_hp_elite_routes=candidate,
        training_seeds=all_training_seeds,
        collect_ppo=False,
        collect_teacher=False,
        diagnostic_trace_path=trace_path,
        diagnostic_metadata=diagnostic_metadata,
    )
    _check_evidence(evidence_path, result, candidate=candidate)
    if trace_path is not None:
        trace_stats = _check_terminal_trace(
            trace_path,
            candidate=candidate,
            expected_overrides=int(result.get("map_elite_avoidance_override_count", 0)),
            expected_fail_closed=int(result.get("map_elite_avoidance_fail_closed_count", 0)),
        )
    else:
        trace_stats = None
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
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        pool, seeds, preflight = _validate_pool(
            args.pool_file.resolve(), args.pools_dir.resolve(), args.exclusion_inventory.resolve()
        )
        _check_private_paths(args.private_output_dir, args.private_usage_ledger)
        checkpoint_sha = _file_sha256(args.g7_checkpoint)
        if checkpoint_sha != G7_SHA256:
            raise EvaluationIntegrityError("local checkpoint does not match the pinned G7 hash")
        if Path(str(args.g7_checkpoint) + ".adapter.pt").exists():
            raise EvaluationIntegrityError("G7 checkpoint has an adapter sidecar; H2 must use the unmodified G7 policy")
        binding = args.module_dir / "slaythespire.cp312-win_amd64.pyd"
        if _file_sha256(binding) != SIMULATOR_BINDING_SHA256:
            raise EvaluationIntegrityError("native simulator binding hash mismatch")
        armg_source = args.armg_root / "armG_train.py"
        if not armg_source.is_file():
            raise EvaluationIntegrityError("pinned ArmG source file is missing")
        if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
            raise EvaluationIntegrityError("contextual card reranking must be disabled for the fixed H2 comparison")

        sts = _load_sts(args.module_dir)
        parent_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.g7_checkpoint)
        candidate_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.g7_checkpoint)
        metadata = {
            "stage": "train_hypothesis_2",
            "pool_id": POOL_ID,
            "pool_manifest_sha256": EXPECTED_POOL_SHA256,
            "simulator_binding_sha256": SIMULATOR_BINDING_SHA256,
            "armg_source_sha256": _file_sha256(armg_source),
            "g7_checkpoint_sha256": checkpoint_sha,
            "mcts_sims": MCTS_SIMS,
            "candidate_intervention": "avoid immediate Elite route only below 50% HP; highest G7 ArmG-scored known non-Elite legal choice; index tie-break",
        }
        if args.preflight_only:
            print(json.dumps({"preflight": "PASS", **preflight, **metadata}, sort_keys=True))
            return 0

        args.private_output_dir.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        _append_jsonl(
            args.private_usage_ledger,
            {
                "pool_id": POOL_ID,
                "manifest_sha256": EXPECTED_POOL_SHA256,
                "stage": "train_hypothesis_2",
                "status": "RUNNING",
                "started_at_utc": started,
                "expected_paired_episodes": 20,
                "trace_invariance_episodes": 1,
            },
        )

        parent_outcomes: list[str] = []
        candidate_outcomes: list[str] = []
        rows: list[dict[str, Any]] = []
        completed_episodes = 0
        try:
            for pair_index, seed in enumerate(seeds):
                print(f"H2 train pair {pair_index + 1}/10: parent")
                parent_result, _, _, _ = _run_one(
                    sts=sts,
                    policy=parent_policy,
                    seed=seed,
                    all_training_seeds=seeds,
                    output_dir=args.private_output_dir,
                    pair_index=pair_index,
                    arm="parent",
                    candidate=False,
                    trace_enabled=True,
                    diagnostic_metadata=metadata,
                )
                completed_episodes += 1
                print(f"H2 train pair {pair_index + 1}/10: candidate")
                candidate_result, candidate_evidence, _, candidate_trace_stats = _run_one(
                    sts=sts,
                    policy=candidate_policy,
                    seed=seed,
                    all_training_seeds=seeds,
                    output_dir=args.private_output_dir,
                    pair_index=pair_index,
                    arm="candidate",
                    candidate=True,
                    trace_enabled=True,
                    diagnostic_metadata=metadata,
                )
                completed_episodes += 1
                if pair_index == 0:
                    print("H2 tracing invariance check: candidate with trace disabled")
                    trace_off_result, trace_off_evidence, _, _ = _run_one(
                        sts=sts,
                        policy=candidate_policy,
                        seed=seed,
                        all_training_seeds=seeds,
                        output_dir=args.private_output_dir,
                        pair_index=pair_index,
                        arm="candidate",
                        candidate=True,
                        trace_enabled=False,
                        diagnostic_metadata=metadata,
                    )
                    completed_episodes += 1
                    if (
                        trace_off_result.get("outcome") != candidate_result.get("outcome")
                        or _summary_view(trace_off_result) != _summary_view(candidate_result)
                        or _action_signature(trace_off_evidence) != _action_signature(candidate_evidence)
                    ):
                        raise EvaluationIntegrityError("trace on/off changed the fixed-seed candidate trajectory")

                parent_outcomes.append(str(parent_result["outcome"]))
                candidate_outcomes.append(str(candidate_result["outcome"]))
                rows.append(
                    {
                        "pair_index": pair_index,
                        "parent": _summary_view(parent_result),
                        "candidate": _summary_view(candidate_result),
                        "candidate_trace": candidate_trace_stats,
                        "trace_invariance": "PASS" if pair_index == 0 else None,
                    }
                )
                _append_jsonl(args.private_output_dir / "paired-runs.ndjson", rows[-1])

            paired = _paired_summary(parent_outcomes, candidate_outcomes)
            candidate_fail_closed_reasons: dict[str, int] = {}
            for row in rows:
                for reason, count in row["candidate_trace"]["fail_closed_reasons"].items():
                    candidate_fail_closed_reasons[reason] = candidate_fail_closed_reasons.get(reason, 0) + int(count)
            final = {
                "schema_version": "sts1-g7-h2-train-paired-v1",
                "status": "COMPLETE",
                "pool_id": POOL_ID,
                "pool_manifest_sha256": EXPECTED_POOL_SHA256,
                "paired_seed_count": 10,
                "episode_count": completed_episodes,
                "trace_invariance": "PASS",
                "preflight": preflight,
                "identities": metadata,
                "paired": paired,
                "parent_map_overrides": sum(int(row["parent"]["map_elite_avoidance_override_count"] or 0) for row in rows),
                "candidate_map_overrides": sum(int(row["candidate"]["map_elite_avoidance_override_count"] or 0) for row in rows),
                "candidate_map_fail_closed": sum(int(row["candidate"]["map_elite_avoidance_fail_closed_count"] or 0) for row in rows),
                "candidate_map_fail_closed_reasons": candidate_fail_closed_reasons,
                "candidate_map_decisions": sum(int(row["candidate_trace"]["map_decisions"]) for row in rows),
                "candidate_low_hp_elite_recommendations": sum(
                    int(row["candidate_trace"]["low_hp_elite_recommendations"]) for row in rows
                ),
                "communication_errors": "N/A_LOCAL_SIMULATOR",
            }
            (args.private_output_dir / "paired-summary.json").write_text(
                json.dumps(final, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _append_jsonl(
                args.private_usage_ledger,
                {
                    "pool_id": POOL_ID,
                    "manifest_sha256": EXPECTED_POOL_SHA256,
                    "stage": "train_hypothesis_2",
                    "status": "COMPLETE",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "episode_count": completed_episodes,
                    "trace_invariance": "PASS",
                },
            )
            print(json.dumps({
                "status": final["status"],
                "episode_count": completed_episodes,
                **paired,
                "candidate_map_decisions": final["candidate_map_decisions"],
                "candidate_low_hp_elite_recommendations": final["candidate_low_hp_elite_recommendations"],
                "candidate_map_overrides": final["candidate_map_overrides"],
                "candidate_map_fail_closed": final["candidate_map_fail_closed"],
                "candidate_map_fail_closed_reasons": candidate_fail_closed_reasons,
            }, sort_keys=True))
            return 0
        except BaseException as exc:
            _append_jsonl(
                args.private_usage_ledger,
                {
                    "pool_id": POOL_ID,
                    "manifest_sha256": EXPECTED_POOL_SHA256,
                    "stage": "train_hypothesis_2",
                    "status": "NOT_VERIFIED",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_episodes": completed_episodes,
                    "error_class": type(exc).__name__,
                },
            )
            print(f"NOT_VERIFIED: stopped after {completed_episodes} episodes ({type(exc).__name__}); private evidence retained")
            return 2
    except Exception as exc:
        print(f"PREFLIGHT_FAILED: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
