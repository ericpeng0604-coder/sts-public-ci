"""Run a pre-registered H3, H4, or H5 pool as a private paired G7 counterfactual."""

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


H3_POOL_FILE = "train_hypothesis_3.json"
H4_POOL_FILE = "train_hypothesis_1.json"
H4_POLICY = "h4-campfire-overheal"
H5_POOL_FILE = "train_hypothesis_1.json"
H5_POLICY = "h5-effective-rest-heal"
H3_POLICY = "h3-emergency-potion"
G7_SHA256 = "8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0"
SIMULATOR_COMMIT = "7476a81954020087da31d41d16fddf475746ec2d"
SIMULATOR_PYTHON_SOURCE_SHA256 = "fe735348978f2886fe7b5bc3a840c743c0de6fc3e37f124ceb6644367a1e0a49"
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
    *,
    pool_id: str,
    pool_file: str,
    expected_pool_sha256: str,
    pool_key: str,
    inventory_hash_mode: str = "canonical",
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    if pool_path.name != pool_file:
        raise EvaluationIntegrityError("pool filename differs from the frozen train registration")
    pool = shared._read_json(pool_path)
    if pool.get("pool_id") != pool_id or pool.get("purpose") != "train":
        raise EvaluationIntegrityError("pool identity or purpose mismatch")
    if pool.get("status") != "GENERATED_NOT_RUN":
        raise EvaluationIntegrityError("train pool is not marked unused")
    recorded_hash = pool.get("manifest_sha256")
    payload = {key: value for key, value in pool.items() if key != "manifest_sha256"}
    if recorded_hash != expected_pool_sha256 or sha256_json(payload) != recorded_hash:
        raise EvaluationIntegrityError("train pool manifest hash mismatch")
    seed_values = pool.get("seed_ids")
    if not isinstance(seed_values, list) or len(seed_values) != 10:
        raise EvaluationIntegrityError("train pool must contain exactly ten seeds")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or not 1 <= seed <= MAX_SEED
        for seed in seed_values
    ):
        raise EvaluationIntegrityError("train pool contains an invalid simulator seed")
    seeds = tuple(seed_values)
    if len(set(seeds)) != 10:
        raise EvaluationIntegrityError("train pool contains duplicate seeds")

    inventory = shared._read_json(inventory_path)
    excluded, source_audit = validate_inventory(inventory)
    if inventory_hash_mode == "raw":
        inventory_hash = _sha256(inventory_path)
    elif inventory_hash_mode == "canonical":
        inventory_hash = sha256_json(inventory)
    else:
        raise EvaluationIntegrityError("unsupported inventory hash mode")
    if (
        pool.get("inventory_id") != inventory.get("inventory_id")
        or pool.get("inventory_sha256") != inventory_hash
        or pool.get("source_audit_sha256") != sha256_json(source_audit)
        or pool.get("source_audit") != source_audit
    ):
        raise EvaluationIntegrityError("registered train pool does not match the verified exclusion inventory")

    ledger = shared._read_json(pools_dir / "ledger.json")
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if sha256_json(ledger_payload) != ledger.get("ledger_sha256"):
        raise EvaluationIntegrityError("round seed ledger hash mismatch")
    ledger_pool = ledger.get("pools", {}).get(pool_key)
    if (
        not isinstance(ledger_pool, dict)
        or ledger_pool.get("manifest_sha256") != recorded_hash
        or ledger_pool.get("seed_ids") != seed_values
    ):
        raise EvaluationIntegrityError("round ledger and train pool file differ")

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
        raise EvaluationIntegrityError("train pool or round inventory is incomplete")

    return pool, seeds, {
        "train_seed_count": len(seeds),
        "exclusion_seed_count": len(excluded),
        "pool_manifest_count": pool_count,
        "inventory_sha256": inventory_hash,
        "inventory_hash_mode": inventory_hash_mode,
        "disjoint_from_exclusion_inventory": True,
        "round_pools_pairwise_disjoint": True,
    }


def _public_pool_registration(candidate_policy: str, pool_key: str) -> dict[str, Any]:
    """Load a public seed-free registration instead of embedding pool IDs or hashes."""
    round_by_policy = {
        H3_POLICY: "round-003-20261009",
        H4_POLICY: "round-004-20261009",
        H5_POLICY: "round-005-20261009",
    }
    expected_status = {
        H3_POLICY: "COMPLETED",
        H4_POLICY: "EVALUATED_COMPLETE",
        H5_POLICY: "GENERATED_NOT_RUN",
    }
    round_id = round_by_policy.get(candidate_policy)
    if round_id is None:
        raise EvaluationIntegrityError("candidate policy has no public pool registration")
    summary_path = (
        REPO_ROOT
        / "evidence"
        / "sts1"
        / "g7-improvement"
        / "seeds"
        / round_id
        / "summary.json"
    )
    summary = shared._read_json(summary_path)
    if (
        summary.get("schema_version") != "sts1-g7-seed-public-summary-v1"
        or summary.get("round_id") != round_id
    ):
        raise EvaluationIntegrityError("public seed registration identity mismatch")
    registration = summary.get("pools", {}).get(pool_key)
    if not isinstance(registration, dict):
        raise EvaluationIntegrityError("public seed registration is missing the requested pool")
    expected_pool_id = f"{round_id}-{pool_key}"
    pool_id = registration.get("pool_id", expected_pool_id)
    manifest_sha256 = registration.get("manifest_sha256")
    if (
        registration.get("status") != expected_status[candidate_policy]
        or registration.get("count") != 10
        or pool_id != expected_pool_id
        or registration.get("purpose", "train") != "train"
        or not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise EvaluationIntegrityError("public train registration is invalid or has the wrong status")
    return {"pool_id": pool_id, "manifest_sha256": manifest_sha256}


def _variant_override_count_key(candidate_policy: str) -> str:
    if candidate_policy in {H4_POLICY, H5_POLICY}:
        return "campfire_overheal_override_count"
    return "emergency_potion_override_count"


def _check_private_paths(output_dir: Path, usage_ledger: Path, *, pool_id: str) -> None:
    repo = REPO_ROOT.resolve()
    for path in (output_dir.resolve(), usage_ledger.resolve()):
        if path == repo or repo in path.parents:
            raise EvaluationIntegrityError("raw evaluation evidence must stay outside the repository")
    if output_dir.exists():
        raise EvaluationIntegrityError("private output directory exists; never rerun this pool")
    if usage_ledger.exists():
        if any(event.get("pool_id") == pool_id for event in shared._read_jsonl(usage_ledger)):
            raise EvaluationIntegrityError("pool already has a private use record")


def _summary_view(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "result", "outcome", "combat_policy", "noncombat_policy", "final_floor", "final_hp",
        "armg_action_count", "mcts_action_count", "potion_reserve_override_count",
        "use_potion_below_hp_fraction", "emergency_potion_override_count",
        "campfire_overheal_override_count", "prefer_smith_when_rest_overheals",
        "prefer_smith_when_overheal_exceeds_effective_rest_heal",
        "map_elite_avoidance_override_count", "map_elite_avoidance_fail_closed_count",
        "illegal_action_count", "timeout_count", "crash_count", "game_steps",
    )
    return {key: result.get(key) for key in keys}


def _paired_summary_for_stage(parent: list[str], candidate: list[str]) -> dict[str, Any]:
    """Summarize any registered paired stage without inheriting H2's fixed n=10 guard."""
    if not parent or len(parent) != len(candidate):
        raise EvaluationIntegrityError("paired stage requires equal nonempty outcome lists")
    if any(value not in {"victory", "defeat"} for value in parent + candidate):
        raise EvaluationIntegrityError("unknown outcomes cannot be counted as losses")

    parent_wins = sum(value == "victory" for value in parent)
    candidate_wins = sum(value == "victory" for value in candidate)
    candidate_only = sum(
        p != "victory" and c == "victory" for p, c in zip(parent, candidate, strict=True)
    )
    parent_only = sum(
        p == "victory" and c != "victory" for p, c in zip(parent, candidate, strict=True)
    )
    discordant = candidate_only + parent_only
    p_value = (
        1.0
        if discordant == 0
        else sum(math.comb(discordant, value) for value in range(candidate_only, discordant + 1))
        / (2**discordant)
    )
    return {
        "parent_wins": parent_wins,
        "candidate_wins": candidate_wins,
        "candidate_only_wins": candidate_only,
        "parent_only_wins": parent_only,
        "net_wins": candidate_wins - parent_wins,
        "discordant_pairs": discordant,
        "exact_one_sided_sign_p_candidate_positive": p_value,
    }


def _check_trace(
    path: Path,
    *,
    candidate: bool,
    expected_overrides: int,
    candidate_policy: str = H3_POLICY,
) -> dict[str, Any]:
    terminal = None
    combat_decisions = 0
    low_hp_potion_opportunities = 0
    campfire_overheal_opportunities = 0
    potion_overrides = 0
    campfire_overrides = 0
    for event in shared._read_jsonl(path):
        event_type = event.get("type")
        if event_type == "diagnostic_trace_header_v1":
            policy = event.get("campfire_policy_intervention")
            expects_campfire_rule = candidate and candidate_policy in {H4_POLICY, H5_POLICY}
            if expects_campfire_rule:
                expected_kind = (
                    "prefer_smith_when_overheal_exceeds_effective_rest_heal"
                    if candidate_policy == H5_POLICY
                    else "prefer_smith_when_rest_overheals"
                )
                expected_condition = (
                    "overheal > min(rest_heal, max_hp - current_hp)"
                    if candidate_policy == H5_POLICY
                    else "rest_heal > max_hp - current_hp"
                )
                if (
                    not isinstance(policy, dict)
                    or policy.get("kind") != expected_kind
                    or policy.get("override_condition") != expected_condition
                ):
                    raise EvaluationIntegrityError("campfire intervention header differs from the frozen policy")
            elif isinstance(policy, dict):
                raise EvaluationIntegrityError("unexpected Campfire intervention is enabled")
            continue
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
            if event.get("screen_before") == "REST_ROOM":
                choices = event.get("legal_choices")
                recommended_index = event.get("recommended_legal_action_index")
                selected_index = event.get("selected_legal_action_index")
                if (
                    not isinstance(choices, list)
                    or not isinstance(recommended_index, int)
                    or isinstance(recommended_index, bool)
                    or not 0 <= recommended_index < len(choices)
                    or not isinstance(selected_index, int)
                    or isinstance(selected_index, bool)
                    or not 0 <= selected_index < len(choices)
                ):
                    raise EvaluationIntegrityError("Campfire choice indices are incomplete")
                semantics = [item.get("semantics") if isinstance(item, dict) else None for item in choices]
                option_indices = [
                    item.get("option_index") if isinstance(item, dict) else None
                    for item in semantics
                ]
                if candidate and candidate_policy in {H4_POLICY, H5_POLICY} and any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in option_indices
                ):
                    raise EvaluationIntegrityError("Campfire legal option semantics are incomplete")
                rest = [i for i, value in enumerate(option_indices) if value == 0]
                smith = [i for i, value in enumerate(option_indices) if value == 1]
                state = event.get("state_before")
                hp = state.get("hp") if isinstance(state, dict) else None
                max_hp = state.get("max_hp") if isinstance(state, dict) else None
                relics = state.get("relics") if isinstance(state, dict) else None
                hp_valid = (
                    isinstance(hp, (int, float)) and not isinstance(hp, bool)
                    and isinstance(max_hp, (int, float)) and not isinstance(max_hp, bool)
                    and math.isfinite(float(hp)) and math.isfinite(float(max_hp))
                    and hp >= 0 and hp <= max_hp and max_hp > 0
                )
                overflow = False
                overheal_amount = None
                effective_rest_heal = None
                if len(rest) == 1 and len(smith) == 1 and hp_valid and isinstance(relics, list):
                    pillow = any(str(item).upper().endswith("REGAL_PILLOW") for item in relics)
                    heal = int(math.floor(float(max_hp) * 0.30 + 0.5)) + (15 if pillow else 0)
                    overflow = hp + heal > max_hp
                    overheal_amount = max(0.0, float(hp) + heal - float(max_hp))
                    effective_rest_heal = min(float(heal), float(max_hp) - float(hp))
                is_candidate_rule = candidate and candidate_policy in {H4_POLICY, H5_POLICY}
                rule_opportunity = bool(
                    overflow
                    and (
                        candidate_policy == H4_POLICY
                        or (
                            candidate_policy == H5_POLICY
                            and overheal_amount is not None
                            and effective_rest_heal is not None
                            and overheal_amount > effective_rest_heal
                        )
                    )
                )
                expected_intervention = bool(
                    is_candidate_rule
                    and len(rest) == 1
                    and len(smith) == 1
                    and option_indices[recommended_index] == 0
                    and rule_opportunity
                )
                intervention = event.get("campfire_policy_intervention")
                if is_candidate_rule and not isinstance(intervention, dict):
                    raise EvaluationIntegrityError("Campfire intervention detail is missing")
                if (
                    candidate
                    and candidate_policy == H5_POLICY
                    and len(rest) == 1
                    and len(smith) == 1
                    and option_indices[recommended_index] == 0
                    and hp_valid
                    and isinstance(relics, list)
                ):
                    expected_reason = (
                        "rest_heal_does_not_overflow"
                        if not overflow
                        else (
                            "overheal_exceeds_effective_rest_heal"
                            if rule_opportunity
                            else "overheal_not_greater_than_effective_rest_heal"
                        )
                    )
                    if (
                        not isinstance(intervention, dict)
                        or intervention.get("kind") != "prefer_smith_when_overheal_exceeds_effective_rest_heal"
                        or intervention.get("rest_heal_amount") != heal
                        or intervention.get("overheal_amount") != overheal_amount
                        or intervention.get("effective_rest_heal_amount") != effective_rest_heal
                        or intervention.get("recommended_index") != recommended_index
                        or intervention.get("selected_index") != selected_index
                        or intervention.get("reason") != expected_reason
                    ):
                        raise EvaluationIntegrityError("H5 Campfire value accounting differs from the fixed rule")
                observed_intervention = isinstance(intervention, dict) and intervention.get("overridden") is True
                if observed_intervention is not expected_intervention:
                    raise EvaluationIntegrityError("Campfire intervention differs from the preregistered fixed rule")
                if expected_intervention:
                    expected_kind = (
                        "prefer_smith_when_overheal_exceeds_effective_rest_heal"
                        if candidate_policy == H5_POLICY
                        else "prefer_smith_when_rest_overheals"
                    )
                    if (
                        selected_index != smith[0]
                        or (event.get("state_after") or {}).get("screen_state") != "CARD_SELECT"
                        or intervention.get("kind") != expected_kind
                        or intervention.get("rest_heal_amount") != heal
                        or intervention.get("recommended_index") != recommended_index
                        or intervention.get("selected_index") != selected_index
                    ):
                        raise EvaluationIntegrityError("Smith override did not execute the exact legal option")
                    campfire_overrides += 1
                elif selected_index != recommended_index:
                    raise EvaluationIntegrityError("Campfire choice changed outside the preregistered override")
                if (
                    len(rest) == 1
                    and len(smith) == 1
                    and option_indices[recommended_index] == 0
                    and rule_opportunity
                ):
                    campfire_overheal_opportunities += 1
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
        potion_candidate = candidate and candidate_policy == H3_POLICY
        opportunity = bool(potion_candidate and hp_eligible and potion_actions and not recommended_is_potion)
        if opportunity:
            low_hp_potion_opportunities += 1
        if override is not opportunity:
            raise EvaluationIntegrityError("emergency potion policy action differs from the frozen candidate variant")
        if override:
            expected_slot, expected_index, _ = min(potion_actions, key=lambda row: (row[0], row[1]))
            if not isinstance(selected, dict) or selected.get("kind") != "use_potion" or selected.get("potion_index") != expected_slot or selected_index != expected_index:
                raise EvaluationIntegrityError("emergency potion did not choose the first legal slot/action")
            potion_overrides += 1
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
    expected_variant_overrides = (
        campfire_overrides
        if candidate_policy in {H4_POLICY, H5_POLICY}
        else potion_overrides
    )
    if candidate and expected_variant_overrides != expected_overrides:
        raise EvaluationIntegrityError("trace override count differs from the run summary")
    if not candidate and (potion_overrides or campfire_overrides):
        raise EvaluationIntegrityError("parent run unexpectedly used a candidate intervention")
    return {
        "combat_decisions": combat_decisions,
        "low_hp_potion_opportunities": low_hp_potion_opportunities,
        "campfire_overheal_opportunities": campfire_overheal_opportunities,
        "campfire_overheal_overrides": campfire_overrides,
        "overrides": expected_variant_overrides,
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
                "campfire_policy_intervention": event.get("campfire_policy_intervention"),
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
    candidate_policy: str = H3_POLICY,
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
        use_potion_below_hp_fraction=(
            HP_RATIO_THRESHOLD if candidate and candidate_policy == H3_POLICY else None
        ),
        avoid_low_hp_elite_routes=False,
        prefer_smith_when_rest_overheals=(candidate and candidate_policy == H4_POLICY),
        prefer_smith_when_overheal_exceeds_effective_rest_heal=(
            candidate and candidate_policy == H5_POLICY
        ),
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
            expected_overrides=int(result.get(_variant_override_count_key(candidate_policy), 0)),
            candidate_policy=candidate_policy,
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
    parser.add_argument(
        "--candidate-policy",
        choices=(H3_POLICY, H4_POLICY, H5_POLICY),
        default=H3_POLICY,
    )
    parser.add_argument("--candidate-commit")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        candidate_mode = args.candidate_policy
        if candidate_mode == H4_POLICY:
            pool_file = H4_POOL_FILE
            pool_key = "train_hypothesis_1"
            stage_name = "train_hypothesis_1_h4_campfire_overheal"
            intervention_description = (
                "choose legal Smith only when ArmG recommends Rest and the exact native Rest heal would exceed max HP"
            )
        elif candidate_mode == H5_POLICY:
            pool_file = H5_POOL_FILE
            pool_key = "train_hypothesis_1"
            stage_name = "train_hypothesis_1_h5_effective_rest_heal"
            intervention_description = (
                "choose legal Smith only when ArmG recommends Rest and heal overflow exceeds the effective capped Rest heal"
            )
        else:
            pool_file = H3_POOL_FILE
            pool_key = "train_hypothesis_3"
            stage_name = "train_hypothesis_3"
            intervention_description = (
                "at HP/maxHP <= 0.5, use the lowest-slot legal potion; preserve the parent MCTS action when required data or a legal potion is unavailable"
            )
        registration = _public_pool_registration(candidate_mode, pool_key)
        pool_id = registration["pool_id"]
        expected_pool_sha256 = registration["manifest_sha256"]
        pool, seeds, preflight = _validate_pool(
            args.pool_file.resolve(),
            args.pools_dir.resolve(),
            args.exclusion_inventory.resolve(),
            pool_id=pool_id,
            pool_file=pool_file,
            expected_pool_sha256=expected_pool_sha256,
            pool_key=pool_key,
            inventory_hash_mode="raw" if candidate_mode in {H4_POLICY, H5_POLICY} else "canonical",
        )
        _check_private_paths(args.private_output_dir, args.private_usage_ledger, pool_id=pool_id)
        checkpoint_sha = _sha256(args.g7_checkpoint)
        if checkpoint_sha != G7_SHA256:
            raise EvaluationIntegrityError("checkpoint does not match the pinned G7 parent")
        if Path(str(args.g7_checkpoint) + ".adapter.pt").exists():
            raise EvaluationIntegrityError("evaluation requires the unmodified G7 parent")
        binding = args.module_dir / "slaythespire.cp312-win_amd64.pyd"
        if _sha256(binding) != SIMULATOR_BINDING_SHA256:
            raise EvaluationIntegrityError("native simulator binding hash mismatch")
        simulator_source_sha = _sha256(Path(simulator_module.__file__).resolve())
        if simulator_source_sha != SIMULATOR_PYTHON_SOURCE_SHA256:
            raise EvaluationIntegrityError("simulator Python source differs from the frozen H5 build")
        armg_source = args.armg_root / "armG_train.py"
        if not armg_source.is_file():
            raise EvaluationIntegrityError("pinned ArmG source file is missing")
        if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
            raise EvaluationIntegrityError("contextual reranking must be disabled for fixed paired comparisons")

        candidate_commit = args.candidate_commit
        if not args.preflight_only:
            if not isinstance(candidate_commit, str) or len(candidate_commit) != 40:
                raise EvaluationIntegrityError("run requires the frozen 40-character candidate commit SHA")
            observed_commit = subprocess.check_output(
                ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
            if observed_commit != candidate_commit:
                raise EvaluationIntegrityError("local HEAD differs from the frozen candidate commit SHA")

        sts = _load_sts(args.module_dir)
        parent_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.g7_checkpoint)
        candidate_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.g7_checkpoint)
        metadata = {
            "stage": stage_name,
            "candidate_policy": candidate_mode,
            "pool_id": pool_id,
            "pool_manifest_sha256": expected_pool_sha256,
            "simulator_commit": SIMULATOR_COMMIT,
            "simulator_python_source_sha256": simulator_source_sha,
            "runner_source_sha256": _sha256(Path(__file__).resolve()),
            "candidate_git_commit_sha": candidate_commit or "NOT_COMMITTED_PREFLIGHT",
            "simulator_binding_sha256": SIMULATOR_BINDING_SHA256,
            "armg_source_sha256": _sha256(armg_source),
            "g7_checkpoint_sha256": checkpoint_sha,
            "mcts_sims": MCTS_SIMS,
            "candidate_intervention": intervention_description,
            "tie_break": (
                "native option 0 REST to native option 1 SMITH only when overheal exceeds effective capped Rest healing"
                if candidate_mode == H5_POLICY
                else "native option 0 REST to native option 1 SMITH only when exact Rest heal overflows max HP"
                if candidate_mode == H4_POLICY
                else "lowest_potion_slot_then_native_legal_action_order"
            ),
        }
        if args.preflight_only:
            print(json.dumps({"preflight": "PASS", **preflight, **metadata}, sort_keys=True))
            return 0

        args.private_output_dir.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        shared._append_jsonl(args.private_usage_ledger, {
            "pool_id": pool_id,
            "manifest_sha256": expected_pool_sha256,
            "stage": stage_name,
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
                print(f"{stage_name} pair {pair_index + 1}/10: parent")
                parent_result, _, _, parent_trace_stats = _run_one(
                    sts=sts, policy=parent_policy, seed=seed, all_training_seeds=seeds,
                    output_dir=args.private_output_dir, pair_index=pair_index, arm="parent",
                    candidate=False, trace_enabled=True, metadata=metadata,
                    candidate_policy=candidate_mode,
                )
                completed_episodes += 1
                print(f"{stage_name} pair {pair_index + 1}/10: candidate")
                candidate_result, candidate_evidence, _, candidate_trace_stats = _run_one(
                    sts=sts, policy=candidate_policy, seed=seed, all_training_seeds=seeds,
                    output_dir=args.private_output_dir, pair_index=pair_index, arm="candidate",
                    candidate=True, trace_enabled=True, metadata=metadata,
                    candidate_policy=candidate_mode,
                )
                completed_episodes += 1
                trace_invariance = None
                if pair_index == 0:
                    print(f"{stage_name} tracing invariance check: candidate with trace disabled")
                    trace_off_result, trace_off_evidence, _, _ = _run_one(
                        sts=sts, policy=candidate_policy, seed=seed, all_training_seeds=seeds,
                        output_dir=args.private_output_dir, pair_index=pair_index, arm="candidate",
                        candidate=True, trace_enabled=False, metadata=metadata,
                        candidate_policy=candidate_mode,
                    )
                    completed_episodes += 1
                    if (
                        trace_off_result.get("outcome") != candidate_result.get("outcome")
                        or _summary_view(trace_off_result) != _summary_view(candidate_result)
                        or _action_signature(trace_off_evidence) != _action_signature(candidate_evidence)
                    ):
                        raise EvaluationIntegrityError("trace on/off changed the fixed-seed candidate trajectory")
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

            paired = _paired_summary_for_stage(parent_outcomes, candidate_outcomes)
            override_key = (
                "campfire_overheal_override_count"
                if candidate_mode in {H4_POLICY, H5_POLICY}
                else "emergency_potion_override_count"
            )
            candidate_overrides = sum(int(row["candidate"].get(override_key) or 0) for row in rows)
            low_hp_opportunities = sum(int(row["candidate_trace"]["low_hp_potion_opportunities"]) for row in rows)
            campfire_opportunities = sum(
                int(row["candidate_trace"]["campfire_overheal_opportunities"])
                for row in rows
            )
            final = {
                "schema_version": (
                    "sts1-g7-h5-train-paired-v1"
                    if candidate_mode == H5_POLICY
                    else "sts1-g7-h4-train-paired-v1"
                    if candidate_mode == H4_POLICY
                    else "sts1-g7-h3-train-paired-v1"
                ),
                "status": "COMPLETE",
                "candidate_policy": candidate_mode,
                "pool_id": pool_id,
                "pool_manifest_sha256": expected_pool_sha256,
                "paired_seed_count": 10,
                "episode_count": completed_episodes,
                "trace_invariance": "PASS",
                "preflight": preflight,
                "identities": metadata,
                "paired": paired,
                "candidate_combat_decisions": sum(int(row["candidate_trace"]["combat_decisions"]) for row in rows),
                "communication_errors": "N/A_LOCAL_SIMULATOR",
            }
            if candidate_mode in {H4_POLICY, H5_POLICY}:
                final["candidate_campfire_overheal_opportunities"] = campfire_opportunities
                final["candidate_campfire_overheal_overrides"] = candidate_overrides
            else:
                final["candidate_low_hp_potion_opportunities"] = low_hp_opportunities
                final["candidate_emergency_potion_overrides"] = candidate_overrides
            (args.private_output_dir / "paired-summary.json").write_text(
                json.dumps(final, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            shared._append_jsonl(args.private_usage_ledger, {
                "pool_id": pool_id,
                "manifest_sha256": expected_pool_sha256,
                "stage": stage_name,
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
                **(
                    {
                        "candidate_campfire_overheal_opportunities": campfire_opportunities,
                        "candidate_campfire_overheal_overrides": candidate_overrides,
                    }
                    if candidate_mode in {H4_POLICY, H5_POLICY}
                    else {
                        "candidate_low_hp_potion_opportunities": low_hp_opportunities,
                        "candidate_emergency_potion_overrides": candidate_overrides,
                    }
                ),
            }, sort_keys=True))
            return 0
        except BaseException as exc:
            shared._append_jsonl(args.private_usage_ledger, {
                "pool_id": pool_id,
                "manifest_sha256": expected_pool_sha256,
                "stage": stage_name,
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
