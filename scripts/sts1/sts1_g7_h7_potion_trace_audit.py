"""Run the preregistered, trace-only H7 audit on Round005 train_hypothesis_3.

This is a diagnostic of unchanged G7. It does not train, tune, or evaluate a
candidate and it never prints or publishes individual seed IDs or raw traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sts1"))

from roguelike_ai.sts1_phase3.simulator import (  # noqa: E402
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)
from sts1_g7_seed_ledger import MAX_SEED, SeedLedgerError, sha256_json, validate_inventory  # noqa: E402
from sts1_g7_h2_elite_route_eval import (  # noqa: E402
    EvaluationIntegrityError,
    _action_signature,
    _check_evidence,
    _check_terminal_trace,
    _read_jsonl,
    _summary_view,
)


POOL_ID = "round-005-20261009-train_hypothesis_3"
POOL_KEY = "train_hypothesis_3"
POOL_FILE = "train_hypothesis_3.json"
EXPECTED_POOL_SHA256 = "9e088016ba055d58a45df132ed20423fcea8becc3677aa04613794dae64e64cc"
ROUND_POOL_FILES = {
    "train_hypothesis_1": "train_hypothesis_1.json",
    "train_hypothesis_2": "train_hypothesis_2.json",
    "train_hypothesis_3": "train_hypothesis_3.json",
    "probe": "probe.json",
    "dev": "dev.json",
}
G7_SHA256 = "8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0"
SIMULATOR_SOURCE_SHA256 = "92b641f3d3568d5fec0cc09744976b151d96bc62a368a0a942ee1f7ef49a288e"
SIMULATOR_BINDING_SHA256 = "bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e"
SIMULATOR_SOURCE_COMMIT = "7476a81954020087da31d41d16fddf475746ec2d"
SIM_RL_HOOKS_PATCH_SHA256 = "83d3a89ba0f7639e93f35df6b8f27bf2fe6326a8"
GAMECONTEXT_CLONE_PATCH_SHA256 = "fce772bd04b7b04d5e574ea7de252cec17056fb0ef846665da19fa5380d327dd"
H7_POTION_PATCH_SHA256 = "b04c47034e5908f9d906b55656d7fed9732f81cd2200802291d5fbc99b281d02"
ARMG_SOURCE_SHA256 = "7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b"
ARMG_VOCAB_SHA256 = "832e199c359af8408ea430ffa3f9fcdc68f32533f7292bb102848d3fb558eb6a"
EXPECTED_SUMMARY_SHA256 = "7de490c94c6487f309064e86b6bd5db3d968b8203253d28590d78084cfac4870"
EXPECTED_SUMMARY_SHA256_BEFORE_H7_RECOVERY = "9f923376c9e3ee23565bfc0812b6e0a2728a92aff0921bee357cc3336847deb5"
MCTS_SIMS = 2000
MAX_EPISODE_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_ARTIFACT_BYTES = 512 * 1024 * 1024
POTION_SOURCE = "native_gamecontext_potion_enum_names_v1"
INVALID_POTION_NAMES = {"INVALID", "UNKNOWN"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationIntegrityError("JSON root must be an object")
    return value


def _inventory_sha256(path: Path) -> str:
    return _file_sha256(path)


def _validate_pool(
    pool_path: Path,
    pools_dir: Path,
    inventory_path: Path,
    public_summary_path: Path,
    *,
    resume_partial: bool = False,
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    if pool_path.name != POOL_FILE:
        raise EvaluationIntegrityError("only the preregistered H7 Round005 train pool is accepted")
    pool = _read_json(pool_path)
    if pool.get("pool_id") != POOL_ID or pool.get("purpose") != "train":
        raise EvaluationIntegrityError("H7 pool identity or purpose mismatch")
    if pool.get("status") != "GENERATED_NOT_RUN":
        raise EvaluationIntegrityError("H7 pool is not marked unused")
    recorded_hash = pool.get("manifest_sha256")
    payload = {key: value for key, value in pool.items() if key != "manifest_sha256"}
    if recorded_hash != EXPECTED_POOL_SHA256 or sha256_json(payload) != recorded_hash:
        raise EvaluationIntegrityError("H7 pool manifest hash mismatch")
    seed_values = pool.get("seed_ids")
    if not isinstance(seed_values, list) or len(seed_values) != 10:
        raise EvaluationIntegrityError("H7 train pool must contain exactly ten seeds")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or not 1 <= seed <= MAX_SEED
        for seed in seed_values
    ):
        raise EvaluationIntegrityError("H7 pool contains an invalid simulator seed")
    seeds = tuple(seed_values)
    if len(set(seeds)) != 10:
        raise EvaluationIntegrityError("H7 train pool contains duplicate seeds")

    inventory = _read_json(inventory_path)
    excluded, source_audit = validate_inventory(inventory)
    inventory_hash = _inventory_sha256(inventory_path)
    source_audit_hash = sha256_json(source_audit)
    if (
        pool.get("inventory_id") != inventory.get("inventory_id")
        or pool.get("inventory_sha256") != inventory_hash
        or pool.get("source_audit_sha256") != source_audit_hash
        or pool.get("source_audit") != source_audit
    ):
        raise EvaluationIntegrityError("H7 pool does not match the verified exclusion inventory")
    if set(seeds) & excluded:
        raise EvaluationIntegrityError("H7 train pool overlaps the existing exclusion inventory")

    ledger = _read_json(pools_dir / "ledger.json")
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if sha256_json(ledger_payload) != ledger.get("ledger_sha256"):
        raise EvaluationIntegrityError("Round005 seed ledger hash mismatch")
    ledger_pools = ledger.get("pools", {})
    if not isinstance(ledger_pools, dict) or set(ledger_pools) != set(ROUND_POOL_FILES):
        raise EvaluationIntegrityError("Round005 ledger pools are missing")

    all_seen: set[int] = set()
    for pool_key, filename in ROUND_POOL_FILES.items():
        other = _read_json(pools_dir / filename)
        other_payload = {key: value for key, value in other.items() if key != "manifest_sha256"}
        other_hash = other.get("manifest_sha256")
        other_seeds = other.get("seed_ids")
        expected_purpose = "train" if pool_key.startswith("train_") else pool_key
        ledger_pool = ledger_pools.get(pool_key)
        if (
            not isinstance(other_seeds, list)
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or not 1 <= seed <= MAX_SEED
                for seed in other_seeds
            )
            or len(set(other_seeds)) != len(other_seeds)
            or sha256_json(other_payload) != other_hash
            or not isinstance(ledger_pool, dict)
            or ledger_pool.get("manifest_sha256") != other_hash
            or ledger_pool.get("seed_ids") != other_seeds
            or other.get("purpose") != expected_purpose
            or other.get("inventory_id") != inventory.get("inventory_id")
            or other.get("inventory_sha256") != inventory_hash
            or other.get("source_audit_sha256") != source_audit_hash
            or other.get("source_audit") != source_audit
            or set(other_seeds) & excluded
            or set(other_seeds) & all_seen
        ):
            raise EvaluationIntegrityError("Round005 ledger, pool, or disjointness audit failed")
        all_seen.update(other_seeds)

    public_summary = _read_json(public_summary_path)
    if (
        public_summary.get("round_id") != "round-005-20261009"
        or public_summary.get("manifest_sha256") != ledger.get("ledger_sha256")
        or public_summary.get("source_audit_sha256") != source_audit_hash
    ):
        raise EvaluationIntegrityError("public Round005 summary does not match the private ledger")
    public_pools = public_summary.get("pools", {})
    if not isinstance(public_pools, dict) or set(public_pools) != set(ROUND_POOL_FILES):
        raise EvaluationIntegrityError("public Round005 pool registration is incomplete")
    for pool_key, filename in ROUND_POOL_FILES.items():
        item = _read_json(pools_dir / filename)
        public_item = public_pools.get(pool_key)
        if (
            not isinstance(public_item, dict)
            or public_item.get("manifest_sha256") != item.get("manifest_sha256")
            or public_item.get("count") != len(item.get("seed_ids", []))
            or public_item.get("purpose") != item.get("purpose")
        ):
            raise EvaluationIntegrityError("public Round005 pool registration differs from private manifests")
    usage = public_summary.get("evaluation_usage")
    if not isinstance(usage, list):
        raise EvaluationIntegrityError("public Round005 evaluation usage is invalid")
    h7_usage = [
        record for record in usage
        if isinstance(record, dict)
        and (record.get("pool_key") == POOL_KEY or record.get("pool_manifest_sha256") == EXPECTED_POOL_SHA256)
    ]
    if resume_partial:
        if (
            len(h7_usage) != 2
            or any(record.get("status") != "NOT_VERIFIED" for record in h7_usage)
            or any(record.get("episode_count") != 1 for record in h7_usage)
            or any(record.get("error_class") != "EvaluationIntegrityError" for record in h7_usage)
            or _file_sha256(public_summary_path) != EXPECTED_SUMMARY_SHA256_BEFORE_H7_RECOVERY
        ):
            raise EvaluationIntegrityError("H7 recovery does not match the two recorded validator failures")
    elif h7_usage or _file_sha256(public_summary_path) != EXPECTED_SUMMARY_SHA256:
        raise EvaluationIntegrityError("H7 pool has prior usage or public summary changed since preregistration")

    return pool, seeds, {
        "train_seed_count": len(seeds),
        "exclusion_seed_count": len(excluded),
        "pool_manifest_count": len(ROUND_POOL_FILES),
        "inventory_sha256": inventory_hash,
        "inventory_hash_mode": "raw",
        "round005_pairwise_disjoint": True,
        "disjoint_from_exclusion_inventory": True,
        "public_summary_sha256_before_run": _file_sha256(public_summary_path),
        "prior_h7_usage_record_count": len(h7_usage),
    }


def _read_usage_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise EvaluationIntegrityError("usage ledger contains a non-object entry")
        rows.append(row)
    return rows


def _check_private_paths(
    output_dir: Path,
    usage_ledger: Path,
    usage_history: Iterable[Path],
    *,
    resume_partial: bool = False,
) -> None:
    repo = REPO_ROOT.resolve()
    for path in (output_dir.resolve(), usage_ledger.resolve()):
        if path == repo or repo in path.parents:
            raise EvaluationIntegrityError("raw H7 evidence must stay outside the repository")
    if resume_partial:
        if not output_dir.is_dir() or not usage_ledger.is_file():
            raise EvaluationIntegrityError("H7 partial recovery files are missing")
        pool_rows = [row for row in _read_usage_ledger(usage_ledger) if row.get("pool_id") == POOL_ID]
        if (
            [row.get("status") for row in pool_rows]
            != ["RUNNING", "NOT_VERIFIED", "RESUMED_AFTER_VALIDATOR_FIX", "NOT_VERIFIED"]
            or pool_rows[1].get("completed_episodes") != 1
            or pool_rows[1].get("error_class") != "EvaluationIntegrityError"
            or pool_rows[3].get("completed_episodes") != 1
            or pool_rows[3].get("error_class") != "EvaluationIntegrityError"
        ):
            raise EvaluationIntegrityError("H7 private recovery ledger is not the expected two validator failures")
        expected_files = {
            "audit-summary-private.json",
            "audit-summary-resume-failure-private.json",
            "episode-01.evidence.ndjson",
            "episode-01.trace.ndjson",
        }
        actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
        if actual_files != expected_files:
            raise EvaluationIntegrityError("H7 recovery output contains missing or unexpected files")
        if not (REPO_ROOT / "evidence/sts1/g7-improvement/round-005-h7-trace-audit-report.md").is_file():
            raise EvaluationIntegrityError("the first H7 NOT_VERIFIED report is missing")
    else:
        if output_dir.exists():
            raise EvaluationIntegrityError("private output directory already exists; do not rerun this pool")
        if usage_ledger.exists():
            raise EvaluationIntegrityError("new H7 private usage ledger path already exists")
    for ledger_path in usage_history:
        for row in _read_usage_ledger(ledger_path):
            if row.get("pool_id") == POOL_ID or row.get("manifest_sha256") == EXPECTED_POOL_SHA256:
                raise EvaluationIntegrityError("H7 train pool already has a private usage record")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _valid_potion_snapshot(state: Any, where: str, *, source: str = POTION_SOURCE) -> None:
    if not isinstance(state, dict):
        raise EvaluationIntegrityError(f"{where} has no diagnostic state object")
    slots = state.get("potions")
    if (
        state.get("potion_inventory_complete") is not True
        or state.get("potion_inventory_source") != source
        or state.get("potion_inventory_reason") != "exactly_five_valid_native_slots"
        or not isinstance(slots, list)
        or len(slots) != 5
        or any(not isinstance(slot, str) or slot in INVALID_POTION_NAMES for slot in slots)
    ):
        raise EvaluationIntegrityError(f"{where} potion-slot inventory is incomplete or invalid")


def _valid_run_snapshot(state: Any, where: str) -> None:
    _valid_potion_snapshot(state, where)
    if not isinstance(state.get("deck"), list) or not isinstance(state.get("relics"), list):
        raise EvaluationIntegrityError(f"{where} deck or relic snapshot is incomplete")


def _validate_trace(path: Path, result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, int]:
    events = list(_read_jsonl(path))
    headers = [event for event in events if event.get("type") == "diagnostic_trace_header_v1"]
    terminals = [event for event in events if event.get("type") == "terminal_trace_v1"]
    if (
        len(headers) != 1
        or headers[0].get("trace_schema") != "sts1-diagnostic-trace-v1"
        or headers[0].get("run_metadata") != metadata
        or len(terminals) != 1
    ):
        raise EvaluationIntegrityError("trace provenance header or unique terminal record is missing")

    combat_count = 0
    live_resources_required = headers[0].get("live_combat_resources_required") is True
    encounter_count = 0
    noncombat_count = 0
    route_count = 0
    for event in events:
        event_type = event.get("type")
        if event_type == "encounter_started_v1":
            encounter_count += 1
            state = event.get("state")
            if not isinstance(state, dict):
                raise EvaluationIntegrityError("encounter snapshot is missing")
            _valid_run_snapshot(state.get("run"), "encounter")
        elif event_type == "combat_decision_trace_v1":
            combat_count += 1
            if event.get("legal_actions_complete") is not True or event.get("mcts_sims") != MCTS_SIMS:
                raise EvaluationIntegrityError("combat legal actions or MCTS-2000 record is incomplete")
            native_actions = event.get("canonical_native_legal_actions")
            policy_actions = event.get("policy_legal_actions")
            native_index = event.get("selected_native_action_index")
            public_index = event.get("selected_public_action_index")
            if (
                not isinstance(native_actions, list)
                or not native_actions
                or any(not isinstance(action, dict) for action in native_actions)
                or not isinstance(policy_actions, list)
                or not policy_actions
                or any(not isinstance(action, dict) for action in policy_actions)
                or not isinstance(native_index, int)
                or isinstance(native_index, bool)
                or not 0 <= native_index < len(native_actions)
                or (
                    public_index is not None
                    and (
                        not isinstance(public_index, int)
                        or isinstance(public_index, bool)
                        or not 0 <= public_index < len(policy_actions)
                    )
                )
            ):
                raise EvaluationIntegrityError("combat legal-action provenance or selection is invalid")
            _valid_potion_snapshot(event.get("public_state"), "combat decision")
            if live_resources_required:
                state = event["public_state"]
                if state.get("potion_inventory_scope") != "run_context":
                    raise EvaluationIntegrityError("combat run inventory has no explicit scope")
                _valid_potion_snapshot(
                    state.get("combat_potion_inventory"), "live combat resources",
                    source="native_battlecontext_potion_enum_names_v1",
                )
        elif event_type == "noncombat_decision_trace_v1":
            noncombat_count += 1
            if event.get("legal_choices_complete") is not True or not isinstance(event.get("legal_choices"), list):
                raise EvaluationIntegrityError("noncombat legal choices are incomplete")
            for key in ("state_before", "state_after"):
                _valid_run_snapshot(event.get(key), f"noncombat {key}")
            if event.get("screen_before") == "MAP_SCREEN":
                route_count += 1

    terminal = terminals[0]
    final_state = terminal.get("final_state")
    if (
        terminal.get("complete") is not True
        or terminal.get("outcome") not in {"victory", "defeat"}
        or terminal.get("error") is not None
        or terminal.get("legal_actions_complete") is not True
        or terminal.get("illegal_action_count") != 0
        or terminal.get("timeout_count") != 0
        or terminal.get("crash_count") != 0
        or terminal.get("potion_inventory_snapshot_complete") is not True
        or terminal.get("potion_inventory_source") != POTION_SOURCE
        or terminal.get("potion_inventory_reason") != "exactly_five_valid_native_slots"
        or not isinstance(final_state, dict)
    ):
        raise EvaluationIntegrityError("terminal completeness, safety, or potion snapshot failed")
    _valid_run_snapshot(final_state, "terminal")
    if result.get("outcome") != terminal.get("outcome"):
        raise EvaluationIntegrityError("returned run outcome differs from terminal trace")
    if combat_count < 1 or encounter_count < 1 or noncombat_count < 1 or route_count < 1:
        raise EvaluationIntegrityError("bounded trace lacks combat, encounter, noncombat, or route coverage")

    # Reuse the existing route/terminal validator for the complete legal map-edge mapping.
    _check_terminal_trace(path, candidate=False, expected_overrides=0, expected_fail_closed=0)
    return {
        "combat_decision_count": combat_count,
        "encounter_count": encounter_count,
        "noncombat_decision_count": noncombat_count,
        "route_decision_count": route_count,
        "potion_snapshot_count": combat_count + encounter_count + 2 * noncombat_count + 1,
    }


def _validate_partial_first_episode(
    output_dir: Path,
    *,
    expected_seed: int,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    failure = _read_json(output_dir / "audit-summary-private.json")
    if failure.get("status") != "NOT_VERIFIED" or failure.get("episode_count") != 1:
        raise EvaluationIntegrityError("partial H7 record is not the expected one-episode failure")
    evidence_path = output_dir / "episode-01.evidence.ndjson"
    trace_path = output_dir / "episode-01.trace.ndjson"
    evidence = list(_read_jsonl(evidence_path))
    summary = next((event for event in evidence if event.get("type") == "summary"), None)
    header = next(
        (event for event in _read_jsonl(trace_path) if event.get("type") == "diagnostic_trace_header_v1"),
        None,
    )
    if (
        not isinstance(summary, dict)
        or summary.get("simulator_seed_long") != expected_seed
        or not evidence
        or evidence[0].get("simulator_seed_long") != expected_seed
        or not isinstance(header, dict)
        or header.get("simulator_seed_long") != expected_seed
    ):
        raise EvaluationIntegrityError("retained H7 episode is not the first registered train seed")
    _check_evidence(evidence_path, summary, candidate=False)
    trace_stats = _validate_trace(trace_path, summary, metadata)
    if evidence_path.stat().st_size > MAX_EPISODE_ARTIFACT_BYTES or trace_path.stat().st_size > MAX_EPISODE_ARTIFACT_BYTES:
        raise EvaluationIntegrityError("retained H7 evidence exceeds the per-episode bound")
    return summary, trace_stats


def _artifact_inventory(output_dir: Path) -> tuple[list[dict[str, Any]], int, str]:
    items = []
    total_bytes = 0
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact-manifest.json":
            continue
        size = path.stat().st_size
        total_bytes += size
        items.append({
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": size,
            "sha256": _file_sha256(path),
        })
    manifest = {"schema_version": "sts1-g7-h7-private-artifact-manifest-v1", "files": items}
    digest = sha256_json(manifest)
    return items, total_bytes, digest


def _summary_usage_bytes(
    summary_path: Path,
    usage: dict[str, Any],
    *,
    expected_sha256: str = EXPECTED_SUMMARY_SHA256,
) -> bytes:
    original = summary_path.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise EvaluationIntegrityError("public Round005 summary changed before append")
    text = original.decode("utf-8")
    marker = '"evaluation_usage": ['
    if text.count(marker) != 1:
        raise EvaluationIntegrityError("public summary evaluation_usage array is not unique")
    start = text.index(marker) + len(marker)
    depth = 1
    in_string = False
    escaped = False
    end = -1
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < 0:
        raise EvaluationIntegrityError("public summary evaluation_usage array is unterminated")
    insertion = end
    while insertion > start and text[insertion - 1].isspace():
        insertion -= 1
    has_records = bool(text[start:insertion].strip())
    rendered = json.dumps(usage, ensure_ascii=False, sort_keys=True, indent=2)
    rendered = "\n".join("    " + line for line in rendered.splitlines())
    prefix = "," if has_records else ""
    updated = text[:insertion] + prefix + "\n" + rendered + text[insertion:]
    parsed = json.loads(updated)
    if parsed["evaluation_usage"][-1] != usage:
        raise EvaluationIntegrityError("public summary append did not validate")
    return updated.encode("utf-8")


def _append_public_summary(summary_path: Path, updated: bytes) -> str:
    temp_path = summary_path.with_name(summary_path.name + ".h7-tmp")
    if temp_path.exists():
        raise EvaluationIntegrityError("public summary temporary path already exists")
    with temp_path.open("xb") as handle:
        handle.write(updated)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, summary_path)
    return hashlib.sha256(updated).hexdigest()


def _write_report(path: Path, result: dict[str, Any], *, append_existing: bool = False) -> None:
    if path.exists() and not append_existing:
        raise EvaluationIntegrityError("H7 report path already exists")
    if append_existing and (
        not path.is_file()
        or "NOT_VERIFIED" not in path.read_text(encoding="utf-8")
        or "episodes completed: `1`" not in path.read_text(encoding="utf-8")
    ):
        raise EvaluationIntegrityError("existing H7 report is not the expected preserved first-attempt failure")
    trace = result.get("trace_summary", {})
    heading = "## Recovery result after validator repair" if append_existing else "# H7 potion-inventory trace audit — diagnostic only"
    lines = [
        heading,
        "",
        f"- Status: `{result['status']}`.",
        f"- Date: `{result['finished_at_utc']}`.",
        f"- Scope: unchanged G7 on 10 registered Round005 train-only seeds, plus one trace-off replay ({result['episode_count']} episodes).",
        *(["- Recovery: the first attempt stopped after one complete episode because the validator incorrectly required deck/relics on a combat-only snapshot. The validator was corrected and that retained episode was revalidated; its seed was not rerun."] if append_existing else []),
        f"- Pool manifest SHA-256: `{EXPECTED_POOL_SHA256}`; seed IDs and raw traces remain private.",
        f"- G7 checkpoint SHA-256: `{G7_SHA256}`; simulator source SHA-256: `{SIMULATOR_SOURCE_SHA256}`; native binding SHA-256: `{SIMULATOR_BINDING_SHA256}`; MCTS budget: `{MCTS_SIMS}`.",
        f"- Native simulator commit: `{SIMULATOR_SOURCE_COMMIT}`; ArmG source SHA-256: `{ARMG_SOURCE_SHA256}`; ArmG vocabulary SHA-256: `{ARMG_VOCAB_SHA256}`.",
        f"- Trace-on outcomes (diagnostic baseline only): victories `{result['trace_on_outcomes'].get('victory', 0)}`, defeats `{result['trace_on_outcomes'].get('defeat', 0)}`; no candidate comparison or win-rate claim.",
        f"- Trace coverage: combat decisions `{trace.get('combat_decision_count', 0)}`, encounters `{trace.get('encounter_count', 0)}`, noncombat decisions `{trace.get('noncombat_decision_count', 0)}`, map routes `{trace.get('route_decision_count', 0)}`, complete potion snapshots `{trace.get('potion_snapshot_count', 0)}`.",
        f"- Trace-on/off invariance: `{result['trace_invariance']}`; legal terminal traces `{result['complete_legal_terminal_traces']}`; illegal actions/crashes/timeouts `{result['illegal_actions']}/{result['crashes']}/{result['timeouts']}`; communication errors `{result['communication_errors']}`.",
        f"- Private artifact inventory: `{result['artifact_file_count']}` files, `{result['artifact_bytes']}` bytes; manifest SHA-256 `{result['private_artifact_manifest_sha256']}`.",
        f"- Tests: focused H7 trace tests and existing simulator trace tests passed before the diagnostic run.",
        "- Probe10, Dev30, confirmation, Gate/Fresh, real-game verification, promotion, and deployment: `NOT_RUN`.",
        "- G7 remains Champion; this audit made no policy or checkpoint changes.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if append_existing:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n" + "\n".join(lines))
    else:
        path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--pools-dir", type=Path, required=True)
    parser.add_argument("--exclusion-inventory", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, default=REPO_ROOT / "evidence/sts1/g7-improvement/seeds/round-005-20261009/summary.json")
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--g7-checkpoint", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--private-usage-ledger", type=Path, required=True)
    parser.add_argument("--usage-history", type=Path, action="append", default=[])
    parser.add_argument("--report-file", type=Path, default=REPO_ROOT / "evidence/sts1/g7-improvement/round-005-h7-trace-audit-report.md")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-partial", action="store_true", help="continue only the recorded one-episode H7 validator failure without rerunning that seed")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        pool, seeds, preflight = _validate_pool(
            args.pool_file.resolve(),
            args.pools_dir.resolve(),
            args.exclusion_inventory.resolve(),
            args.public_summary.resolve(),
            resume_partial=args.resume_partial,
        )
        _check_private_paths(
            args.private_output_dir,
            args.private_usage_ledger,
            [path.resolve() for path in args.usage_history],
            resume_partial=args.resume_partial,
        )
        checkpoint_sha = _file_sha256(args.g7_checkpoint)
        if checkpoint_sha != G7_SHA256:
            raise EvaluationIntegrityError("local checkpoint does not match the pinned G7 hash")
        if Path(str(args.g7_checkpoint) + ".adapter.pt").exists():
            raise EvaluationIntegrityError("G7 checkpoint has an adapter sidecar")
        binding = args.module_dir / "slaythespire.cp312-win_amd64.pyd"
        if _file_sha256(binding) != SIMULATOR_BINDING_SHA256:
            raise EvaluationIntegrityError("native simulator binding hash mismatch")
        simulator_source = REPO_ROOT / "src" / "roguelike_ai" / "sts1_phase3" / "simulator.py"
        if _file_sha256(simulator_source) != SIMULATOR_SOURCE_SHA256:
            raise EvaluationIntegrityError("pinned H7 simulator source hash mismatch")
        armg_source = args.armg_root / "armG_train.py"
        vocab_source = args.armg_root / "armS_card_vocab.json"
        if _file_sha256(armg_source) != ARMG_SOURCE_SHA256 or _file_sha256(vocab_source) != ARMG_VOCAB_SHA256:
            raise EvaluationIntegrityError("pinned ArmG code or vocabulary hash mismatch")
        if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
            raise EvaluationIntegrityError("contextual Teacher reranking must be disabled")
        h7_patch = REPO_ROOT / "patches" / "sts1" / "g7_h7_potion_inventory_trace.patch"
        clone_patch = REPO_ROOT / "patches" / "sts1" / "armg_gamecontext_clone.patch"
        if _file_sha256(h7_patch) != H7_POTION_PATCH_SHA256 or _file_sha256(clone_patch) != GAMECONTEXT_CLONE_PATCH_SHA256:
            raise EvaluationIntegrityError("pinned read-only simulator binding patch hash mismatch")
        identities = {
            "stage": "h7_potion_inventory_trace_audit",
            "pool_id": POOL_ID,
            "pool_manifest_sha256": EXPECTED_POOL_SHA256,
            "simulator_source_sha256": SIMULATOR_SOURCE_SHA256,
            "simulator_source_commit": SIMULATOR_SOURCE_COMMIT,
            "sim_rl_hooks_patch_sha256": SIM_RL_HOOKS_PATCH_SHA256,
            "gamecontext_clone_patch_sha256": GAMECONTEXT_CLONE_PATCH_SHA256,
            "h7_potion_binding_patch_sha256": H7_POTION_PATCH_SHA256,
            "simulator_binding_sha256": SIMULATOR_BINDING_SHA256,
            "armg_source_sha256": ARMG_SOURCE_SHA256,
            "armg_vocab_sha256": ARMG_VOCAB_SHA256,
            "g7_checkpoint_sha256": checkpoint_sha,
            "mcts_sims": MCTS_SIMS,
            "policy_change": "none; unchanged G7 diagnostic only",
        }
        if args.preflight_only:
            print(json.dumps({"preflight": "PASS", **preflight, **identities}, sort_keys=True))
            return 0
    except Exception as exc:
        print(f"PREFLIGHT_FAILED: {type(exc).__name__}")
        return 2

    output_dir = args.private_output_dir.resolve()
    usage_ledger = args.private_usage_ledger.resolve()
    audit_attempt = 1 + int(preflight.get("prior_h7_usage_record_count", 0))
    completed_episodes = 1 if args.resume_partial else 0
    public_summary_recorded = False
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        if args.resume_partial:
            _append_jsonl(usage_ledger, {
                "pool_id": POOL_ID,
                "manifest_sha256": EXPECTED_POOL_SHA256,
                "stage": "h7_potion_inventory_trace_audit",
                "status": "RESUMED_AFTER_VALIDATOR_FIX",
                "started_at_utc": started_at,
                "already_completed_episodes": 1,
                "remaining_trace_on_episodes": 9,
                "remaining_trace_off_invariance_replays": 1,
            })
        else:
            output_dir.mkdir(parents=True, exist_ok=False)
            _append_jsonl(usage_ledger, {
                "pool_id": POOL_ID,
                "manifest_sha256": EXPECTED_POOL_SHA256,
                "stage": "h7_potion_inventory_trace_audit",
                "status": "RUNNING",
                "started_at_utc": started_at,
                "expected_episodes": 11,
                "trace_on_episodes": 10,
                "trace_off_invariance_replays": 1,
            })
        sts = _load_sts(args.module_dir.resolve())
        policy = ArmGNoncombatPolicy(root=args.armg_root.resolve(), weight_path=args.g7_checkpoint.resolve())
        rows: list[dict[str, Any]] = []
        outcomes: list[str] = []
        aggregate_trace = {
            "combat_decision_count": 0,
            "encounter_count": 0,
            "noncombat_decision_count": 0,
            "route_decision_count": 0,
            "potion_snapshot_count": 0,
        }
        outcomes: list[str] = []
        first_evidence = output_dir / "episode-01.evidence.ndjson"
        first_result: dict[str, Any] | None = None
        start_index = 0

        if args.resume_partial:
            first_result, first_trace_stats = _validate_partial_first_episode(
                output_dir,
                expected_seed=seeds[0],
                metadata=identities,
            )
            rows.append({
                "episode_index": 1,
                "seed_id": seeds[0],
                "trace_mode": "on_preserved_from_attempt_1",
                "outcome": first_result.get("outcome"),
                "summary": _summary_view(first_result),
                "trace": first_trace_stats,
            })
            outcomes.append(str(first_result["outcome"]))
            for key, count in first_trace_stats.items():
                aggregate_trace[key] += count
            (output_dir / "recovery-state-private.json").write_text(
                json.dumps({
                    "schema_version": "sts1-g7-h7-recovery-state-v1",
                    "recovery_reason": "validator_required_deck_relics_on_combat_snapshot",
                    "seed_index": 1,
                    "seed_id": seeds[0],
                    "preserved_episode_validated": True,
                    "evidence_sha256": _file_sha256(first_evidence),
                    "trace_sha256": _file_sha256(output_dir / "episode-01.trace.ndjson"),
                    "trace_stats": first_trace_stats,
                }, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print("H7 trace invariance replay for preserved episode 1 (trace disabled)")
            trace_off_evidence = output_dir / "episode-01-trace-off.evidence.ndjson"
            if trace_off_evidence.exists():
                raise EvaluationIntegrityError("trace-off replay already exists; refusing to rerun it")
            trace_off_metadata = {**identities, "episode_index": 1, "seed_id": seeds[0]}
            trace_off_result = run_simulator_game(
                student=None,
                sts=sts,
                seed=seeds[0],
                evidence_path=trace_off_evidence,
                armg_policy=policy,
                combat_mcts_sims=MCTS_SIMS,
                reserve_last_potion_until_floor=None,
                avoid_low_hp_elite_routes=False,
                training_seeds=seeds,
                collect_ppo=False,
                collect_teacher=False,
                diagnostic_trace_path=None,
                diagnostic_metadata=trace_off_metadata,
            )
            completed_episodes += 1
            _check_evidence(trace_off_evidence, trace_off_result, candidate=False)
            if (
                _summary_view(trace_off_result) != _summary_view(first_result)
                or _action_signature(trace_off_evidence) != _action_signature(first_evidence)
            ):
                raise EvaluationIntegrityError("trace-on/off changed the fixed-seed G7 trajectory")
            rows.append({
                "episode_index": 1,
                "seed_id": seeds[0],
                "trace_mode": "off_invariance_replay_after_recovery",
                "outcome": trace_off_result.get("outcome"),
                "summary": _summary_view(trace_off_result),
            })
            if trace_off_evidence.stat().st_size > MAX_EPISODE_ARTIFACT_BYTES:
                raise EvaluationIntegrityError("trace-off replay evidence exceeded the per-episode bound")
            start_index = 1

        for index in range(start_index, len(seeds)):
            seed = seeds[index]
            print(f"H7 diagnostic trace-on episode {index + 1}/10")
            evidence_path = output_dir / f"episode-{index + 1:02d}.evidence.ndjson"
            trace_path = output_dir / f"episode-{index + 1:02d}.trace.ndjson"
            if evidence_path.exists() or trace_path.exists():
                raise EvaluationIntegrityError("an H7 episode already exists; refusing to rerun a seed")
            episode_metadata = {**identities, "episode_index": index + 1, "seed_id": seed}
            result = run_simulator_game(
                student=None,
                sts=sts,
                seed=seed,
                evidence_path=evidence_path,
                armg_policy=policy,
                combat_mcts_sims=MCTS_SIMS,
                reserve_last_potion_until_floor=None,
                avoid_low_hp_elite_routes=False,
                training_seeds=seeds,
                collect_ppo=False,
                collect_teacher=False,
                diagnostic_trace_path=trace_path,
                diagnostic_metadata=episode_metadata,
            )
            completed_episodes += 1
            _check_evidence(evidence_path, result, candidate=False)
            trace_stats = _validate_trace(trace_path, result, episode_metadata)
            for key, count in trace_stats.items():
                aggregate_trace[key] += count
            outcomes.append(str(result["outcome"]))
            rows.append({
                "episode_index": index + 1,
                "seed_id": seed,
                "trace_mode": "on",
                "outcome": result.get("outcome"),
                "summary": _summary_view(result),
                "trace": trace_stats,
            })
            if index == 0:
                first_result = result
                print("H7 trace invariance replay (trace disabled)")
                trace_off_evidence = output_dir / "episode-01-trace-off.evidence.ndjson"
                if trace_off_evidence.exists():
                    raise EvaluationIntegrityError("trace-off replay already exists; refusing to rerun it")
                trace_off_metadata = {**identities, "episode_index": 1, "seed_id": seed}
                trace_off_result = run_simulator_game(
                    student=None,
                    sts=sts,
                    seed=seed,
                    evidence_path=trace_off_evidence,
                    armg_policy=policy,
                    combat_mcts_sims=MCTS_SIMS,
                    reserve_last_potion_until_floor=None,
                    avoid_low_hp_elite_routes=False,
                    training_seeds=seeds,
                    collect_ppo=False,
                    collect_teacher=False,
                    diagnostic_trace_path=None,
                    diagnostic_metadata=trace_off_metadata,
                )
                completed_episodes += 1
                _check_evidence(trace_off_evidence, trace_off_result, candidate=False)
                if (
                    _summary_view(trace_off_result) != _summary_view(first_result)
                    or _action_signature(trace_off_evidence) != _action_signature(first_evidence)
                ):
                    raise EvaluationIntegrityError("trace-on/off changed the fixed-seed G7 trajectory")
                rows.append({
                    "episode_index": 1,
                    "seed_id": seed,
                    "trace_mode": "off_invariance_replay",
                    "outcome": trace_off_result.get("outcome"),
                    "summary": _summary_view(trace_off_result),
                })

            for artifact in (evidence_path, trace_path):
                if artifact.stat().st_size > MAX_EPISODE_ARTIFACT_BYTES:
                    raise EvaluationIntegrityError("per-episode raw trace/evidence size exceeded the bound")
            if sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file()) > MAX_TOTAL_ARTIFACT_BYTES:
                raise EvaluationIntegrityError("private H7 artifact size exceeded the total bound")

        trace_on_outcomes = {"victory": outcomes.count("victory"), "defeat": outcomes.count("defeat")}
        private_summary_path = output_dir / (
            "audit-summary-resumed-private.json" if args.resume_partial else "audit-summary-private.json"
        )
        private_summary_path.write_text(
            json.dumps({
                "schema_version": "sts1-g7-h7-private-audit-v1",
                "status": "COMPLETE",
                "audit_attempt": audit_attempt,
                "recovered_after_validator_fix": args.resume_partial,
                "identities": identities,
                "preflight": preflight,
                "episode_count": completed_episodes,
                "trace_on_outcomes": trace_on_outcomes,
                "trace_invariance": "PASS",
                "trace_summary": aggregate_trace,
                "episodes": rows,
            }, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_files, artifact_bytes, artifact_manifest_sha = _artifact_inventory(output_dir)
        private_manifest = {
            "schema_version": "sts1-g7-h7-private-artifact-manifest-v1",
            "files": artifact_files,
        }
        (output_dir / "artifact-manifest.json").write_text(
            json.dumps(private_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        finished_at = datetime.now(timezone.utc).isoformat()
        result_record = {
            "status": "COMPLETE_DIAGNOSTIC_ONLY_AFTER_VALIDATOR_REPAIR" if args.resume_partial else "COMPLETE_DIAGNOSTIC_ONLY",
            "finished_at_utc": finished_at,
            "episode_count": completed_episodes,
            "trace_on_outcomes": trace_on_outcomes,
            "trace_invariance": "PASS",
            "complete_legal_terminal_traces": 10,
            "illegal_actions": 0,
            "crashes": 0,
            "timeouts": 0,
            "communication_errors": "N/A_LOCAL_SIMULATOR",
            "trace_summary": aggregate_trace,
            "artifact_file_count": len(artifact_files),
            "artifact_bytes": artifact_bytes,
            "private_artifact_manifest_sha256": artifact_manifest_sha,
        }
        public_usage = {
            "audit_attempt": audit_attempt,
            "audit_type": "trace_only_unchanged_g7",
            "candidate_policy": "none_diagnostic_only",
            "communication_errors": "N/A_LOCAL_SIMULATOR",
            "complete_legal_terminal_traces": 10,
            "crashes": 0,
            "episode_count": completed_episodes,
            "illegal_actions": 0,
            "pool_key": POOL_KEY,
            "pool_manifest_sha256": EXPECTED_POOL_SHA256,
            "status": result_record["status"],
            "timeouts": 0,
            "trace_invariance": "PASS",
            "trace_on_episode_count": 10,
            "trace_off_invariance_replays": 1,
            "trace_summary": aggregate_trace,
            "trace_on_outcomes": trace_on_outcomes,
            "usage_ledger_status": "COMPLETE",
            "private_artifact_file_count": len(artifact_files),
            "private_artifact_bytes": artifact_bytes,
            "private_artifact_manifest_sha256": artifact_manifest_sha,
        }
        if args.resume_partial:
            public_usage["previous_attempt_status"] = "NOT_VERIFIED"
            public_usage["recovery_reason"] = "validator_schema_assumption_fixed; preserved episode revalidated without rerun"
        expected_summary_sha = (
            EXPECTED_SUMMARY_SHA256_BEFORE_H7_RECOVERY if args.resume_partial else EXPECTED_SUMMARY_SHA256
        )
        summary_bytes = _summary_usage_bytes(
            args.public_summary.resolve(), public_usage, expected_sha256=expected_summary_sha
        )
        result_record["public_summary_sha256_after_append"] = hashlib.sha256(summary_bytes).hexdigest()
        report_path = args.report_file.resolve()
        _write_report(report_path, result_record, append_existing=args.resume_partial)
        _append_public_summary(args.public_summary.resolve(), summary_bytes)
        public_summary_recorded = True
        _append_jsonl(usage_ledger, {
            "pool_id": POOL_ID,
            "manifest_sha256": EXPECTED_POOL_SHA256,
            "stage": "h7_potion_inventory_trace_audit",
            "status": result_record["status"],
            "finished_at_utc": finished_at,
            "episode_count": completed_episodes,
            "trace_invariance": "PASS",
            "private_artifact_manifest_sha256": artifact_manifest_sha,
        })
        print(json.dumps({
            "status": result_record["status"],
            "episode_count": completed_episodes,
            "trace_on_outcomes": trace_on_outcomes,
            "trace_invariance": "PASS",
            "trace_summary": aggregate_trace,
            "safety": {"illegal_actions": 0, "crashes": 0, "timeouts": 0, "communication_errors": "N/A_LOCAL_SIMULATOR"},
            "private_artifact_file_count": len(artifact_files),
            "private_artifact_bytes": artifact_bytes,
            "private_artifact_manifest_sha256": artifact_manifest_sha,
            "public_summary_sha256_after_append": result_record["public_summary_sha256_after_append"],
        }, sort_keys=True))
        return 0
    except BaseException as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        failure = {
            "status": "NOT_VERIFIED",
            "finished_at_utc": finished_at,
            "episode_count": completed_episodes,
            "error_class": type(exc).__name__,
            "trace_invariance": "NOT_VERIFIED",
        }
        if output_dir.exists():
            failure_name = "audit-summary-resume-failure-private.json" if args.resume_partial else "audit-summary-private.json"
            failure_path = output_dir / failure_name
            if not failure_path.exists():
                failure_path.write_text(
                    json.dumps({"schema_version": "sts1-g7-h7-private-audit-v1", **failure}, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
        if not public_summary_recorded:
            try:
                failure_usage = {
                    "audit_attempt": audit_attempt,
                    "audit_type": "trace_only_unchanged_g7",
                    "candidate_policy": "none_diagnostic_only",
                    "communication_errors": "NOT_VERIFIED",
                    "episode_count": completed_episodes,
                    "pool_key": POOL_KEY,
                    "pool_manifest_sha256": EXPECTED_POOL_SHA256,
                    "status": "NOT_VERIFIED",
                    "trace_invariance": "NOT_VERIFIED",
                    "usage_ledger_status": "NOT_VERIFIED",
                    "error_class": type(exc).__name__,
                }
                expected_summary_sha = (
                    EXPECTED_SUMMARY_SHA256_BEFORE_H7_RECOVERY if args.resume_partial else EXPECTED_SUMMARY_SHA256
                )
                summary_bytes = _summary_usage_bytes(
                    args.public_summary.resolve(), failure_usage, expected_sha256=expected_summary_sha
                )
                _append_public_summary(args.public_summary.resolve(), summary_bytes)
                public_summary_recorded = True
            except Exception:
                pass
        try:
            report_path = args.report_file.resolve()
            if report_path.exists() and args.resume_partial:
                with report_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(
                        "\n## H7 recovery attempt NOT_VERIFIED\n\n"
                        f"- Episodes completed across attempts: `{completed_episodes}`. Failure class: `{type(exc).__name__}`.\n"
                        "- All prior files were retained; no completed seed was rerun.\n"
                    )
            elif not report_path.exists():
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    "# H7 potion-inventory trace audit — NOT_VERIFIED\n\n"
                    f"- Stage: bounded diagnostic only; episodes completed: `{completed_episodes}`.\n"
                    f"- Failure class: `{type(exc).__name__}`. Individual seed IDs and raw evidence remain private.\n"
                    "- No win-rate conclusion, candidate, Probe10, Dev30, or confirmation result is claimed.\n"
                    "- G7 remains Champion.\n",
                    encoding="utf-8",
                    newline="\n",
                )
        except Exception:
            pass
        try:
            if usage_ledger.exists():
                _append_jsonl(usage_ledger, {
                    "pool_id": POOL_ID,
                    "manifest_sha256": EXPECTED_POOL_SHA256,
                    "stage": "h7_potion_inventory_trace_audit",
                    "status": "NOT_VERIFIED",
                    "finished_at_utc": finished_at,
                    "completed_episodes": completed_episodes,
                    "error_class": type(exc).__name__,
                })
        except Exception:
            pass
        print(f"NOT_VERIFIED: stopped after {completed_episodes} episodes ({type(exc).__name__}); private evidence retained")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
