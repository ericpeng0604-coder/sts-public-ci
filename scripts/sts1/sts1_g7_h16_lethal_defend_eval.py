"""Run the preregistered H16/H17 Round008 and H18 Round009 Defend evaluations.

Raw seed IDs, episodes, traces, and usage records must remain in the private
Temp evaluation directory. This runner never tunes the registered rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sts1"))

from roguelike_ai.sts1_phase3.simulator import (  # noqa: E402
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)
import sts1_g7_h2_elite_route_eval as h2  # noqa: E402
import sts1_g7_h3_emergency_potion_eval as h3  # noqa: E402
import sts1_g7_h7_potion_trace_audit as h7  # noqa: E402
import sts1_g7_h15_train_trace_audit as h15  # noqa: E402
import sts1_g7_seed_ledger as seed_ledger  # noqa: E402

EvaluationIntegrityError = h2.EvaluationIntegrityError
ROUND_ID = "round-008-20261009"
ROUND009_ID = "round-009-20261009"
TRIAL_ROUND_IDS = {"h16": ROUND_ID, "h17": ROUND_ID, "h18": ROUND009_ID}
TRIAL_POOL_ROLE = {
    "h16": {"train": "train_hypothesis_2", "probe": "probe", "dev": "dev"},
    "h17": {"train": "train_hypothesis_3", "probe": "probe", "dev": "dev"},
    "h18": {"train": "train_hypothesis_1", "probe": "probe", "dev": "dev"},
}
STAGE_CONFIG = {
    "train": {"paired_seed_count": 10, "minimum_override_seed_coverage": 3},
    "probe": {"paired_seed_count": 10, "minimum_override_seed_coverage": 3},
    "dev": {"paired_seed_count": 30, "minimum_override_seed_coverage": 0},
    "confirmation_a": {"paired_seed_count": 100, "minimum_override_seed_coverage": 0},
    "confirmation_b": {"paired_seed_count": 100, "minimum_override_seed_coverage": 0},
}
POOL_COUNTS = {
    stage: config["paired_seed_count"] for stage, config in STAGE_CONFIG.items()
}
POOL_FILES = {
    "train_hypothesis_1": "train_hypothesis_1.json",
    "train_hypothesis_2": "train_hypothesis_2.json",
    "train_hypothesis_3": "train_hypothesis_3.json",
    "probe": "probe.json",
    "dev": "dev.json",
}
# Private seed-pool and historical summary pins are stored only in the local usage ledger.
EXPECTED_H17_ALLOCATION = {
    "record_type": "h17_pool_allocation",
    "trial_id": "h17",
    "round_id": ROUND_ID,
    "prior_h16": {
        "train": {"pool_id": "round-008-20261009-train_hypothesis_2", "status": "COMPLETE"},
        "probe": {"pool_id": "round-008-20261009-probe", "status": "NOT_RUN"},
        "dev": {"pool_id": "round-008-20261009-dev", "status": "NOT_RUN"},
    },
    "assigned_h17": {
        "train": {"pool_id": "round-008-20261009-train_hypothesis_3"},
        "probe": {"pool_id": "round-008-20261009-probe"},
        "dev": {"pool_id": "round-008-20261009-dev"},
    },
}
EXPECTED_H18_ALLOCATION = {
    "record_type": "h18_pool_allocation",
    "trial_id": "h18",
    "round_id": ROUND009_ID,
    "prior_h16": {
        "train": {
            "pool_id": "round-008-20261009-train_hypothesis_2",
            "status": "NOT_VERIFIED_IMPLEMENTATION_COVERAGE",
        },
        "probe": {"pool_id": "round-008-20261009-probe", "status": "NOT_RUN"},
        "dev": {"pool_id": "round-008-20261009-dev", "status": "NOT_RUN"},
    },
    "prior_h17": {
        "train": {
            "pool_id": "round-008-20261009-train_hypothesis_3",
            "status": "NOT_VERIFIED_IMPLEMENTATION_COVERAGE",
        },
        "probe": {"pool_id": "round-008-20261009-probe", "status": "NOT_RUN"},
        "dev": {"pool_id": "round-008-20261009-dev", "status": "NOT_RUN"},
    },
    "exclusion_inventory": {
        "inventory_id": "round-009-20261009-extended-exclusion",
        "source_count": 86,
        "unique_id_count": 23106,
    },
    "assigned_h18": {
        "train": {"pool_id": "round-009-20261009-train_hypothesis_1"},
        "probe": {"pool_id": "round-009-20261009-probe"},
        "dev": {"pool_id": "round-009-20261009-dev"},
    },
    "reserved_unused_train_pools": [
        {"pool_id": "round-009-20261009-train_hypothesis_2"},
        {"pool_id": "round-009-20261009-train_hypothesis_3"},
    ],
}

def _without_private_hashes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_private_hashes(item)
            for key, item in value.items()
            if key != "sha256" and not key.lower().endswith("_sha256")
        }
    if isinstance(value, list):
        return [_without_private_hashes(item) for item in value]
    return value
G7_SHA256 = "8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0"
PINNED_GAMEPLAY_COMMIT = "7476a81954020087da31d41d16fddf475746ec2d"
PINNED_GAMEPLAY_REPOSITORY_URL = "https://github.com/gamerpuppy/sts_lightspeed"
PINNED_BINDING_SHA256 = "bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e"
ARMG_SHA256 = "7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b"
ARMG_VOCAB_SHA256 = "832e199c359af8408ea430ffa3f9fcdc68f32533f7292bb102848d3fb558eb6a"
MCTS_SIMS = h7.MCTS_SIMS
MAX_EPISODE_BYTES = h7.MAX_EPISODE_ARTIFACT_BYTES
MAX_TOTAL_BYTES = h7.MAX_TOTAL_ARTIFACT_BYTES
TRACE_COVERAGE_FIELDS = (
    "combat_decision_count",
    "encounter_count",
    "noncombat_decision_count",
    "route_decision_count",
    "potion_snapshot_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise EvaluationIntegrityError("private H16 usage record is not an object")
            rows.append(value)
    return rows


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_selected_seeds(stage: str, seeds: tuple[Any, ...] | list[Any]) -> None:
    config = STAGE_CONFIG.get(stage)
    if not isinstance(config, dict):
        raise EvaluationIntegrityError("stage has no registered paired-count configuration")
    expected_count = config["paired_seed_count"]
    if (
        not isinstance(seeds, (tuple, list))
        or len(seeds) != expected_count
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise EvaluationIntegrityError("selected paired seed pool has invalid count, IDs, or duplicates")


def _stage_seed_roles(stage: str, stage_seeds: tuple[int, ...]) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    _validate_selected_seeds(stage, stage_seeds)
    if stage == "train":
        return stage_seeds, None
    if stage in {"probe", "dev", "confirmation_a", "confirmation_b"}:
        return None, stage_seeds
    raise EvaluationIntegrityError("stage has no registered seed role")


def _round008_assets(
    *, trial_id: str, stage: str, pool_file: Path, pools_dir: Path, inventory_path: Path
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    if trial_id not in TRIAL_POOL_ROLE or stage not in TRIAL_POOL_ROLE[trial_id]:
        raise EvaluationIntegrityError("H16/H17 trial or stage is not registered")
    role = TRIAL_POOL_ROLE[trial_id][stage]
    expected_pool = (pools_dir / POOL_FILES[role]).resolve()
    if pool_file.resolve() != expected_pool:
        raise EvaluationIntegrityError("selected H16/H17 pool path does not match the registered stage")

    # H15's audited loader validates the full Round008 ledger, all five pool
    # manifests, the ID-only exclusion inventory, and pairwise disjointness.
    # It requires the H1 path as its anchor; the active trial selects H2 or H3
    # and never executes H1 for this validation step.
    _, validated_pools, round_preflight = h15._validate_round008_assets(
        pools_dir / "train_hypothesis_1.json", pools_dir, inventory_path
    )
    pool = h15._read_json(expected_pool)
    seeds = validated_pools.get(role)
    if not isinstance(seeds, tuple):
        raise EvaluationIntegrityError("selected H16/H17 pool count or validation is invalid")
    _validate_selected_seeds(stage, seeds)
    preflight = {
        **round_preflight,
        "pool_id": pool.get("pool_id"),
        "pool_manifest_sha256": pool.get("manifest_sha256"),
        "stage": stage,
        "seed_count": len(seeds),
        "seed_disjointness_verified": True,
    }
    return pool, seeds, preflight


def _round009_assets(
    *,
    trial_id: str,
    stage: str,
    pool_file: Path,
    pools_dir: Path,
    inventory_path: Path,
    allocation_event: dict[str, Any] | None,
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any], dict[str, Any]]:
    if trial_id != "h18" or stage not in TRIAL_POOL_ROLE["h18"]:
        raise EvaluationIntegrityError("Round009 assets are registered only for H18")
    if allocation_event is None and stage != "train":
        raise EvaluationIntegrityError("H18 held-out stages require a pre-existing private allocation record")
    role = TRIAL_POOL_ROLE[trial_id][stage]
    expected_pool = (pools_dir / POOL_FILES[role]).resolve()
    if pool_file.resolve() != expected_pool:
        raise EvaluationIntegrityError("selected H18 pool path does not match the registered stage")

    inventory = h15._read_json(inventory_path)
    excluded, source_audit = seed_ledger.validate_inventory(inventory)
    inventory_sha256 = _sha256(inventory_path)
    if (
        inventory.get("inventory_id") != EXPECTED_H18_ALLOCATION["exclusion_inventory"]["inventory_id"]
        or len(source_audit) != 86
        or len(excluded) != 23106
    ):
        raise EvaluationIntegrityError("Round009 exclusion inventory identity or coverage mismatch")

    ledger_path = pools_dir / "ledger.json"
    ledger = h15._read_json(ledger_path)
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if (
        ledger.get("schema_version") != seed_ledger.SCHEMA_VERSION
        or ledger.get("round_id") != ROUND009_ID
        or ledger.get("inventory_id") != inventory.get("inventory_id")
        or ledger.get("inventory_sha256") != inventory_sha256
        or ledger.get("source_audit_sha256") != seed_ledger.sha256_json(source_audit)
        or ledger.get("status") != "GENERATED_NOT_RUN"
        or ledger.get("ledger_sha256") != seed_ledger.sha256_json(ledger_payload)
        or set(ledger.get("pools", {})) != set(POOL_FILES)
    ):
        raise EvaluationIntegrityError("Round009 seed-ledger identity or provenance mismatch")

    pool_manifests: dict[str, dict[str, Any]] = {}
    for pool_name, expected_count in seed_ledger.EXPLORATION_POOL_SIZES.items():
        manifest = h15._read_json(pools_dir / POOL_FILES[pool_name])
        manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if (
            manifest != ledger["pools"].get(pool_name)
            or manifest.get("manifest_sha256") != seed_ledger.sha256_json(manifest_payload)
            or manifest.get("purpose") != ("train" if pool_name.startswith("train_") else pool_name)
            or len(manifest.get("seed_ids", [])) != expected_count
        ):
            raise EvaluationIntegrityError("Round009 pool manifest does not match its ledger entry")
        pool_manifests[pool_name] = manifest
    seed_ledger._verify_generated(ledger, excluded)

    if allocation_event is None:
        allocation_event = deepcopy(EXPECTED_H18_ALLOCATION)
        allocation_event["exclusion_inventory"]["sha256"] = inventory_sha256
        allocation_event["round009_ledger_sha256"] = ledger["ledger_sha256"]
        allocation_event["assigned_h18"]["train"]["manifest_sha256"] = pool_manifests[
            "train_hypothesis_1"
        ]["manifest_sha256"]
        allocation_event["assigned_h18"]["probe"]["manifest_sha256"] = pool_manifests["probe"][
            "manifest_sha256"
        ]
        allocation_event["assigned_h18"]["dev"]["manifest_sha256"] = pool_manifests["dev"][
            "manifest_sha256"
        ]
        for reserved, pool_name in zip(
            allocation_event["reserved_unused_train_pools"],
            ("train_hypothesis_2", "train_hypothesis_3"),
            strict=True,
        ):
            reserved["manifest_sha256"] = pool_manifests[pool_name]["manifest_sha256"]
    if _without_private_hashes(allocation_event) != EXPECTED_H18_ALLOCATION:
        raise EvaluationIntegrityError("H18 private allocation does not match the registered safe structure")
    if (
        not _valid_sha256(allocation_event["exclusion_inventory"].get("sha256"))
        or allocation_event["exclusion_inventory"]["sha256"] != inventory_sha256
        or not _valid_sha256(allocation_event.get("round009_ledger_sha256"))
        or allocation_event["round009_ledger_sha256"] != ledger.get("ledger_sha256")
        or len(allocation_event.get("reserved_unused_train_pools", [])) != 2
    ):
        raise EvaluationIntegrityError("H18 private allocation pins do not match the generated seed ledger")
    allocated_pools = {
        "train_hypothesis_1": allocation_event["assigned_h18"]["train"],
        "probe": allocation_event["assigned_h18"]["probe"],
        "dev": allocation_event["assigned_h18"]["dev"],
        "train_hypothesis_2": allocation_event["reserved_unused_train_pools"][0],
        "train_hypothesis_3": allocation_event["reserved_unused_train_pools"][1],
    }
    for pool_name, manifest in pool_manifests.items():
        pin = allocated_pools[pool_name]
        if (
            pin.get("pool_id") != manifest.get("pool_id")
            or not _valid_sha256(pin.get("manifest_sha256"))
            or pin["manifest_sha256"] != manifest.get("manifest_sha256")
        ):
            raise EvaluationIntegrityError("H18 private pool allocation pin differs from its manifest")

    pool = h15._read_json(expected_pool)
    raw_seeds = pool.get("seed_ids")
    if not isinstance(raw_seeds, list):
        raise EvaluationIntegrityError("selected H18 pool seed IDs are unavailable")
    seeds = tuple(raw_seeds)
    _validate_selected_seeds(stage, seeds)
    preflight = {
        "round_id": ROUND009_ID,
        "pool_id": pool.get("pool_id"),
        "pool_manifest_sha256": pool.get("manifest_sha256"),
        "inventory_sha256": inventory_sha256,
        "ledger_sha256": ledger.get("ledger_sha256"),
        "source_count": len(source_audit),
        "unique_excluded_seed_count": len(excluded),
        "stage": stage,
        "seed_count": len(seeds),
        "seed_disjointness_verified": True,
    }
    return pool, seeds, preflight, allocation_event


def _validate_private_paths(
    pools_dir: Path, trial_id: str, stage: str, output_dir: Path, usage_path: Path
) -> None:
    if trial_id not in TRIAL_POOL_ROLE or stage not in TRIAL_POOL_ROLE[trial_id]:
        raise EvaluationIntegrityError("H16/H17 trial or stage is not registered")
    pools_dir = pools_dir.resolve()
    round_parts = TRIAL_ROUND_IDS[trial_id].split("-")
    round_prefix = "-".join(round_parts[:2])
    round_date = round_parts[2]
    expected_output = pools_dir.parent.parent / f"{round_prefix}-{trial_id}-{stage}-{round_date}"
    expected_usage = pools_dir.parent / f"{trial_id}-usage-private.jsonl"
    if output_dir.resolve() != expected_output.resolve() or usage_path.resolve() != expected_usage.resolve():
        raise EvaluationIntegrityError("trial requires canonical private output and usage paths")
    repo = REPO_ROOT.resolve()
    for path in (output_dir.resolve(), usage_path.resolve()):
        if path == repo or repo in path.parents:
            raise EvaluationIntegrityError("raw trial records must remain outside the repository")
    if output_dir.exists():
        raise EvaluationIntegrityError("stage output already exists; refusing to reuse it")


def _validate_pinned_gameplay_manifest(path: Path) -> str:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationIntegrityError("pinned simulator upstream manifest could not be read") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("repository") != PINNED_GAMEPLAY_REPOSITORY_URL
        or manifest.get("commit") != PINNED_GAMEPLAY_COMMIT
    ):
        raise EvaluationIntegrityError("upstream manifest does not match the pinned gameplay commit")
    return PINNED_GAMEPLAY_COMMIT


def _validate_identity(module_dir: Path, armg_root: Path, checkpoint: Path) -> dict[str, str]:
    simulator_source = REPO_ROOT / "src" / "roguelike_ai" / "sts1_phase3" / "simulator.py"
    simulator_upstream_manifest = REPO_ROOT / "external" / "sts_lightspeed" / "UPSTREAM.json"
    binding = module_dir / "slaythespire.cp312-win_amd64.pyd"
    armg_source = armg_root / "armG_train.py"
    armg_vocab = armg_root / "armS_card_vocab.json"
    expected = {
        "simulator_binding_sha256": (binding, PINNED_BINDING_SHA256),
        "armg_source_sha256": (armg_source, ARMG_SHA256),
        "armg_vocab_sha256": (armg_vocab, ARMG_VOCAB_SHA256),
        "g7_checkpoint_sha256": (checkpoint, G7_SHA256),
    }
    identities: dict[str, str] = {}
    for name, (path, digest) in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise EvaluationIntegrityError(f"pinned {name} identity mismatch")
        identities[name] = digest
    if Path(str(checkpoint) + ".adapter.pt").exists():
        raise EvaluationIntegrityError("pinned G7 checkpoint has an adapter sidecar")
    if not simulator_source.is_file():
        raise EvaluationIntegrityError("H16 simulator policy hook source is missing")
    if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
        raise EvaluationIntegrityError("contextual Teacher reranking must be disabled")
    identities["simulator_gameplay_commit"] = _validate_pinned_gameplay_manifest(
        simulator_upstream_manifest
    )
    identities["simulator_policy_source_sha256"] = _sha256(simulator_source)
    identities["candidate_evaluator_sha256"] = _sha256(Path(__file__).resolve())
    return identities


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["rtk", "git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationIntegrityError("local H16 candidate commit could not be verified") from exc


def _stage_summary(
    events: list[dict[str, Any]], stage: str, trial_id: str
) -> dict[str, Any] | None:
    rows = [
        event
        for event in events
        if event.get("record_type") == f"{trial_id}_stage_summary"
        and event.get("stage") == stage
        and event.get("trial_id", "h16" if trial_id == "h16" else None) == trial_id
    ]
    return rows[-1] if rows else None


def _validate_transition(
    *, trial_id: str, stage: str, events: list[dict[str, Any]], candidate_commit: str, identities: dict[str, str]
) -> None:
    if trial_id not in TRIAL_POOL_ROLE or stage not in TRIAL_POOL_ROLE[trial_id]:
        raise EvaluationIntegrityError("trial or stage is not registered")

    allocation_rows = [
        event
        for event in events
        if event.get("record_type") in {"h17_pool_allocation", "h18_pool_allocation"}
    ]
    if trial_id == "h17":
        if (
            len(allocation_rows) != 1
            or not events
            or _without_private_hashes(events[0]) != EXPECTED_H17_ALLOCATION
        ):
            raise EvaluationIntegrityError("H17 usage ledger lacks its exact first allocation record")
    elif trial_id == "h18":
        if (
            len(allocation_rows) != 1
            or not events
            or _without_private_hashes(events[0]) != EXPECTED_H18_ALLOCATION
        ):
            raise EvaluationIntegrityError("H18 usage ledger lacks its exact first allocation record")
    elif allocation_rows:
        raise EvaluationIntegrityError("H16 usage ledger contains a later-trial allocation record")

    stage_start_type = f"{trial_id}_stage_start"
    if any(
        event.get("record_type") == stage_start_type
        and event.get("trial_id", "h16" if trial_id == "h16" else None) == trial_id
        and event.get("stage") == stage
        for event in events
    ):
        raise EvaluationIntegrityError("stage has already been attempted; its pool cannot be rerun")
    if stage == "train":
        if trial_id in {"h17", "h18"}:
            expected_allocation = EXPECTED_H17_ALLOCATION if trial_id == "h17" else EXPECTED_H18_ALLOCATION
            if len(events) != 1 or _without_private_hashes(events[0]) != expected_allocation:
                raise EvaluationIntegrityError("train stage requires its exact first private allocation record")
        elif events:
            raise EvaluationIntegrityError("train stage requires its exact new private allocation record")
        return

    predecessor = "train" if stage == "probe" else "probe"
    prior = _stage_summary(events, predecessor, trial_id)
    if not isinstance(prior, dict) or prior.get("status") != "COMPLETE":
        raise EvaluationIntegrityError("required preceding stage is not complete")
    if prior.get("trial_id", "h16" if trial_id == "h16" else None) != trial_id:
        raise EvaluationIntegrityError("usage ledger belongs to a different trial")
    if prior.get("candidate_commit") != candidate_commit:
        raise EvaluationIntegrityError("candidate commit changed between stages")
    for key in (
        "simulator_policy_source_sha256",
        "candidate_evaluator_sha256",
        "simulator_binding_sha256",
        "armg_source_sha256",
        "armg_vocab_sha256",
        "g7_checkpoint_sha256",
        "simulator_gameplay_commit",
    ):
        if prior.get(key) != identities.get(key):
            raise EvaluationIntegrityError("policy, evaluator, simulator, or parent changed between stages")
    if prior.get("advance_eligible") is not True:
        raise EvaluationIntegrityError("preceding paired result did not pass its preregistered gate")


def _check_pair_integrity(path: Path, result: dict[str, Any]) -> None:
    h2._check_evidence(path, result, candidate=False)
    if result.get("outcome") not in {"victory", "defeat"} or result.get("error") is not None:
        raise EvaluationIntegrityError("H16 episode has no complete terminal outcome")
    for key in ("illegal_action_count", "timeout_count", "crash_count", "communication_error_count"):
        value = result.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise EvaluationIntegrityError("canonical safety counter was nonzero or unavailable")
    if result.get("potion_inventory_snapshot_complete") is not True:
        raise EvaluationIntegrityError("H16 complete potion inventory was not recorded")


SAFETY_COUNTERS = (
    "illegal_action_count",
    "crash_count",
    "timeout_count",
    "communication_error_count",
)


def _accumulate_safety(totals: dict[str, int], result: dict[str, Any]) -> None:
    for key in SAFETY_COUNTERS:
        value = result.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise EvaluationIntegrityError("canonical safety counter was nonzero or unavailable")
        totals[key] += value


def _empty_coverage() -> dict[str, int]:
    return {key: 0 for key in TRACE_COVERAGE_FIELDS}


def _accumulate_coverage(total: dict[str, int], stats: dict[str, int]) -> None:
    if set(stats) != set(TRACE_COVERAGE_FIELDS):
        raise EvaluationIntegrityError("H16 trace coverage fields are incomplete")
    for key, value in stats.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvaluationIntegrityError("H16 trace coverage counter is invalid")
        total[key] += value


def _trace_key(event: dict[str, Any]) -> tuple[int, int] | None:
    encounter = event.get("encounter_index")
    step = event.get("battle_step")
    if (
        not isinstance(encounter, int) or isinstance(encounter, bool)
        or not isinstance(step, int) or isinstance(step, bool)
    ):
        return None
    return encounter, step


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _pair_has_effective_override(
    parent_trace: dict[str, list[dict[str, Any]]],
    candidate_trace: dict[str, list[dict[str, Any]]],
) -> bool:
    parent_decisions = {
        _trace_key(event): event
        for event in parent_trace.get("decisions", [])
        if _trace_key(event) is not None
    }
    parent_applied = {
        _trace_key(event): event
        for event in parent_trace.get("applied", [])
        if _trace_key(event) is not None
    }
    candidate_applied = {
        _trace_key(event): event
        for event in candidate_trace.get("applied", [])
        if _trace_key(event) is not None
    }
    for candidate in candidate_trace.get("decisions", []):
        if candidate.get("lethal_intent_defend_override") is not True:
            continue
        key = _trace_key(candidate)
        parent = parent_decisions.get(key)
        parent_after = parent_applied.get(key)
        candidate_after = candidate_applied.get(key)
        before = candidate.get("state_before_signature_sha256")
        recommended = candidate.get("mcts_recommended_action")
        selected = candidate.get("selected_action")
        if (
            parent is None
            or parent_after is None
            or candidate_after is None
            or not _valid_sha256(before)
            or parent.get("state_before_signature_sha256") != before
            or parent.get("mcts_recommended_action") != recommended
            or not isinstance(selected, dict)
            or not isinstance(recommended, dict)
            or selected == recommended
            or parent.get("selected_action") != recommended
            or candidate_after.get("selected_action") != selected
            or parent_after.get("selected_action") != recommended
            or not _valid_sha256(candidate_after.get("state_after_signature_sha256"))
            or not _valid_sha256(parent_after.get("state_after_signature_sha256"))
            or candidate_after.get("state_after_signature_sha256")
            == parent_after.get("state_after_signature_sha256")
        ):
            continue
        return True
    return False


def _validate_action_trace_completeness(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    decisions = [event for event in events if event.get("type") == "combat_decision_trace_v1"]
    applied = [event for event in events if event.get("type") == "combat_action_applied_v1"]
    if len(decisions) != len(applied):
        raise EvaluationIntegrityError("combat decision and post-action traces are incomplete")
    decision_keys = [_trace_key(event) for event in decisions]
    applied_keys = [_trace_key(event) for event in applied]
    if (
        any(key is None for key in decision_keys + applied_keys)
        or len(set(decision_keys)) != len(decision_keys)
        or set(decision_keys) != set(applied_keys)
    ):
        raise EvaluationIntegrityError("combat action trace keys are missing, repeated, or unmatched")
    applied_by_key = {_trace_key(event): event for event in applied}
    for decision in decisions:
        key = _trace_key(decision)
        after = applied_by_key[key]
        if (
            not _valid_sha256(decision.get("state_before_signature_sha256"))
            or not _valid_sha256(after.get("state_after_signature_sha256"))
            or after.get("selected_action") != decision.get("selected_action")
        ):
            raise EvaluationIntegrityError("combat action trace lacks state signatures or matching action")
    return {"decisions": decisions, "applied": applied}


def _evidence_action_signature(path: Path) -> tuple[str, int]:
    actions: list[dict[str, Any]] = []
    for event in h7._read_jsonl(path):
        event_type = event.get("type")
        if event_type == "simulator_noncombat":
            if "choice" not in event:
                raise EvaluationIntegrityError("noncombat action signature is unavailable")
            actions.append({
                "type": event_type,
                "floor": event.get("floor"),
                "act": event.get("act"),
                "screen": event.get("screen"),
                "choice": event["choice"],
            })
        elif event_type == "simulator_combat_action":
            identity = {
                key: event.get(key)
                for key in (
                    "chosen_bits", "action_id", "native_action_index", "action_index",
                    "decision_signature",
                )
                if key in event
            }
            if not any(value is not None for value in identity.values()):
                raise EvaluationIntegrityError("combat action signature is unavailable")
            actions.append({
                "type": event_type,
                "floor": event.get("floor"),
                "policy": event.get("policy"),
                "mcts_sims": event.get("mcts_sims"),
                **identity,
            })
    if not actions:
        raise EvaluationIntegrityError("episode contains no auditable selected-action signatures")
    payload = json.dumps(actions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(actions)


def _trace_replay_matches(
    trace_on: dict[str, Any],
    trace_off: dict[str, Any],
    trace_on_signature: str,
    trace_off_signature: str,
    trace_on_action_count: int,
    trace_off_action_count: int,
) -> bool:
    return (
        h3._summary_view(trace_on) == h3._summary_view(trace_off)
        and trace_on_signature == trace_off_signature
        and trace_on_action_count == trace_off_action_count
        and trace_on.get("lethal_intent_defend_override_count")
        == trace_off.get("lethal_intent_defend_override_count")
    )


def _stage_gate(
    stage: str,
    paired: dict[str, Any],
    effective_override_seed_count: int,
    hard_guards_passed: bool,
) -> dict[str, Any]:
    config = STAGE_CONFIG.get(stage)
    if not isinstance(config, dict):
        raise EvaluationIntegrityError("stage has no registered advancement configuration")
    if not isinstance(hard_guards_passed, bool):
        raise EvaluationIntegrityError("hard safety and integrity gate is unavailable")
    coverage_required = config["minimum_override_seed_coverage"]
    coverage_passed = effective_override_seed_count >= coverage_required
    if stage in {"train", "probe"}:
        stage_signal = coverage_passed and paired.get("net_wins", -1) >= 0
    elif stage == "dev":
        stage_signal = paired.get("net_wins", -1) > 0
    else:
        stage_signal = False
    return {
        "coverage_required": coverage_required,
        "coverage_passed": coverage_passed,
        "stage_signal_passed": stage_signal,
        "hard_guards_passed": hard_guards_passed,
        "advance_eligible": bool(stage_signal and hard_guards_passed),
    }


def _delta_summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _terminal_floor_hp_diagnostic(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise EvaluationIntegrityError("terminal floor/HP diagnostics require complete pairs")
    relations = {
        "terminal_floor": Counter({"lower": 0, "equal": 0, "higher": 0}),
        "final_hp": Counter({"lower": 0, "equal": 0, "higher": 0}),
    }
    deltas = {"terminal_floor": [], "final_hp": []}
    grouped: dict[str, dict[str, list[int]]] = {
        name: {"terminal_floor": [], "final_hp": []}
        for name in ("both_defeat", "both_victory", "candidate_only", "parent_only")
    }
    group_counts: Counter[str] = Counter()
    for pair in pairs:
        parent_outcome = pair.get("parent_outcome")
        candidate_outcome = pair.get("candidate_outcome")
        if parent_outcome not in {"victory", "defeat"} or candidate_outcome not in {"victory", "defeat"}:
            raise EvaluationIntegrityError("terminal floor/HP pair has an unknown outcome")
        if parent_outcome == candidate_outcome:
            group = "both_victory" if parent_outcome == "victory" else "both_defeat"
        else:
            group = "candidate_only" if candidate_outcome == "victory" else "parent_only"
        group_counts[group] += 1
        for metric, parent_key, candidate_key in (
            ("terminal_floor", "parent_floor", "candidate_floor"),
            ("final_hp", "parent_hp", "candidate_hp"),
        ):
            parent_value = pair.get(parent_key)
            candidate_value = pair.get(candidate_key)
            if (
                not isinstance(parent_value, int) or isinstance(parent_value, bool)
                or not isinstance(candidate_value, int) or isinstance(candidate_value, bool)
            ):
                raise EvaluationIntegrityError("paired final floor or HP is unavailable")
            delta = candidate_value - parent_value
            relation = "lower" if delta < 0 else "higher" if delta > 0 else "equal"
            relations[metric][relation] += 1
            deltas[metric].append(delta)
            grouped[group][metric].append(delta)

    return {
        "purpose": "secondary_diagnostic_only",
        "blocking": False,
        "candidate_relation_counts": {
            metric: dict(sorted(counts.items())) for metric, counts in relations.items()
        },
        "paired_delta_summary": {
            metric: _delta_summary(values) for metric, values in deltas.items()
        },
        "by_paired_outcome": {
            group: {
                "pair_count": group_counts[group],
                "terminal_floor_delta": _delta_summary(values["terminal_floor"]),
                "final_hp_delta": _delta_summary(values["final_hp"]),
            }
            for group, values in grouped.items()
        },
    }


def _confirmation_alpha(trial_k: int, prior_trial_ks: list[int]) -> float:
    if (
        not isinstance(trial_k, int) or isinstance(trial_k, bool) or trial_k < 1
        or prior_trial_ks != list(range(1, trial_k))
    ):
        raise EvaluationIntegrityError("confirmation trial index is missing, repeated, or reset")
    return 0.05 / (2 ** trial_k)


def _confirmation_gate(
    batch_a: dict[str, Any],
    batch_b: dict[str, Any],
    *,
    trial_k: int,
    prior_trial_ks: list[int],
) -> dict[str, Any]:
    alpha = _confirmation_alpha(trial_k, prior_trial_ks)
    identity_keys = (
        "candidate_commit",
        "candidate_evaluator_sha256",
        "simulator_policy_source_sha256",
        "simulator_binding_sha256",
        "simulator_gameplay_commit",
        "armg_source_sha256",
        "armg_vocab_sha256",
        "g7_checkpoint_sha256",
    )
    batches = (batch_a, batch_b)
    if (
        batch_a.get("stage") != "confirmation_a"
        or batch_b.get("stage") != "confirmation_b"
        or any(batch.get("status") != "COMPLETE" for batch in batches)
        or any(batch.get("hard_guards_passed") is not True for batch in batches)
        or any(batch.get("seed_disjointness_verified") is not True for batch in batches)
        or any(batch.get("seed_count") != 100 for batch in batches)
        or any(not _valid_sha256(batch.get("pool_manifest_sha256")) for batch in batches)
        or batch_a.get("pool_manifest_sha256") == batch_b.get("pool_manifest_sha256")
        or any(batch_a.get(key) != batch_b.get(key) for key in identity_keys)
    ):
        raise EvaluationIntegrityError("confirmation batches are incomplete, unsafe, overlapping, or unfrozen")

    parent_outcomes: list[str] = []
    candidate_outcomes: list[str] = []
    for batch in batches:
        pairs = batch.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != 100:
            raise EvaluationIntegrityError("confirmation batch does not contain exactly 100 complete pairs")
        for pair in pairs:
            parent = pair.get("parent")
            candidate = pair.get("candidate")
            if not isinstance(parent, dict) or not isinstance(candidate, dict):
                raise EvaluationIntegrityError("confirmation pair is missing an arm")
            parent_outcome = parent.get("outcome")
            candidate_outcome = candidate.get("outcome")
            if parent_outcome not in {"victory", "defeat"} or candidate_outcome not in {"victory", "defeat"}:
                raise EvaluationIntegrityError("confirmation pair has an unknown terminal outcome")
            parent_outcomes.append(parent_outcome)
            candidate_outcomes.append(candidate_outcome)

    paired = h3._paired_summary_for_stage(parent_outcomes, candidate_outcomes)
    batch_nets = [
        h3._paired_summary_for_stage(
            [pair["parent"]["outcome"] for pair in batch["pairs"]],
            [pair["candidate"]["outcome"] for pair in batch["pairs"]],
        )["net_wins"]
        for batch in batches
    ]
    alpha_passed = paired["exact_one_sided_sign_p_candidate_positive"] <= alpha
    return {
        "trial_k": trial_k,
        "alpha_k": alpha,
        "batch_a_net": batch_nets[0],
        "batch_b_net": batch_nets[1],
        "combined": paired,
        "both_batches_positive_net": all(net > 0 for net in batch_nets),
        "minimum_combined_net_passed": paired["net_wins"] >= 10,
        "sign_test_passed": alpha_passed,
        "accepted": (
            all(net > 0 for net in batch_nets)
            and paired["net_wins"] >= 10
            and alpha_passed
        ),
    }


def _run_episode(
    *,
    sts: Any,
    policy: ArmGNoncombatPolicy,
    seed: int,
    stage_seeds: tuple[int, ...],
    output_dir: Path,
    pair_index: int,
    arm: str,
    candidate: bool,
    identity: dict[str, Any],
    trace_mode: str = "on",
    episode_index: int | None = None,
) -> tuple[
    dict[str, Any], Path, dict[str, int], int, dict[str, int],
    dict[str, list[dict[str, Any]]], str, int,
]:
    if trace_mode not in {"on", "off"}:
        raise EvaluationIntegrityError("episode trace mode is not registered")
    stem = f"pair-{pair_index:02d}-{arm}"
    evidence_path = output_dir / f"{stem}.evidence.ndjson"
    trace_path = output_dir / f"{stem}.trace.ndjson"
    if evidence_path.exists() or (trace_mode == "on" and trace_path.exists()):
        raise EvaluationIntegrityError("H16 episode output already exists; refusing to reuse a seed")
    training_seeds, heldout_seeds = _stage_seed_roles(identity["stage"], stage_seeds)
    metadata = {
        **identity,
        "stage": identity["stage"],
        "pair_index": pair_index,
        "episode_index": episode_index or (pair_index - 1) * 2 + (2 if candidate else 1),
        "arm": arm,
        "trace_mode": trace_mode,
        "seed_id": seed,
    }
    result = run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=evidence_path,
        armg_policy=policy,
        combat_mcts_sims=MCTS_SIMS,
        reserve_last_potion_until_floor=None,
        use_potion_below_hp_fraction=None,
        lethal_potion_rescue=False,
        lethal_intent_defend_rescue=candidate,
        avoid_low_hp_elite_routes=False,
        prefer_smith_when_rest_overheals=False,
        prefer_smith_when_overheal_exceeds_effective_rest_heal=False,
        training_seeds=training_seeds,
        heldout_seeds=heldout_seeds,
        collect_ppo=False,
        collect_teacher=False,
        diagnostic_trace_path=trace_path if trace_mode == "on" else None,
        diagnostic_metadata=metadata,
    )
    _check_pair_integrity(evidence_path, result)
    if result.get("lethal_intent_defend_rescue_enabled") is not candidate:
        raise EvaluationIntegrityError("H16 policy activation flag differs from the requested arm")
    if trace_mode == "on":
        trace_stats = h7._validate_trace(trace_path, result, metadata)
        events = list(h7._read_jsonl(trace_path))
        trace_payload = _validate_action_trace_completeness(events)
    else:
        trace_stats = _empty_coverage()
        events = []
        trace_payload = {"decisions": [], "applied": []}
    combat_events = [
        event for event in events if event.get("type") == "combat_decision_trace_v1"
    ]
    override_events = [event for event in combat_events if event.get("lethal_intent_defend_override") is True]
    if trace_mode == "on" and len(override_events) != result.get("lethal_intent_defend_override_count"):
        raise EvaluationIntegrityError("H16 trace and result override counts differ")
    if not candidate and override_events:
        raise EvaluationIntegrityError("unchanged G7 parent unexpectedly recorded an H16 override")
    for event in override_events:
        detail = event.get("lethal_intent_defend_detail")
        selected = event.get("selected_action")
        recommended = event.get("mcts_recommended_action")
        native_index = event.get("selected_native_action_index")
        native_actions = event.get("canonical_native_legal_actions")
        if (
            not isinstance(detail, dict)
            or not isinstance(selected, dict)
            or not isinstance(recommended, dict)
            or selected == recommended
            or selected.get("kind") != "play_card"
            or selected.get("hand_index") != detail.get("selected_defend_hand_index")
            or not isinstance(native_index, int)
            or not isinstance(native_actions, list)
            or not 0 <= native_index < len(native_actions)
            or native_actions[native_index] != selected
            or detail.get("incoming_damage", -1) < detail.get("player_hp", 0) + detail.get("player_block", 0)
            or detail.get("selected_defend_base_block", -1) < detail.get("projected_deficit", 0)
        ):
            raise EvaluationIntegrityError("H16 override did not select a legal registered Defend action")
    reasons = Counter(
        str(event.get("lethal_intent_defend_reason", "missing")) for event in combat_events
    )
    if candidate and sum(reasons.values()) != len(combat_events):
        raise EvaluationIntegrityError("H16 candidate trace lacks per-decision reasons")
    for artifact in (evidence_path, trace_path) if trace_mode == "on" else (evidence_path,):
        if artifact.stat().st_size > MAX_EPISODE_BYTES:
            raise EvaluationIntegrityError("H16 private episode artifact exceeded its bound")
    action_signature, action_count = _evidence_action_signature(evidence_path)
    return (
        result,
        evidence_path,
        trace_stats,
        len(override_events) if trace_mode == "on" else int(result["lethal_intent_defend_override_count"]),
        dict(reasons),
        trace_payload,
        action_signature,
        action_count,
    )


def _artifact_manifest(output_dir: Path) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    total = 0
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "private-summary.json":
            continue
        size = path.stat().st_size
        if size > MAX_EPISODE_BYTES:
            raise EvaluationIntegrityError("an H16 raw artifact exceeds the per-file limit")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise EvaluationIntegrityError("H16 raw artifacts exceed the total-size limit")
        entries.append({"name": path.relative_to(output_dir).as_posix(), "bytes": size, "sha256": _sha256(path)})
    return entries, total


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", choices=tuple(TRIAL_POOL_ROLE), default="h16")
    parser.add_argument("--stage", choices=tuple(STAGE_CONFIG), default="train")
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--pools-dir", type=Path, required=True)
    parser.add_argument("--exclusion-inventory", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--g7-checkpoint", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--private-usage-ledger", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    trial_id = args.trial_id
    stage = args.stage
    output_dir = args.private_output_dir.resolve()
    usage_path = args.private_usage_ledger.resolve()
    try:
        _validate_private_paths(args.pools_dir.resolve(), trial_id, stage, output_dir, usage_path)
        if len(args.candidate_commit) != 40 or args.candidate_commit != _git_head():
            raise EvaluationIntegrityError("candidate commit does not match the frozen local HEAD")
        identities = _validate_identity(
            args.module_dir.resolve(), args.armg_root.resolve(), args.g7_checkpoint.resolve()
        )
        events = _read_jsonl(usage_path)
        if trial_id == "h18":
            allocation_event = events[0] if events else None
            pool, seeds, pool_preflight, allocation_event = _round009_assets(
                trial_id=trial_id,
                stage=stage,
                pool_file=args.pool_file.resolve(),
                pools_dir=args.pools_dir.resolve(),
                inventory_path=args.exclusion_inventory.resolve(),
                allocation_event=allocation_event,
            )
            if not events:
                _append_jsonl(usage_path, allocation_event)
                events = [allocation_event]
        else:
            pool, seeds, pool_preflight = _round008_assets(
                trial_id=trial_id,
                stage=stage,
                pool_file=args.pool_file.resolve(),
                pools_dir=args.pools_dir.resolve(),
                inventory_path=args.exclusion_inventory.resolve(),
            )
        _validate_transition(
            trial_id=trial_id,
            stage=stage,
            events=events,
            candidate_commit=args.candidate_commit,
            identities=identities,
        )
        sts = _load_sts(args.module_dir.resolve())
        policy = ArmGNoncombatPolicy(
            root=args.armg_root.resolve(), weight_path=args.g7_checkpoint.resolve()
        )
    except Exception as exc:
        print(json.dumps({"status": "PREFLIGHT_FAIL", "error_class": type(exc).__name__}, sort_keys=True))
        return 2

    identity = {
        "round_id": TRIAL_ROUND_IDS[trial_id],
        "hypothesis": trial_id,
        "trial_id": trial_id,
        "stage": stage,
        "pool_id": pool["pool_id"],
        "pool_manifest_sha256": pool["manifest_sha256"],
        "paired_seed_count": STAGE_CONFIG[stage]["paired_seed_count"],
        "candidate_commit": args.candidate_commit,
        "mcts_sims": MCTS_SIMS,
        "g7_checkpoint_sha256": G7_SHA256,
        **identities,
        **pool_preflight,
    }
    if args.preflight_only:
        print(json.dumps({
            "status": "PREFLIGHT_PASS",
            "trial_id": trial_id,
            "stage": stage,
            "seed_count": len(seeds),
            "paired_episode_count": 2 * len(seeds),
            "trace_invariance_replay_count": int(trial_id == "h18" and stage == "train"),
            "candidate_commit": args.candidate_commit,
            "simulator_gameplay_commit": identities["simulator_gameplay_commit"],
            "simulator_policy_source_sha256": identities["simulator_policy_source_sha256"],
            "candidate_evaluator_sha256": identities["candidate_evaluator_sha256"],
            "simulator_binding_sha256": identities["simulator_binding_sha256"],
            "armg_source_sha256": identities["armg_source_sha256"],
            "armg_vocab_sha256": identities["armg_vocab_sha256"],
            "g7_checkpoint_sha256": G7_SHA256,
            "seed_disjointness_verified": pool_preflight.get("seed_disjointness_verified") is True,
        }, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    replay_count = int(trial_id == "h18" and stage == "train")
    paired_episode_count = 2 * len(seeds)
    expected_episodes = paired_episode_count + replay_count
    _append_jsonl(usage_path, {
        "record_type": f"{trial_id}_stage_start",
        **identity,
        "status": "RUNNING",
        "started_at_utc": started_at,
        "expected_episodes": expected_episodes,
        "paired_episode_count": paired_episode_count,
        "trace_invariance_replay_count": replay_count,
        "seed_ids": list(seeds),
        "private_output_dir": str(output_dir),
    })

    parent_outcomes: list[str] = []
    candidate_outcomes: list[str] = []
    parent_trace_totals = _empty_coverage()
    candidate_trace_totals = _empty_coverage()
    candidate_overrides = 0
    effective_override_seed_count = 0
    candidate_reasons: Counter[str] = Counter()
    terminal_pairs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    episodes_attempted = 0
    legal_trace_episodes = 0
    safety_totals = {key: 0 for key in SAFETY_COUNTERS}
    trace_replay_summary: dict[str, Any] | None = None
    try:
        for index, seed in enumerate(seeds):
            pair_index = index + 1
            print(f"{trial_id.upper()} {stage} pair {pair_index}/{len(seeds)}: parent")
            episodes_attempted += 1
            (
                parent, parent_path, parent_stats, parent_overrides, parent_reasons,
                parent_trace, parent_action_signature, parent_action_count,
            ) = _run_episode(
                sts=sts,
                policy=policy,
                seed=seed,
                stage_seeds=seeds,
                output_dir=output_dir,
                pair_index=pair_index,
                arm="parent",
                candidate=False,
                identity=identity,
                episode_index=episodes_attempted,
            )
            _accumulate_safety(safety_totals, parent)
            legal_trace_episodes += 1
            _accumulate_coverage(parent_trace_totals, parent_stats)
            if parent_overrides != 0 or parent.get("lethal_intent_defend_override_count") != 0:
                raise EvaluationIntegrityError("G7 parent arm had a registered H16/H17 action override")

            print(f"{trial_id.upper()} {stage} pair {pair_index}/{len(seeds)}: candidate")
            episodes_attempted += 1
            (
                candidate, candidate_path, candidate_stats, overrides, reasons,
                candidate_trace, candidate_action_signature, candidate_action_count,
            ) = _run_episode(
                sts=sts,
                policy=policy,
                seed=seed,
                stage_seeds=seeds,
                output_dir=output_dir,
                pair_index=pair_index,
                arm="candidate",
                candidate=True,
                identity=identity,
                episode_index=episodes_attempted,
            )
            _accumulate_safety(safety_totals, candidate)
            legal_trace_episodes += 1
            _accumulate_coverage(candidate_trace_totals, candidate_stats)
            candidate_overrides += overrides
            effective_override_for_seed = _pair_has_effective_override(parent_trace, candidate_trace)
            if effective_override_for_seed:
                effective_override_seed_count += 1
            candidate_reasons.update(reasons)

            if trial_id == "h18" and stage == "train" and pair_index == 1:
                print("H18 train trace-invariance replay (candidate, trace off)")
                episodes_attempted += 1
                (
                    replay, replay_path, _, replay_overrides, replay_reasons,
                    _, replay_action_signature, replay_action_count,
                ) = _run_episode(
                    sts=sts,
                    policy=policy,
                    seed=seed,
                    stage_seeds=seeds,
                    output_dir=output_dir,
                    pair_index=pair_index,
                    arm="candidate_trace_off",
                    candidate=True,
                    identity=identity,
                    trace_mode="off",
                    episode_index=episodes_attempted,
                )
                _accumulate_safety(safety_totals, replay)
                if (
                    not _trace_replay_matches(
                        candidate,
                        replay,
                        candidate_action_signature,
                        replay_action_signature,
                        candidate_action_count,
                        replay_action_count,
                    )
                    or replay_overrides != overrides
                ):
                    raise EvaluationIntegrityError("trace-off replay changed outcome or selected-action signature")
                trace_replay_summary = {
                    "status": "PASS",
                    "seed_id": seed,
                    "paired_denominator_included": False,
                    "trace_mode": "off",
                    "outcome_matches_trace_on": True,
                    "action_signature_matches_trace_on": True,
                    "action_signature_sha256": replay_action_signature,
                    "action_count": replay_action_count,
                    "override_count_matches_trace_on": True,
                    "evidence": str(replay_path),
                }

            parent_outcomes.append(str(parent["outcome"]))
            candidate_outcomes.append(str(candidate["outcome"]))

            parent_floor = parent.get("final_floor")
            candidate_floor = candidate.get("final_floor")
            parent_hp = parent.get("final_hp")
            candidate_hp = candidate.get("final_hp")
            values = (parent_floor, candidate_floor, parent_hp, candidate_hp)
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                raise EvaluationIntegrityError("paired final floor or HP is unavailable")
            terminal_pairs.append({
                "parent_outcome": parent["outcome"],
                "candidate_outcome": candidate["outcome"],
                "parent_floor": parent_floor,
                "candidate_floor": candidate_floor,
                "parent_hp": parent_hp,
                "candidate_hp": candidate_hp,
            })
            _append_jsonl(output_dir / "paired-results.private.jsonl", {
                "pair_index": pair_index,
                "seed_id": seed,
                "parent_outcome": parent["outcome"],
                "candidate_outcome": candidate["outcome"],
                "parent_summary": h3._summary_view(parent),
                "candidate_summary": h3._summary_view(candidate),
                "parent_action_signature_sha256": parent_action_signature,
                "candidate_action_signature_sha256": candidate_action_signature,
                "parent_action_count": parent_action_count,
                "candidate_action_count": candidate_action_count,
                "candidate_override_count": overrides,
                "effective_override_verified": effective_override_for_seed,
                "candidate_override_reasons": reasons,
                "parent_evidence": str(parent_path),
                "candidate_evidence": str(candidate_path),
            })
            _append_jsonl(usage_path, {
                "record_type": f"{trial_id}_stage_progress",
                "trial_id": trial_id,
                "stage": stage,
                "candidate_commit": args.candidate_commit,
                "completed_pairs": pair_index,
                "episodes_attempted": episodes_attempted,
                "candidate_overrides": candidate_overrides,
                "effective_override_seed_count": effective_override_seed_count,
            })
            rows.append({
                "pair_index": pair_index,
                "seed_id": seed,
                "parent": h3._summary_view(parent),
                "candidate": h3._summary_view(candidate),
                "candidate_override_count": overrides,
                "effective_override_verified": effective_override_for_seed,
            })

        paired = h3._paired_summary_for_stage(parent_outcomes, candidate_outcomes)
        terminal_diagnostic = _terminal_floor_hp_diagnostic(terminal_pairs)
        if replay_count and (trace_replay_summary is None or trace_replay_summary["status"] != "PASS"):
            raise EvaluationIntegrityError("required trace-invariance replay did not complete")
        hard_guards_passed = (
            episodes_attempted == expected_episodes
            and len(parent_outcomes) == len(seeds)
            and len(candidate_outcomes) == len(seeds)
            and paired_episode_count == 2 * len(seeds)
            and legal_trace_episodes == paired_episode_count
            and all(safety_totals.get(key) == 0 for key in SAFETY_COUNTERS)
        )
        gate = _stage_gate(
            stage,
            paired,
            effective_override_seed_count,
            hard_guards_passed,
        )
        coverage_required = gate["coverage_required"]
        coverage_passed = gate["coverage_passed"]
        stage_signal = gate["stage_signal_passed"]
        advance_eligible = gate["advance_eligible"]
        safety = {
            "illegal_action_count": safety_totals["illegal_action_count"],
            "crash_count": safety_totals["crash_count"],
            "timeout_count": safety_totals["timeout_count"],
            "communication_error_count": safety_totals["communication_error_count"],
            "complete_terminal_records": episodes_attempted,
            "paired_episode_count": paired_episode_count,
            "legal_trace_episode_count": legal_trace_episodes,
            "trace_invariance_replay_count": replay_count,
            "trace_mode": "paired_on_with_preregistered_trace_off_replay"
            if replay_count
            else "on_for_both_arms",
        }
        artifact_entries, artifact_bytes = _artifact_manifest(output_dir)
        summary = {
            "record_type": f"{trial_id}_stage_summary",
            **identity,
            "status": "COMPLETE",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed_count": len(seeds),
            "episodes": episodes_attempted,
            "paired_episodes": paired_episode_count,
            "trace_invariance_replay": trace_replay_summary,
            "parent_outcomes": {"victory": parent_outcomes.count("victory"), "defeat": parent_outcomes.count("defeat")},
            "candidate_outcomes": {"victory": candidate_outcomes.count("victory"), "defeat": candidate_outcomes.count("defeat")},
            "paired": paired,
            "candidate_override_count": candidate_overrides,
            "effective_override_seed_count": effective_override_seed_count,
            "override_seed_coverage_required": coverage_required,
            "override_seed_coverage_passed": coverage_passed,
            "stage_signal_passed": stage_signal,
            "candidate_override_reason_counts": dict(sorted(candidate_reasons.items())),
            "parent_trace_coverage": parent_trace_totals,
            "candidate_trace_coverage": candidate_trace_totals,
            "hard_guards_passed": hard_guards_passed,
            "terminal_floor_hp_diagnostic": terminal_diagnostic,
            "safety": safety,
            "advance_eligible": advance_eligible,
            "pairs": rows,
            "artifacts": artifact_entries,
            "artifact_bytes": artifact_bytes,
        }
        private_summary_path = output_dir / "private-summary.json"
        private_summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        summary["private_summary_sha256"] = _sha256(private_summary_path)
        _append_jsonl(usage_path, summary)
        print(json.dumps({
            "status": summary["status"],
            "stage": stage,
            "seed_count": len(seeds),
            "episodes": episodes_attempted,
            "parent_outcomes": summary["parent_outcomes"],
            "candidate_outcomes": summary["candidate_outcomes"],
            "paired": paired,
            "candidate_override_count": candidate_overrides,
            "effective_override_seed_count": effective_override_seed_count,
            "override_seed_coverage_required": coverage_required,
            "override_seed_coverage_passed": coverage_passed,
            "stage_signal_passed": stage_signal,
            "candidate_override_reason_counts": dict(sorted(candidate_reasons.items())),
            "trace_invariance_replay": (
                {
                    "status": trace_replay_summary["status"],
                    "paired_denominator_included": False,
                    "trace_mode": trace_replay_summary["trace_mode"],
                    "outcome_matches_trace_on": trace_replay_summary["outcome_matches_trace_on"],
                    "action_signature_matches_trace_on": trace_replay_summary[
                        "action_signature_matches_trace_on"
                    ],
                    "action_signature_sha256": trace_replay_summary[
                        "action_signature_sha256"
                    ],
                    "action_count": trace_replay_summary["action_count"],
                    "override_count_matches_trace_on": trace_replay_summary[
                        "override_count_matches_trace_on"
                    ],
                }
                if trace_replay_summary is not None
                else None
            ),
            "hard_guards_passed": hard_guards_passed,
            "terminal_floor_hp_diagnostic": terminal_diagnostic,
            "advance_eligible": advance_eligible,
            "safety": safety,
            "candidate_commit": args.candidate_commit,
            "simulator_policy_source_sha256": identities["simulator_policy_source_sha256"],
            "candidate_evaluator_sha256": identities["candidate_evaluator_sha256"],
            "artifact_bytes": artifact_bytes,
        }, sort_keys=True))
        return 0
    except Exception as exc:
        _append_jsonl(usage_path, {
            "record_type": f"{trial_id}_stage_failure",
            **identity,
            "status": "NOT_VERIFIED",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_attempted": episodes_attempted,
            "completed_pairs": len(parent_outcomes),
            "canonical_safety_counters_complete": False,
            "error_class": type(exc).__name__,
            "private_output_dir": str(output_dir),
        })
        print(json.dumps({
            "status": "NOT_VERIFIED",
            "stage": stage,
            "episodes_attempted": episodes_attempted,
            "completed_pairs": len(parent_outcomes),
            "error_class": type(exc).__name__,
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
