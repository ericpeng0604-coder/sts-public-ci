"""Run the preregistered Round008 G7-only train trace audit.

This diagnostic runs no candidate and performs no training or tuning. Raw seed
IDs, traces, and episode evidence must remain in a private output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sts1"))

from roguelike_ai.sts1_phase3.simulator import (  # noqa: E402
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)
from sts1_g7_h2_elite_route_eval import (  # noqa: E402
    EvaluationIntegrityError,
    _action_signature,
    _check_evidence,
    _summary_view,
)
from sts1_g7_h7_potion_trace_audit import MCTS_SIMS, _validate_trace  # noqa: E402
from sts1_g7_seed_ledger import (  # noqa: E402
    MAX_SEED,
    SeedLedgerError,
    sha256_json,
    validate_inventory,
)

ROUND_ID = "round-008-20261009"
GENERATION_KEY = "sts1-g7-round-008-direction-review-20261009-v1"
INVENTORY_ID = "g7-exclusions-plus-round-007-20261009"
EXPECTED_INVENTORY_SHA256 = "0d47f8956ad9f962ede5bac7bbd99931bbcfb1f7e9892dfbfb33259f921749be"
EXPECTED_G7_SHA256 = "8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0"
EXPECTED_SIMULATOR_SOURCE_SHA256 = "981b33b0f39a0f139457185caa48dc84e93d179e4e0b67940cb2392890f1f72c"
EXPECTED_SIMULATOR_BINDING_SHA256 = "bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e"
EXPECTED_ARMG_SOURCE_SHA256 = "7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b"
EXPECTED_ARMG_VOCAB_SHA256 = "832e199c359af8408ea430ffa3f9fcdc68f32533f7292bb102848d3fb558eb6a"
POOL_SIZES = {
    "train_hypothesis_1": 10,
    "train_hypothesis_2": 10,
    "train_hypothesis_3": 10,
    "probe": 10,
    "dev": 30,
}
EXPECTED_POOL_SHA256 = {
    "train_hypothesis_1": "b24f900ee92864b32dfc090997c20e252f9fda4a952ea1f64ce714da313831ae",
    "train_hypothesis_2": "a71b491d582dfd358c9fc48c0a3f3e9f2a907e3623a447aff598f19417eb2190",
    "train_hypothesis_3": "fe8f0bc33ba50bac941ddb40278775cb0f36934df027b9e2c1df595b7b3d4b7d",
    "probe": "56de6fce082c49227c24652ab152ba331aa0730fc087366d068ba5c33902e5d0",
    "dev": "8636a6121e8123e2f8746dfeb57c6422ee15cf425c403ab4f276ca10aaf29d14",
}
MAX_EPISODE_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_ARTIFACT_BYTES = 512 * 1024 * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationIntegrityError(f"could not read {path.name}") from exc
    if not isinstance(value, dict):
        raise EvaluationIntegrityError(f"{path.name} must contain an object")
    return value


def _validate_pool_seed_sets(
    pool_seeds: Mapping[str, Sequence[Any]], excluded: set[int]
) -> dict[str, tuple[int, ...]]:
    if set(pool_seeds) != set(POOL_SIZES):
        raise EvaluationIntegrityError("Round008 pool roles are incomplete or unexpected")
    all_seen = set(excluded)
    validated: dict[str, tuple[int, ...]] = {}
    for role, expected_count in POOL_SIZES.items():
        raw = pool_seeds[role]
        if not isinstance(raw, (list, tuple)) or len(raw) != expected_count:
            raise EvaluationIntegrityError(f"{role} has an invalid seed count")
        if any(
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 1 <= seed <= MAX_SEED
            for seed in raw
        ):
            raise EvaluationIntegrityError(f"{role} contains an invalid simulator seed")
        seeds = tuple(raw)
        if len(set(seeds)) != len(seeds):
            raise EvaluationIntegrityError(f"{role} contains duplicate seeds")
        if all_seen.intersection(seeds):
            raise EvaluationIntegrityError(f"{role} overlaps an excluded or Round008 seed")
        all_seen.update(seeds)
        validated[role] = seeds
    return validated


def _validate_round008_assets(
    pool_file: Path,
    pools_dir: Path,
    inventory_file: Path,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]], dict[str, Any]]:
    inventory = _read_json(inventory_file)
    if inventory.get("inventory_id") != INVENTORY_ID:
        raise EvaluationIntegrityError("Round008 exclusion inventory identity mismatch")
    inventory_sha = _file_sha256(inventory_file)
    if inventory_sha != EXPECTED_INVENTORY_SHA256:
        raise EvaluationIntegrityError("Round008 exclusion inventory hash mismatch")
    try:
        excluded, source_audit = validate_inventory(inventory)
    except SeedLedgerError as exc:
        raise EvaluationIntegrityError("Round008 exclusion inventory validation failed") from exc
    source_audit_sha = sha256_json(source_audit)

    ledger_path = pools_dir / "ledger.json"
    ledger = _read_json(ledger_path)
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if (
        ledger.get("status") != "GENERATED_NOT_RUN"
        or ledger.get("round_id") != ROUND_ID
        or ledger.get("generation_key") != GENERATION_KEY
        or ledger.get("inventory_id") != INVENTORY_ID
        or ledger.get("inventory_sha256") != inventory_sha
        or ledger.get("source_audit_sha256") != source_audit_sha
        or sha256_json(ledger_payload) != ledger.get("ledger_sha256")
    ):
        raise EvaluationIntegrityError("Round008 seed ledger provenance or hash mismatch")

    ledger_pools = ledger.get("pools")
    if not isinstance(ledger_pools, dict) or set(ledger_pools) != set(POOL_SIZES):
        raise EvaluationIntegrityError("Round008 ledger pool map is incomplete")
    pool_docs: dict[str, dict[str, Any]] = {}
    for role, count in POOL_SIZES.items():
        path = pools_dir / f"{role}.json"
        pool = _read_json(path)
        pool_payload = {key: value for key, value in pool.items() if key != "manifest_sha256"}
        expected_purpose = "train" if role.startswith("train_") else role
        ledger_entry = ledger_pools[role]
        if (
            pool.get("pool_id") != f"{ROUND_ID}-{role}"
            or pool.get("round_id") != ROUND_ID
            or pool.get("generation_key") != GENERATION_KEY
            or pool.get("purpose") != expected_purpose
            or pool.get("status") != "GENERATED_NOT_RUN"
            or pool.get("inventory_id") != INVENTORY_ID
            or pool.get("inventory_sha256") != inventory_sha
            or pool.get("source_audit_sha256") != source_audit_sha
            or pool.get("source_audit") != source_audit
            or pool.get("manifest_sha256") != EXPECTED_POOL_SHA256[role]
            or sha256_json(pool_payload) != pool.get("manifest_sha256")
            or not isinstance(ledger_entry, dict)
            or ledger_entry.get("manifest_sha256") != pool.get("manifest_sha256")
            or ledger_entry.get("seed_ids") != pool.get("seed_ids")
            or len(pool.get("seed_ids", [])) != count
        ):
            raise EvaluationIntegrityError(f"Round008 {role} manifest or ledger entry mismatch")
        pool_docs[role] = pool

    expected_files = {"ledger.json", *(f"{role}.json" for role in POOL_SIZES)}
    actual_files = {path.name for path in pools_dir.glob("*.json")}
    if actual_files != expected_files:
        raise EvaluationIntegrityError("Round008 generated-pool directory has unexpected JSON files")

    validated = _validate_pool_seed_sets(
        {role: pool["seed_ids"] for role, pool in pool_docs.items()}, excluded
    )
    selected = pool_file.resolve()
    if selected != (pools_dir / "train_hypothesis_1.json").resolve():
        raise EvaluationIntegrityError("H15 is restricted to train_hypothesis_1")
    preflight = {
        "round_id": ROUND_ID,
        "pool_id": pool_docs["train_hypothesis_1"]["pool_id"],
        "pool_manifest_sha256": EXPECTED_POOL_SHA256["train_hypothesis_1"],
        "inventory_id": INVENTORY_ID,
        "inventory_sha256": inventory_sha,
        "exclusion_source_count": len(source_audit),
        "excluded_unique_seed_count": len(excluded),
        "pool_sizes": dict(POOL_SIZES),
        "generation_key": GENERATION_KEY,
    }
    return pool_docs["train_hypothesis_1"], validated, preflight


def _validate_private_paths(pools_dir: Path, output_dir: Path, usage_ledger: Path) -> None:
    repo = REPO_ROOT.resolve()
    pools_dir = pools_dir.resolve()
    expected_output_dir = pools_dir.parent.parent / "round-008-h15-train-trace-20261009"
    expected_usage_ledger = pools_dir.parent / "h15-usage-private.jsonl"
    if output_dir.resolve() != expected_output_dir.resolve() or usage_ledger.resolve() != expected_usage_ledger.resolve():
        raise EvaluationIntegrityError("H15 requires its single canonical private output and usage-ledger paths")
    for path in (output_dir.resolve(), usage_ledger.resolve()):
        if path == repo or repo in path.parents:
            raise EvaluationIntegrityError("raw H15 artifacts must stay outside the repository")
    if output_dir.exists() or usage_ledger.exists():
        raise EvaluationIntegrityError("H15 private output or usage-ledger path already exists")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_manifest(output_dir: Path) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    total = 0
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_EPISODE_ARTIFACT_BYTES:
            raise EvaluationIntegrityError("an H15 private artifact exceeds the per-file limit")
        total += size
        if total > MAX_TOTAL_ARTIFACT_BYTES:
            raise EvaluationIntegrityError("H15 private artifacts exceed the total-size limit")
        entries.append({"name": path.relative_to(output_dir).as_posix(), "bytes": size, "sha256": _file_sha256(path)})
    return entries, total


def _validate_identity(module_dir: Path, armg_root: Path, checkpoint: Path) -> dict[str, str]:
    simulator_source = REPO_ROOT / "src" / "roguelike_ai" / "sts1_phase3" / "simulator.py"
    binding = module_dir / "slaythespire.cp312-win_amd64.pyd"
    armg_source = armg_root / "armG_train.py"
    armg_vocab = armg_root / "armS_card_vocab.json"
    expected = {
        "simulator_source_sha256": (simulator_source, EXPECTED_SIMULATOR_SOURCE_SHA256),
        "simulator_binding_sha256": (binding, EXPECTED_SIMULATOR_BINDING_SHA256),
        "armg_source_sha256": (armg_source, EXPECTED_ARMG_SOURCE_SHA256),
        "armg_vocab_sha256": (armg_vocab, EXPECTED_ARMG_VOCAB_SHA256),
        "g7_checkpoint_sha256": (checkpoint, EXPECTED_G7_SHA256),
    }
    actual: dict[str, str] = {}
    for name, (path, digest) in expected.items():
        if not path.is_file() or _file_sha256(path) != digest:
            raise EvaluationIntegrityError(f"pinned {name} identity mismatch")
        actual[name] = digest
    if Path(str(checkpoint) + ".adapter.pt").exists():
        raise EvaluationIntegrityError("pinned G7 checkpoint has an adapter sidecar")
    if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
        raise EvaluationIntegrityError("contextual Teacher reranking must be disabled")
    return actual


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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.private_output_dir.resolve()
    usage_ledger = args.private_usage_ledger.resolve()
    try:
        pool, pool_seeds, pool_preflight = _validate_round008_assets(
            args.pool_file.resolve(), args.pools_dir.resolve(), args.exclusion_inventory.resolve()
        )
        _validate_private_paths(args.pools_dir.resolve(), output_dir, usage_ledger)
        identity = _validate_identity(args.module_dir.resolve(), args.armg_root.resolve(), args.g7_checkpoint.resolve())
        identities = {
            "stage": "h15_g7_train_trace_audit",
            "policy_change": "none; unchanged pinned G7 diagnostic only",
            "pool_manifest_sha256": pool["manifest_sha256"],
            "simulator_source_sha256": EXPECTED_SIMULATOR_SOURCE_SHA256,
            "mcts_sims": MCTS_SIMS,
            **identity,
        }
        sts = _load_sts(args.module_dir.resolve())
        policy = ArmGNoncombatPolicy(root=args.armg_root.resolve(), weight_path=args.g7_checkpoint.resolve())
    except Exception as exc:
        print(json.dumps({"preflight": "FAIL", "error_class": type(exc).__name__}, sort_keys=True))
        return 2

    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", **pool_preflight, **identities}, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    completed_episodes = 0
    _append_jsonl(usage_ledger, {
        "pool_id": pool_preflight["pool_id"],
        "manifest_sha256": pool["manifest_sha256"],
        "stage": identities["stage"],
        "status": "RUNNING",
        "started_at_utc": started_at,
        "expected_episodes": 11,
        "trace_on_episodes": 10,
        "trace_off_invariance_replays": 1,
    })
    rows: list[dict[str, Any]] = []
    outcomes: list[str] = []
    trace_totals: Counter[str] = Counter()
    trace_invariance = "NOT_VERIFIED"
    try:
        seeds = pool_seeds["train_hypothesis_1"]
        for index, seed in enumerate(seeds):
            episode = index + 1
            evidence_path = output_dir / f"episode-{episode:02d}.evidence.ndjson"
            trace_path = output_dir / f"episode-{episode:02d}.trace.ndjson"
            if evidence_path.exists() or trace_path.exists():
                raise EvaluationIntegrityError("H15 episode output already exists")
            metadata = {**identities, "episode_index": episode, "seed_id": seed}
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
                diagnostic_metadata=metadata,
            )
            completed_episodes += 1
            _check_evidence(evidence_path, result, candidate=False)
            trace_stats = _validate_trace(trace_path, result, metadata)
            outcome = result.get("outcome")
            if outcome not in {"victory", "defeat"}:
                raise EvaluationIntegrityError("H15 trace episode has no complete terminal outcome")
            trace_totals.update(trace_stats)
            outcomes.append(outcome)
            rows.append({
                "episode_index": episode,
                "seed_id": seed,
                "summary": _summary_view(result),
                "trace_stats": trace_stats,
                "evidence_sha256": _file_sha256(evidence_path),
                "trace_sha256": _file_sha256(trace_path),
            })
            _append_jsonl(output_dir / "episode-results-private.jsonl", {
                "episode_index": episode,
                "seed_id": seed,
                "outcome": outcome,
                "final_floor": result.get("final_floor"),
                "final_hp": result.get("final_hp"),
                "trace_stats": trace_stats,
            })
            _append_jsonl(usage_ledger, {
                "pool_id": pool_preflight["pool_id"],
                "manifest_sha256": pool["manifest_sha256"],
                "stage": identities["stage"],
                "status": "PROGRESS",
                "completed_episodes": completed_episodes,
                "trace_on_episodes": len(outcomes),
            })

            if index == 0:
                trace_off_evidence = output_dir / "episode-01-trace-off.evidence.ndjson"
                if trace_off_evidence.exists():
                    raise EvaluationIntegrityError("H15 trace-off replay already exists")
                trace_off_metadata = {**identities, "episode_index": episode, "seed_id": seed}
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
                    _summary_view(trace_off_result) != _summary_view(result)
                    or _action_signature(trace_off_evidence) != _action_signature(evidence_path)
                ):
                    raise EvaluationIntegrityError("trace-on/off changed the fixed-seed G7 trajectory")
                trace_invariance = "PASS"
                _append_jsonl(output_dir / "episode-results-private.jsonl", {
                    "episode_index": episode,
                    "seed_id": seed,
                    "trace_off_replay": True,
                    "outcome": trace_off_result.get("outcome"),
                    "evidence_sha256": _file_sha256(trace_off_evidence),
                    "trace_invariance": trace_invariance,
                })
                _append_jsonl(usage_ledger, {
                    "pool_id": pool_preflight["pool_id"],
                    "manifest_sha256": pool["manifest_sha256"],
                    "stage": identities["stage"],
                    "status": "PROGRESS",
                    "completed_episodes": completed_episodes,
                    "trace_invariance": trace_invariance,
                })

        wins = sum(outcome == "victory" for outcome in outcomes)
        losses = len(outcomes) - wins
        artifact_entries, artifact_bytes = _artifact_manifest(output_dir)
        private_summary = {
            "status": "PASS_DIAGNOSTIC_ONLY",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "preflight": pool_preflight,
            "identities": identities,
            "episodes_completed": completed_episodes,
            "trace_on_episode_count": len(rows),
            "trace_off_invariance_replays": 1,
            "trace_invariance": trace_invariance,
            "outcomes": {"victory": wins, "defeat": losses},
            "trace_totals": dict(sorted(trace_totals.items())),
            "episodes": rows,
            "artifacts": artifact_entries,
            "artifact_bytes": artifact_bytes,
        }
        summary_path = output_dir / "audit-summary-private.json"
        summary_path.write_text(json.dumps(private_summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        summary_sha = _file_sha256(summary_path)
        _append_jsonl(usage_ledger, {
            "pool_id": pool_preflight["pool_id"],
            "manifest_sha256": pool["manifest_sha256"],
            "stage": identities["stage"],
            "status": "PASS_DIAGNOSTIC_ONLY",
            "finished_at_utc": private_summary["finished_at_utc"],
            "completed_episodes": completed_episodes,
            "trace_invariance": trace_invariance,
            "private_summary_sha256": summary_sha,
            "artifact_bytes": artifact_bytes,
        })
    except Exception as exc:
        _append_jsonl(usage_ledger, {
            "pool_id": pool_preflight["pool_id"],
            "manifest_sha256": pool["manifest_sha256"],
            "stage": identities["stage"],
            "status": "NOT_VERIFIED",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_episodes": completed_episodes,
            "error_class": type(exc).__name__,
        })
        print(json.dumps({
            "status": "NOT_VERIFIED",
            "error_class": type(exc).__name__,
            "completed_episodes": completed_episodes,
            "trace_invariance": trace_invariance,
        }, sort_keys=True))
        return 2

    print(json.dumps({
        "status": "PASS_DIAGNOSTIC_ONLY",
        **pool_preflight,
        **identities,
        "episodes_completed": completed_episodes,
        "trace_on_episode_count": len(outcomes),
        "trace_off_invariance_replays": 1,
        "trace_invariance": trace_invariance,
        "outcomes": {"victory": wins, "defeat": losses},
        "trace_totals": dict(sorted(trace_totals.items())),
        "artifact_bytes": artifact_bytes,
        "private_summary_sha256": summary_sha,
        "communication_errors": "N/A_LOCAL_SIMULATOR",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
