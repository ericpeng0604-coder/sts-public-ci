"""Run the preregistered H8/H9 paired train / probe / dev evaluations.

Raw seeds, traces, and episode records are written only to the caller's private
output directory. Console and repository summaries contain aggregates only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sts1"))

import sts1_g7_h2_elite_route_eval as h2  # noqa: E402
import sts1_g7_h3_emergency_potion_eval as h3  # noqa: E402
import sts1_g7_h7_potion_trace_audit as h7  # noqa: E402
from roguelike_ai.sts1_phase3.simulator import (  # noqa: E402
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)
from sts1_g7_seed_ledger import MAX_SEED, sha256_json, validate_inventory  # noqa: E402


ROUND_ID = "round-006-20261009"
POOL_FILES = {
    "train_hypothesis_1": "train_hypothesis_1.json",
    "train_hypothesis_2": "train_hypothesis_2.json",
    "train_hypothesis_3": "train_hypothesis_3.json",
    "probe": "probe.json",
    "dev": "dev.json",
}
POOL_COUNTS = {
    "train_hypothesis_1": 10,
    "train_hypothesis_2": 10,
    "train_hypothesis_3": 10,
    "probe": 10,
    "dev": 30,
}
STAGE_ORDER = tuple(POOL_FILES)
TRAIN_STAGE_BY_HYPOTHESIS = {
    "h8": "train_hypothesis_1",
    "h9": "train_hypothesis_2",
}
TRACE_COVERAGE_FIELDS = (
    "combat_decision_count",
    "encounter_count",
    "noncombat_decision_count",
    "route_decision_count",
    "potion_snapshot_count",
)
MCTS_SIMS = 2000
G7_SHA256 = "8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0"
ACTIONS_ZIP_SHA256 = "3f4fc50b9452138e50ed5274cc7ac71a9ad011077785279de7834a478c934e66"
PINNED_BINDING_SHA256 = "bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e"
NATIVE_SOURCE_COMMIT_LABEL = "7476a81954020087da31d41d16fddf475746ec2d"
MAX_EPISODE_BYTES = h7.MAX_EPISODE_ARTIFACT_BYTES
MAX_TOTAL_BYTES = h7.MAX_TOTAL_ARTIFACT_BYTES


class EvaluationIntegrityError(RuntimeError):
    """H8 evaluation preflight, safety, or completeness validation failed."""


def _sha256(path: Path) -> str:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise EvaluationIntegrityError("usage ledger contains a non-object record")
                records.append(record)
    return records


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _check_actions_archive(path: Path) -> dict[str, str]:
    if not path.is_file() or _sha256(path) != ACTIONS_ZIP_SHA256:
        raise EvaluationIntegrityError("Actions archive hash does not match the verified artifact")
    try:
        with zipfile.ZipFile(path) as archive:
            checkpoint = archive.read("offline-champion.pt")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise EvaluationIntegrityError("verified G7 checkpoint member is unavailable") from exc
    checkpoint_sha = hashlib.sha256(checkpoint).hexdigest()
    if checkpoint_sha != G7_SHA256:
        raise EvaluationIntegrityError("offline champion member is not the pinned G7 checkpoint")
    return {"actions_zip_sha256": ACTIONS_ZIP_SHA256, "g7_checkpoint_sha256": checkpoint_sha}


def _materialize_g7_checkpoint(archive_path: Path, destination: Path) -> Path:
    destination = destination.resolve()
    repository = REPO_ROOT.resolve()
    if destination == repository or repository in destination.parents:
        raise EvaluationIntegrityError("private G7 checkpoint path cannot be inside the repository")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            checkpoint = archive.read("offline-champion.pt")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise EvaluationIntegrityError("verified G7 checkpoint member is unavailable") from exc
    if hashlib.sha256(checkpoint).hexdigest() != G7_SHA256:
        raise EvaluationIntegrityError("offline champion member is not the pinned G7 checkpoint")

    adapter_path = Path(str(destination) + ".adapter.pt")
    if adapter_path.exists():
        raise EvaluationIntegrityError("pinned G7 checkpoint cannot have an adapter sidecar")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != G7_SHA256:
            raise EvaluationIntegrityError("existing private G7 checkpoint has an unexpected hash")
        return destination

    try:
        with destination.open("xb") as handle:
            handle.write(checkpoint)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if not destination.is_file() or _sha256(destination) != G7_SHA256:
            raise EvaluationIntegrityError("concurrent private G7 checkpoint has an unexpected hash")
    if _sha256(destination) != G7_SHA256:
        raise EvaluationIntegrityError("materialized private G7 checkpoint hash mismatch")
    return destination


def _validate_pool(
    *, stage: str, public_summary_path: Path, pools_dir: Path, inventory_path: Path
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    if stage not in POOL_FILES:
        raise EvaluationIntegrityError("unsupported paired-evaluation stage")
    summary = _read_json(public_summary_path)
    if summary.get("round_id") != ROUND_ID or summary.get("record_type") != "exploration_round":
        raise EvaluationIntegrityError("public Round006 summary identity mismatch")

    ledger = _read_json(pools_dir / "ledger.json")
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    ledger_sha = ledger.get("ledger_sha256")
    if not isinstance(ledger_sha, str) or sha256_json(ledger_payload) != ledger_sha:
        raise EvaluationIntegrityError("private Round006 ledger hash mismatch")
    if summary.get("manifest_sha256") != ledger_sha:
        raise EvaluationIntegrityError("public summary does not match the private Round006 ledger")
    if ledger.get("round_id") != ROUND_ID:
        raise EvaluationIntegrityError("private ledger round identity mismatch")

    inventory = _read_json(inventory_path)
    excluded, source_audit = validate_inventory(inventory)
    inventory_sha = _sha256(inventory_path)
    if (
        summary.get("inventory_id") != inventory.get("inventory_id")
        or summary.get("inventory_sha256") != inventory_sha
        or summary.get("source_audit_sha256") != sha256_json(source_audit)
    ):
        raise EvaluationIntegrityError("public summary and private exclusion inventory differ")

    public_pools = summary.get("pools")
    private_pools = ledger.get("pools")
    if not isinstance(public_pools, dict) or not isinstance(private_pools, dict):
        raise EvaluationIntegrityError("Round006 pool registries are incomplete")
    seen: set[int] = set()
    selected_pool: dict[str, Any] | None = None
    selected_seeds: tuple[int, ...] | None = None
    for pool_key, filename in POOL_FILES.items():
        path = pools_dir / filename
        pool = _read_json(path)
        pool_payload = {key: value for key, value in pool.items() if key != "manifest_sha256"}
        pool_hash = pool.get("manifest_sha256")
        pool_ids = pool.get("seed_ids")
        expected_purpose = "dev" if pool_key == "dev" else ("probe" if pool_key == "probe" else "train")
        count = POOL_COUNTS[pool_key]
        if (
            not isinstance(pool_hash, str)
            or sha256_json(pool_payload) != pool_hash
            or pool.get("pool_id") != f"{ROUND_ID}-{pool_key}"
            or pool.get("purpose") != expected_purpose
            or pool.get("status") != "GENERATED_NOT_RUN"
            or not isinstance(pool_ids, list)
            or len(pool_ids) != count
            or any(not isinstance(seed, int) or isinstance(seed, bool) or not 1 <= seed <= MAX_SEED for seed in pool_ids)
            or len(set(pool_ids)) != count
        ):
            raise EvaluationIntegrityError("private pool manifest is invalid or already consumed")
        ledger_entry = private_pools.get(pool_key)
        public_entry = public_pools.get(pool_key)
        if (
            not isinstance(ledger_entry, dict)
            or ledger_entry.get("seed_ids") != pool_ids
            or ledger_entry.get("manifest_sha256") != pool_hash
            or not isinstance(public_entry, dict)
            or public_entry.get("pool_id") != pool.get("pool_id")
            or public_entry.get("purpose") != expected_purpose
            or public_entry.get("count") != count
            or public_entry.get("manifest_sha256") != pool_hash
        ):
            raise EvaluationIntegrityError("public, private, and pool ledger records differ")
        if (
            pool.get("inventory_id") != inventory.get("inventory_id")
            or pool.get("inventory_sha256") != inventory_sha
            or pool.get("source_audit_sha256") != sha256_json(source_audit)
            or pool.get("source_audit") != source_audit
        ):
            raise EvaluationIntegrityError("pool is not bound to the complete exclusion inventory")
        current = set(pool_ids)
        if current & excluded or current & seen:
            raise EvaluationIntegrityError("Round006 pools overlap protected or sibling seeds")
        seen.update(current)
        if pool_key == stage:
            if public_entry.get("status") != "GENERATED_NOT_RUN":
                raise EvaluationIntegrityError("selected pool was already used")
            selected_pool = pool
            selected_seeds = tuple(pool_ids)

    if selected_pool is None or selected_seeds is None:
        raise EvaluationIntegrityError("selected pool is missing")
    return selected_pool, selected_seeds, {
        "train_seed_count": len(selected_seeds),
        "exclusion_unique_seed_count": len(excluded),
        "pool_count": len(POOL_FILES),
        "inventory_sha256": inventory_sha,
        "private_ledger_sha256": ledger_sha,
        "disjoint_from_exclusion_inventory": True,
        "round_pools_pairwise_disjoint": True,
    }


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["rtk", "git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationIntegrityError("local candidate HEAD could not be verified") from exc


def _check_pair_integrity(path: Path, result: dict[str, Any]) -> None:
    h2._check_evidence(path, result, candidate=False)
    if result.get("outcome") not in {"victory", "defeat"} or result.get("error") is not None:
        raise EvaluationIntegrityError("episode did not produce a complete terminal outcome")
    for key in ("illegal_action_count", "timeout_count", "crash_count"):
        value = result.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise EvaluationIntegrityError("episode safety counter was nonzero or unavailable")
    communication_errors = result.get("communication_error_count", 0)
    if communication_errors not in (0, None):
        raise EvaluationIntegrityError("local simulator communication error counter was nonzero")
    if result.get("potion_inventory_snapshot_complete") is not True:
        raise EvaluationIntegrityError("complete potion inventory was not recorded")


def _run_episode(
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
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path | None, dict[str, int] | None]:
    stem = f"pair-{pair_index:02d}-{arm}{'-trace-off' if not trace_enabled else ''}"
    evidence_path = output_dir / f"{stem}.evidence.ndjson"
    trace_path = output_dir / f"{stem}.trace.ndjson" if trace_enabled else None
    if evidence_path.exists() or (trace_path is not None and trace_path.exists()):
        raise EvaluationIntegrityError("episode output already exists; refusing to rerun a seed")
    result = run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=evidence_path,
        armg_policy=policy,
        combat_mcts_sims=MCTS_SIMS,
        reserve_last_potion_until_floor=None,
        use_potion_below_hp_fraction=None,
        lethal_potion_rescue=candidate,
        avoid_low_hp_elite_routes=False,
        prefer_smith_when_rest_overheals=False,
        prefer_smith_when_overheal_exceeds_effective_rest_heal=False,
        training_seeds=all_training_seeds,
        collect_ppo=False,
        collect_teacher=False,
        diagnostic_trace_path=trace_path,
        diagnostic_metadata=metadata if trace_enabled else None,
    )
    _check_pair_integrity(evidence_path, result)
    trace_stats = None
    if trace_path is not None:
        trace_stats = h7._validate_trace(trace_path, result, metadata)
        if result.get("lethal_potion_rescue_enabled") is not candidate:
            raise EvaluationIntegrityError("H8 policy activation flag differs from the requested arm")
        override_events = sum(
            event.get("lethal_potion_rescue_override") is True
            for event in h7._read_jsonl(trace_path)
            if event.get("type") == "combat_decision_trace_v1"
        )
        if override_events != result.get("lethal_potion_rescue_override_count"):
            raise EvaluationIntegrityError("H8 trace and result override counts differ")
    for artifact in (evidence_path, trace_path):
        if artifact is not None and artifact.stat().st_size > MAX_EPISODE_BYTES:
            raise EvaluationIntegrityError("per-episode raw evidence exceeded the registered bound")
    return result, evidence_path, trace_path, trace_stats


def _outcome_counts(values: list[str]) -> dict[str, int]:
    return {"victory": values.count("victory"), "defeat": values.count("defeat")}


def _empty_trace_coverage() -> dict[str, int]:
    return {key: 0 for key in TRACE_COVERAGE_FIELDS}


def _accumulate_trace_coverage(
    totals: dict[str, int], trace_stats: dict[str, int] | None
) -> None:
    if not isinstance(trace_stats, dict) or set(totals) != set(TRACE_COVERAGE_FIELDS):
        raise EvaluationIntegrityError("validated trace coverage is unavailable")
    for key in TRACE_COVERAGE_FIELDS:
        value = trace_stats.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvaluationIntegrityError("validated trace coverage counter is invalid")
        totals[key] += value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGE_ORDER, required=True)
    parser.add_argument("--hypothesis", choices=tuple(TRAIN_STAGE_BY_HYPOTHESIS), default="h8")
    parser.add_argument("--public-summary", type=Path, default=REPO_ROOT / "evidence/sts1/g7-improvement/seeds/round-006-20261009/summary.json")
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--pools-dir", type=Path, required=True)
    parser.add_argument("--exclusion-inventory", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--g7-actions-zip", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--private-usage-ledger", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        train_stage = TRAIN_STAGE_BY_HYPOTHESIS[args.hypothesis]
        if args.stage not in {train_stage, "probe", "dev"}:
            raise EvaluationIntegrityError("stage is not authorized for the selected hypothesis")
        expected_pool_file = POOL_FILES[args.stage]
        expected_count = POOL_COUNTS[args.stage]
        if args.pool_file.name != expected_pool_file:
            raise EvaluationIntegrityError("selected stage pool filename mismatch")
        pool, seeds, inventory_summary = _validate_pool(
            stage=args.stage,
            public_summary_path=args.public_summary.resolve(),
            pools_dir=args.pools_dir.resolve(),
            inventory_path=args.exclusion_inventory.resolve(),
        )
        if len(seeds) != expected_count:
            raise EvaluationIntegrityError("registered stage pool count mismatch")
        if args.candidate_commit != _git_head() or len(args.candidate_commit) != 40:
            raise EvaluationIntegrityError("candidate commit does not match the current local HEAD")
        if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
            raise EvaluationIntegrityError("contextual reranking must be disabled for paired evaluation")

        binding = args.module_dir / "slaythespire.cp312-win_amd64.pyd"
        if not binding.is_file() or _sha256(binding) != PINNED_BINDING_SHA256:
            raise EvaluationIntegrityError("pinned H7 native simulator binding hash mismatch")
        simulator_source = REPO_ROOT / "src/roguelike_ai/sts1_phase3/simulator.py"
        simulator_source_sha = _sha256(simulator_source)
        evaluator_sha = _sha256(Path(__file__).resolve())
        armg_source = args.armg_root / "armG_train.py"
        if not armg_source.is_file():
            raise EvaluationIntegrityError("pinned ArmG source file is missing")
        armg_sha = _sha256(armg_source)
        archive_identity = _check_actions_archive(args.g7_actions_zip.resolve())

        usage_path = args.private_usage_ledger.resolve()
        prior_events = _read_jsonl(usage_path)
        completed = {
            event.get("stage"): event
            for event in prior_events
            if event.get("record_type") in {"h8_stage_summary", "h9_stage_summary"}
            and event.get("hypothesis", "h8") == args.hypothesis
        }
        for event in prior_events:
            if (
                event.get("record_type") == f"{args.hypothesis}_stage_start"
                and event.get("stage") == args.stage
            ):
                raise EvaluationIntegrityError("stage has a prior attempt; seeds cannot be reused")
        if args.stage == train_stage:
            if any(
                event.get("stage") == args.stage
                and event.get("hypothesis", "h8") == args.hypothesis
                for event in prior_events
            ):
                raise EvaluationIntegrityError("train pool already has a private use record")
        else:
            predecessor = train_stage if args.stage == "probe" else "probe"
            previous = completed.get(predecessor)
            if not isinstance(previous, dict) or previous.get("status") != "COMPLETE":
                raise EvaluationIntegrityError("required prior hypothesis stage is not complete")
            if previous.get("candidate_commit") != args.candidate_commit:
                raise EvaluationIntegrityError("candidate changed after the preceding hypothesis stage")
            if previous.get("simulator_source_sha256") != simulator_source_sha:
                raise EvaluationIntegrityError("simulator source changed after the preceding hypothesis stage")
            if previous.get("binding_sha256") != PINNED_BINDING_SHA256:
                raise EvaluationIntegrityError("native simulator binding changed after the preceding hypothesis stage")
            if previous.get("g7_checkpoint_sha256") != G7_SHA256:
                raise EvaluationIntegrityError("G7 parent identity changed after the preceding hypothesis stage")
            if previous.get(
                f"{args.hypothesis}_override_count",
                previous.get("h8_override_count", 0),
            ) <= 0:
                raise EvaluationIntegrityError("preceding hypothesis stage had no valid action change")
            if args.stage == "dev" and previous.get("paired", {}).get("net_wins", -1) < 0:
                raise EvaluationIntegrityError("Dev30 is gated on a nonnegative Probe10 net")

        output_dir = args.private_output_dir.resolve()
        h3._check_private_paths(output_dir, usage_path, pool_id=pool["pool_id"])
        if args.preflight_only:
            print(json.dumps({
                "status": "PREFLIGHT_PASS",
                "stage": args.stage,
                "seed_count": len(seeds),
                "pool_manifest_sha256": pool["manifest_sha256"],
                **inventory_summary,
                **archive_identity,
                "binding_sha256": PINNED_BINDING_SHA256,
                "simulator_source_sha256": simulator_source_sha,
                "armg_source_sha256": armg_sha,
            }, sort_keys=True))
            return 0

        sts = _load_sts(args.module_dir.resolve())
        g7_checkpoint = _materialize_g7_checkpoint(
            args.g7_actions_zip.resolve(), output_dir.parent / "pinned-g7" / "offline-champion.pt"
        )
        policy = ArmGNoncombatPolicy(root=args.armg_root.resolve(), weight_path=g7_checkpoint)
        output_dir.mkdir(parents=True, exist_ok=False)
        identity = {
            "round_id": ROUND_ID,
            "hypothesis": args.hypothesis,
            "stage": args.stage,
            "pool_id": pool["pool_id"],
            "pool_manifest_sha256": pool["manifest_sha256"],
            "candidate_commit": args.candidate_commit,
            "candidate_evaluator_sha256": evaluator_sha,
            "simulator_source_sha256": simulator_source_sha,
            "native_source_commit_label": NATIVE_SOURCE_COMMIT_LABEL,
            "binding_sha256": PINNED_BINDING_SHA256,
            "armg_source_sha256": armg_sha,
            "mcts_sims": MCTS_SIMS,
            "g7_checkpoint_sha256": G7_SHA256,
            **archive_identity,
            **inventory_summary,
        }
        _append_jsonl(usage_path, {
            "record_type": f"{args.hypothesis}_stage_start",
            **identity,
            "seed_ids": list(seeds),
            "private_output_dir": str(output_dir),
            "status": "RUNNING",
        })

        parent_outcomes: list[str] = []
        candidate_outcomes: list[str] = []
        candidate_overrides = 0
        candidate_reasons: dict[str, int] = {}
        trace_totals = _empty_trace_coverage()
        trace_off_passed: bool | None = None
        total_episodes = 0
        try:
            for index, seed in enumerate(seeds):
                pair_index = index + 1
                pair_metadata = {
                    **identity,
                    "episode_index": pair_index,
                    "pair_index": pair_index,
                    "seed_id": seed,
                }
                print(f"{args.hypothesis.upper()} {args.stage} paired seed {pair_index}/{len(seeds)}: baseline")
                parent_result, parent_path, _, parent_trace_stats = _run_episode(
                    sts=sts,
                    policy=policy,
                    seed=seed,
                    all_training_seeds=seeds,
                    output_dir=output_dir,
                    pair_index=pair_index,
                    arm="parent",
                    candidate=False,
                    trace_enabled=True,
                    metadata={**pair_metadata, "arm": "parent", "trace_mode": "on"},
                )
                total_episodes += 1
                print(f"H8 {args.stage} paired seed {pair_index}/{len(seeds)}: candidate")
                candidate_result, candidate_path, candidate_trace_path, candidate_trace_stats = _run_episode(
                    sts=sts,
                    policy=policy,
                    seed=seed,
                    all_training_seeds=seeds,
                    output_dir=output_dir,
                    pair_index=pair_index,
                    arm="candidate",
                    candidate=True,
                    trace_enabled=True,
                    metadata={**pair_metadata, "arm": "candidate", "trace_mode": "on"},
                )
                total_episodes += 1
                parent_outcomes.append(str(parent_result["outcome"]))
                candidate_outcomes.append(str(candidate_result["outcome"]))
                candidate_overrides += int(candidate_result.get("lethal_potion_rescue_override_count", 0))
                for reason, count in candidate_result.get("lethal_potion_rescue_reason_counts", {}).items():
                    candidate_reasons[str(reason)] = candidate_reasons.get(str(reason), 0) + int(count)
                _accumulate_trace_coverage(trace_totals, candidate_trace_stats)

                _append_jsonl(output_dir / "paired_results.private.jsonl", {
                    "pair_index": pair_index,
                    "seed_id": seed,
                    "parent_outcome": parent_result["outcome"],
                    "candidate_outcome": candidate_result["outcome"],
                    "parent_summary": h3._summary_view(parent_result),
                    "candidate_summary": h3._summary_view(candidate_result),
                    f"candidate_{args.hypothesis}_override_count": candidate_result.get("lethal_potion_rescue_override_count"),
                    f"candidate_{args.hypothesis}_reasons": candidate_result.get("lethal_potion_rescue_reason_counts"),
                    "parent_evidence": str(parent_path),
                    "candidate_evidence": str(candidate_path),
                    "candidate_trace": str(candidate_trace_path),
                })

                if args.stage == train_stage and index == 0:
                    trace_off_metadata = {**pair_metadata, "arm": "candidate", "trace_mode": "off"}
                    trace_off_result, trace_off_path, _, _ = _run_episode(
                        sts=sts,
                        policy=policy,
                        seed=seed,
                        all_training_seeds=seeds,
                        output_dir=output_dir,
                        pair_index=pair_index,
                        arm="candidate",
                        candidate=True,
                        trace_enabled=False,
                        metadata=trace_off_metadata,
                    )
                    total_episodes += 1
                    trace_off_passed = (
                        h3._summary_view(trace_off_result) == h3._summary_view(candidate_result)
                        and trace_off_result.get("lethal_potion_rescue_override_count")
                        == candidate_result.get("lethal_potion_rescue_override_count")
                        and h3._action_signature(trace_off_path) == h3._action_signature(candidate_path)
                    )
                    if not trace_off_passed:
                        raise EvaluationIntegrityError("candidate trace-on/off invariance failed")
                    _append_jsonl(output_dir / "paired_results.private.jsonl", {
                        "pair_index": pair_index,
                        "seed_id": seed,
                        "arm": "candidate_trace_off_invariance_replay",
                        "outcome": trace_off_result["outcome"],
                        "trace_invariance_passed": True,
                        "evidence": str(trace_off_path),
                    })

                if sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file()) > MAX_TOTAL_BYTES:
                    raise EvaluationIntegrityError("private H8 artifact set exceeded the registered total bound")

            paired = h3._paired_summary_for_stage(parent_outcomes, candidate_outcomes)
            status = "COMPLETE"
            override_count_field = f"{args.hypothesis}_override_count"
            override_reasons_field = f"{args.hypothesis}_override_reason_counts"
            stage_summary = {
                "record_type": f"{args.hypothesis}_stage_summary",
                **identity,
                "status": status,
                "seed_count": len(seeds),
                "episodes": total_episodes,
                "parent_outcomes": _outcome_counts(parent_outcomes),
                "candidate_outcomes": _outcome_counts(candidate_outcomes),
                "paired": paired,
                override_count_field: candidate_overrides,
                override_reasons_field: candidate_reasons,
                "trace_coverage": trace_totals,
                "trace_on_off_invariance": trace_off_passed,
                "safety": {
                    "illegal_actions": 0,
                    "crashes": 0,
                    "timeouts": 0,
                    "communication_errors": 0,
                    "all_terminal_records_complete": True,
                    "legal_actions_complete": True,
                },
            }
            _append_jsonl(usage_path, stage_summary)
            print(json.dumps({
                "status": status,
                "stage": args.stage,
                "seed_count": len(seeds),
                "episodes": total_episodes,
                "parent_outcomes": stage_summary["parent_outcomes"],
                "candidate_outcomes": stage_summary["candidate_outcomes"],
                "paired": paired,
                override_count_field: candidate_overrides,
                override_reasons_field: candidate_reasons,
                "trace_coverage": trace_totals,
                "trace_on_off_invariance": trace_off_passed,
                "candidate_commit": args.candidate_commit,
                "simulator_source_sha256": simulator_source_sha,
                "binding_sha256": PINNED_BINDING_SHA256,
            }, sort_keys=True))
            return 0
        except Exception as exc:
            _append_jsonl(usage_path, {
                "record_type": f"{args.hypothesis}_stage_failure",
                **identity,
                "status": "NOT_VERIFIED",
                "completed_pairs": len(parent_outcomes),
                "episodes_attempted": total_episodes,
                "error_type": type(exc).__name__,
                "private_output_dir": str(output_dir),
            })
            raise
    except Exception as exc:
        print(json.dumps({"status": "NOT_VERIFIED", "error_type": type(exc).__name__}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
