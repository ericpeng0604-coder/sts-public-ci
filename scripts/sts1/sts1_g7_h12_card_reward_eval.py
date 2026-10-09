"""Run the preregistered H12/H13 card-reward counterfactual on private seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sts1"))

import sts1_g7_h3_emergency_potion_eval as h3  # noqa: E402
import sts1_g7_h7_potion_trace_audit as h7  # noqa: E402
import sts1_g7_h8_lethal_potion_eval as h8  # noqa: E402
from sts1_g7_seed_ledger import MAX_SEED, sha256_json, validate_inventory  # noqa: E402


ROUND_ID = "round-007-20261009"
HYPOTHESES = {
    "h12": {
        "stage": "train_hypothesis_1",
        "registration_key": "experiment_registration",
        "registration_id": "H12_card_reward_skip_deck30",
        "duplicate_only": False,
        "require_all_options_duplicate": False,
    },
    "h13": {
        "stage": "train_hypothesis_2",
        "registration_key": "next_experiment_registration",
        "registration_id": "H13_duplicate_card_skip_deck30",
        "duplicate_only": True,
        "require_all_options_duplicate": False,
    },
    "h14": {
        "stage": "train_hypothesis_3",
        "registration_key": "third_experiment_registration",
        "registration_id": "H14_duplicate_skip_only_when_all_options_duplicate_deck30",
        "duplicate_only": True,
        "require_all_options_duplicate": True,
    },
}
POOL_COUNTS = {
    "train_hypothesis_1": 10,
    "train_hypothesis_2": 10,
    "train_hypothesis_3": 10,
    "probe": 10,
    "dev": 30,
}
MCTS_SIMS = 2000
G7_SHA256 = h8.G7_SHA256
PINNED_BINDING_SHA256 = h8.PINNED_BINDING_SHA256
SKIP_DECK_THRESHOLD = 30


class EvaluationIntegrityError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationIntegrityError(f"expected a JSON object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            [
                "rtk",
                "proxy",
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationIntegrityError("local candidate HEAD could not be verified") from exc


def _git_blob(commit: str, relative_path: str) -> bytes:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise EvaluationIntegrityError("candidate commit must be a full Git SHA")
    try:
        return subprocess.check_output(
            [
                "rtk",
                "proxy",
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "show",
                f"{commit}:{relative_path}",
            ],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationIntegrityError("frozen candidate Git object could not be verified") from exc


def _frozen_candidate_hashes(commit: str) -> tuple[str, str]:
    simulator_rel = "src/roguelike_ai/sts1_phase3/simulator.py"
    evaluator_rel = "scripts/sts1/sts1_g7_h12_card_reward_eval.py"
    simulator_blob = _git_blob(commit, simulator_rel)
    evaluator_blob = _git_blob(commit, evaluator_rel)
    return hashlib.sha256(simulator_blob).hexdigest(), hashlib.sha256(evaluator_blob).hexdigest()


def _require_candidate_ancestor(commit: str) -> None:
    try:
        subprocess.run(
            [
                "rtk",
                "proxy",
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "merge-base",
                "--is-ancestor",
                commit,
                _git_head(),
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationIntegrityError("frozen candidate commit is not an ancestor of the evaluation revision") from exc


def _pool_key_for_stage(hypothesis: str, stage: str) -> str:
    spec = HYPOTHESES.get(hypothesis)
    if not isinstance(spec, dict):
        raise EvaluationIntegrityError("unknown Round007 hypothesis")
    if stage == "train":
        return str(spec["stage"])
    if hypothesis != "h13" or stage not in {"probe", "dev"}:
        raise EvaluationIntegrityError("only the frozen H13 candidate may use registered held-out stages")
    return stage


def _seed_contract_kwargs(stage: str, seeds: tuple[int, ...]) -> dict[str, Any]:
    if not seeds:
        raise EvaluationIntegrityError("seed contract requires a nonempty registered pool")
    if stage == "train":
        return {"heldout_seeds": None, "training_seeds": seeds}
    if stage in {"probe", "dev"}:
        return {"heldout_seeds": seeds, "training_seeds": None}
    raise EvaluationIntegrityError("seed contract requested for an unregistered stage")


def _validate_round7_pool(
    *,
    public_summary_path: Path,
    pools_dir: Path,
    inventory_path: Path,
    hypothesis: str = "h12",
    stage: str = "train",
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    spec = HYPOTHESES.get(hypothesis)
    if not isinstance(spec, dict):
        raise EvaluationIntegrityError("unknown Round007 hypothesis")
    summary = _read_json(public_summary_path)
    if summary.get("round_id") != ROUND_ID or summary.get("record_type") != "exploration_round":
        raise EvaluationIntegrityError("public Round007 summary identity mismatch")
    registration = summary.get(spec["registration_key"])
    if not isinstance(registration, dict) or registration.get("id") != spec["registration_id"]:
        raise EvaluationIntegrityError("selected experiment is not registered in the public seed ledger")
    selected_pool_key = _pool_key_for_stage(hypothesis, stage)
    if stage == "train":
        if registration.get("status") != "REGISTERED_NOT_RUN":
            raise EvaluationIntegrityError("selected train experiment is not in its pre-run state")
    else:
        expected_parent_status = "COMPLETE_PROBE_ALLOWED" if stage == "probe" else "COMPLETE_DEV_ALLOWED"
        stage_registration = registration.get(f"{stage}_registration")
        if (
            registration.get("status") != expected_parent_status
            or not isinstance(stage_registration, dict)
            or stage_registration.get("status") != "REGISTERED_NOT_RUN"
            or stage_registration.get("pool_key") != selected_pool_key
        ):
            raise EvaluationIntegrityError("held-out stage lacks a matching pre-run registration")

    ledger = _read_json(pools_dir / "ledger.json")
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    ledger_sha = ledger.get("ledger_sha256")
    if not isinstance(ledger_sha, str) or sha256_json(ledger_payload) != ledger_sha:
        raise EvaluationIntegrityError("private Round007 ledger hash mismatch")
    if ledger.get("round_id") != ROUND_ID or summary.get("manifest_sha256") != ledger_sha:
        raise EvaluationIntegrityError("public and private Round007 ledgers differ")

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
        raise EvaluationIntegrityError("Round007 pool registries are incomplete")
    if set(public_pools) != set(POOL_COUNTS) or set(private_pools) != set(POOL_COUNTS):
        raise EvaluationIntegrityError("Round007 pool registry does not match the preregistered schema")

    seen: set[int] = set()
    selected_pool: dict[str, Any] | None = None
    selected_seeds: tuple[int, ...] | None = None
    for registered_pool_key, expected_count in POOL_COUNTS.items():
        pool_path = pools_dir / f"{registered_pool_key}.json"
        pool = _read_json(pool_path)
        pool_payload = {key: value for key, value in pool.items() if key != "manifest_sha256"}
        pool_hash = pool.get("manifest_sha256")
        seed_ids = pool.get("seed_ids")
        expected_purpose = (
            "dev" if registered_pool_key == "dev"
            else "probe" if registered_pool_key == "probe"
            else "train"
        )
        if (
            not isinstance(pool_hash, str)
            or sha256_json(pool_payload) != pool_hash
            or pool.get("pool_id") != f"{ROUND_ID}-{registered_pool_key}"
            or pool.get("round_id") != ROUND_ID
            or pool.get("purpose") != expected_purpose
            or pool.get("role") != registered_pool_key
            or pool.get("status") != "GENERATED_NOT_RUN"
            or not isinstance(seed_ids, list)
            or len(seed_ids) != expected_count
            or any(
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not 1 <= seed <= MAX_SEED
                for seed in seed_ids
            )
            or len(set(seed_ids)) != expected_count
        ):
            raise EvaluationIntegrityError("private Round007 pool manifest is invalid or already used")
        public_entry = public_pools.get(registered_pool_key)
        if (
            not isinstance(public_entry, dict)
            or public_entry.get("pool_id") != pool.get("pool_id")
            or public_entry.get("purpose") != expected_purpose
            or public_entry.get("count") != expected_count
            or public_entry.get("manifest_sha256") != pool_hash
            or public_entry.get("status") != "GENERATED_NOT_RUN"
        ):
            raise EvaluationIntegrityError("public and private pool manifests differ")
        if (
            pool.get("inventory_id") != inventory.get("inventory_id")
            or pool.get("inventory_sha256") != inventory_sha
            or pool.get("source_audit_sha256") != sha256_json(source_audit)
            or pool.get("source_audit") != source_audit
        ):
            raise EvaluationIntegrityError("pool is not bound to the complete exclusion inventory")
        current = set(seed_ids)
        if current & excluded or current & seen:
            raise EvaluationIntegrityError("Round007 pools overlap prior or protected seeds")
        seen.update(current)
        if registered_pool_key == selected_pool_key:
            selected_pool = pool
            selected_seeds = tuple(seed_ids)

    if selected_pool is None or selected_seeds is None:
        raise EvaluationIntegrityError("selected registered stage pool is missing")
    return selected_pool, selected_seeds, {
        "seed_count": len(selected_seeds),
        "selected_pool_key": selected_pool_key,
        "pool_count": len(POOL_COUNTS),
        "exclusion_unique_seed_count": len(excluded),
        "inventory_sha256": inventory_sha,
        "private_ledger_sha256": ledger_sha,
        "disjoint_from_exclusion_inventory": True,
        "round_pools_pairwise_disjoint": True,
    }


def _run_episode(
    *,
    sts: Any,
    policy: Any,
    seed: int,
    seed_pool: tuple[int, ...],
    seed_stage: str,
    output_dir: Path,
    pair_index: int,
    arm: str,
    candidate: bool,
    trace_enabled: bool,
    metadata: dict[str, Any],
    hypothesis: str = "h12",
) -> tuple[dict[str, Any], Path, Path | None, dict[str, int] | None]:
    stem = f"pair-{pair_index:02d}-{arm}{'-trace-off' if not trace_enabled else ''}"
    evidence_path = output_dir / f"{stem}.evidence.ndjson"
    trace_path = output_dir / f"{stem}.trace.ndjson" if trace_enabled else None
    if evidence_path.exists() or (trace_path is not None and trace_path.exists()):
        raise EvaluationIntegrityError("episode output already exists; refusing to rerun a seed")

    spec = HYPOTHESES[hypothesis]
    duplicate_only = bool(spec["duplicate_only"])
    duplicate_intervention = candidate and duplicate_only
    require_all_options_duplicate = bool(spec["require_all_options_duplicate"])
    result = h8.run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=evidence_path,
        armg_policy=policy,
        combat_mcts_sims=MCTS_SIMS,
        reserve_last_potion_until_floor=None,
        use_potion_below_hp_fraction=None,
        lethal_potion_rescue=False,
        avoid_low_hp_elite_routes=False,
        prefer_smith_when_rest_overheals=False,
        prefer_smith_when_overheal_exceeds_effective_rest_heal=False,
        skip_card_reward_when_deck_size_at_least=(
            SKIP_DECK_THRESHOLD if candidate and not duplicate_only else None
        ),
        skip_duplicate_card_reward_when_deck_size_at_least=(
            SKIP_DECK_THRESHOLD if duplicate_intervention else None
        ),
        require_all_card_reward_options_are_duplicates=(
            require_all_options_duplicate and duplicate_intervention
        ),
        **_seed_contract_kwargs(seed_stage, seed_pool),
        collect_ppo=False,
        collect_teacher=False,
        diagnostic_trace_path=trace_path,
        diagnostic_metadata=metadata if trace_enabled else None,
    )
    h8._check_pair_integrity(evidence_path, result)
    expected_seed_contract = "training_internal" if seed_stage == "train" else "heldout_internal"
    if result.get("seed_contract") != expected_seed_contract:
        raise EvaluationIntegrityError("simulator seed contract does not match the registered stage")
    expected_broad_threshold = SKIP_DECK_THRESHOLD if candidate and not duplicate_only else None
    expected_duplicate_threshold = SKIP_DECK_THRESHOLD if duplicate_intervention else None
    expected_all_options_duplicate = require_all_options_duplicate and duplicate_intervention
    if (
        result.get("skip_card_reward_when_deck_size_at_least") != expected_broad_threshold
        or result.get("skip_duplicate_card_reward_when_deck_size_at_least")
        != expected_duplicate_threshold
        or result.get("require_all_card_reward_options_are_duplicates")
        is not expected_all_options_duplicate
    ):
        raise EvaluationIntegrityError("card-reward intervention flag differs from the selected arm")

    trace_stats = None
    if trace_path is not None:
        trace_stats = h7._validate_trace(trace_path, result, metadata)
        events = [
            row
            for row in h8._read_jsonl(trace_path)
            if row.get("type") == "noncombat_decision_trace_v1"
        ]
        expected_mode = expected_duplicate_threshold is not None
        if expected_broad_threshold is not None or expected_duplicate_threshold is not None:
            for row in events:
                intervention = row.get("card_reward_skip_intervention")
                if (
                    not isinstance(intervention, dict)
                    or intervention.get("require_recommended_duplicate") is not expected_mode
                    or intervention.get("require_all_options_duplicate")
                    is not expected_all_options_duplicate
                    or (
                        intervention.get("overridden") is True
                        and expected_mode
                        and intervention.get("recommended_card_is_duplicate") is not True
                    )
                    or (
                        intervention.get("overridden") is True
                        and expected_all_options_duplicate
                        and intervention.get("all_reward_options_are_duplicates") is not True
                    )
                ):
                    raise EvaluationIntegrityError("trace intervention mode differs from the registered candidate")
        overrides = sum(
            isinstance(row.get("card_reward_skip_intervention"), dict)
            and row["card_reward_skip_intervention"].get("overridden") is True
            for row in events
        )
        eligible = sum(
            isinstance(row.get("card_reward_skip_intervention"), dict)
            and row["card_reward_skip_intervention"].get("eligible") is True
            for row in events
        )
        if overrides != result.get("card_reward_skip_override_count"):
            raise EvaluationIntegrityError("trace and summary override counts differ")
        if eligible != result.get("card_reward_skip_eligible_count"):
            raise EvaluationIntegrityError("trace and summary eligibility counts differ")
        if not candidate and (overrides != 0 or eligible != 0):
            raise EvaluationIntegrityError("parent arm unexpectedly activated a card-reward intervention")
    for artifact in (evidence_path, trace_path):
        if artifact is not None and artifact.stat().st_size > h8.MAX_EPISODE_BYTES:
            raise EvaluationIntegrityError("per-episode evidence exceeded its registered bound")
    return result, evidence_path, trace_path, trace_stats


def _manifest_private_artifacts(output_dir: Path) -> tuple[dict[str, Any], str]:
    entries = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "artifact-manifest.json":
            continue
        entries.append({
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    payload = {"files": entries, "file_count": len(entries), "total_bytes": sum(x["size_bytes"] for x in entries)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return payload, digest


def _aggregate_safety(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise EvaluationIntegrityError("safety totals require completed episode summaries")
    totals = {
        "illegal_actions": 0,
        "crashes": 0,
        "timeouts": 0,
        "communication_errors": 0,
    }
    fields = {
        "illegal_action_count": "illegal_actions",
        "crash_count": "crashes",
        "timeout_count": "timeouts",
    }
    for result in results:
        for source, target in fields.items():
            value = result.get(source)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                raise EvaluationIntegrityError("episode safety counter was nonzero or unavailable")
            totals[target] += value
        if "communication_error_count" in result:
            value = result["communication_error_count"]
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                raise EvaluationIntegrityError("communication error counter was nonzero or unavailable")
            totals["communication_errors"] += value
    return {
        **totals,
        "communication_error_basis": "0; local in-process simulator evaluation has no remote/game-bridge communication transport",
        "all_terminal_records_complete": True,
        "legal_actions_complete": True,
        "validated_episode_count": len(results),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis", choices=tuple(HYPOTHESES), default="h12")
    parser.add_argument("--stage", choices=("train", "probe", "dev"), default="train")
    parser.add_argument("--public-summary", type=Path, required=True)
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
    hypothesis = args.hypothesis
    spec = HYPOTHESES[hypothesis]
    stage = args.stage
    seed_stage = "train" if stage == "train" else stage
    try:
        if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
            raise EvaluationIntegrityError("contextual reranking must be disabled for paired evaluation")
        selected_pool_key = _pool_key_for_stage(hypothesis, stage)
        _require_candidate_ancestor(args.candidate_commit)
        public_summary = args.public_summary.resolve()
        pools_dir = args.pools_dir.resolve()
        inventory_path = args.exclusion_inventory.resolve()
        pool, seeds, inventory_identity = _validate_round7_pool(
            public_summary_path=public_summary,
            pools_dir=pools_dir,
            inventory_path=inventory_path,
            hypothesis=hypothesis,
            stage=stage,
        )
        binding = args.module_dir.resolve() / "slaythespire.cp312-win_amd64.pyd"
        if not binding.is_file() or _sha256(binding) != PINNED_BINDING_SHA256:
            raise EvaluationIntegrityError("pinned native simulator binding hash mismatch")
        archive = args.g7_actions_zip.resolve()
        archive_identity = h8._check_actions_archive(archive)
        if archive_identity.get("actions_zip_sha256") != h8.ACTIONS_ZIP_SHA256:
            raise EvaluationIntegrityError("pinned G7 Actions artifact archive hash mismatch")
        armg_source = args.armg_root.resolve() / "armG_train.py"
        if not armg_source.is_file():
            raise EvaluationIntegrityError("pinned ArmG source file is missing")
        source_path = REPO_ROOT / "src" / "roguelike_ai" / "sts1_phase3" / "simulator.py"
        frozen_source_sha, frozen_evaluator_sha = _frozen_candidate_hashes(args.candidate_commit)
        current_source_sha = _sha256(source_path)
        if current_source_sha != frozen_source_sha:
            raise EvaluationIntegrityError("working simulator source differs from the frozen candidate")
        stage_evaluator_sha = _sha256(Path(__file__).resolve())
        identity = {
            "round_id": ROUND_ID,
            "hypothesis": hypothesis,
            "experiment_id": spec["registration_id"],
            "stage": str(spec["stage"]) if stage == "train" else stage,
            "pool_id": pool["pool_id"],
            "pool_manifest_sha256": pool["manifest_sha256"],
            "candidate_commit": args.candidate_commit,
            "candidate_source_sha256": frozen_source_sha,
            "candidate_evaluator_sha256": frozen_evaluator_sha,
            "stage_evaluator_sha256": stage_evaluator_sha,
            "g7_checkpoint_sha256": G7_SHA256,
            "simulator_source_sha256": current_source_sha,
            "simulator_native_source_commit_label": "7476a81954020087da31d41d16fddf475746ec2d",
            "binding_sha256": _sha256(binding),
            "armg_source_sha256": _sha256(armg_source),
            "g7_actions_zip_sha256": archive_identity["actions_zip_sha256"],
            "mcts_sims": MCTS_SIMS,
            "seed_contract": "training_internal" if seed_stage == "train" else "heldout_internal",
            "selected_pool_key": selected_pool_key,
            **inventory_identity,
        }
        summary_doc = _read_json(public_summary)
        registration = summary_doc.get(spec["registration_key"])
        if not isinstance(registration, dict):
            raise EvaluationIntegrityError("public experiment registration is missing")
        if stage == "train":
            registered_commit = registration.get("candidate_commit")
            if registered_commit is not None and registered_commit != args.candidate_commit:
                raise EvaluationIntegrityError("train candidate commit differs from its preregistration")
        else:
            stage_registration = registration.get(f"{stage}_registration")
            if not isinstance(stage_registration, dict) or any(
                stage_registration.get(key) != identity.get(identity_key)
                for key, identity_key in (
                    ("candidate_commit", "candidate_commit"),
                    ("candidate_source_sha256", "candidate_source_sha256"),
                    ("candidate_evaluator_sha256", "candidate_evaluator_sha256"),
                    ("stage_evaluator_sha256", "stage_evaluator_sha256"),
                    ("pool_manifest_sha256", "pool_manifest_sha256"),
                    ("seed_contract", "seed_contract"),
                )
            ):
                raise EvaluationIntegrityError("held-out candidate or evaluator differs from preregistration")
            if (
                stage_registration.get("mcts_sims") != MCTS_SIMS
                or stage_registration.get("seed_count") != len(seeds)
                or stage_registration.get("episodes_planned") != 2 * len(seeds) + 1
            ):
                raise EvaluationIntegrityError("held-out pool size or MCTS budget differs from preregistration")
        output_dir = args.private_output_dir.resolve()
        usage_path = args.private_usage_ledger.resolve()
        repo = REPO_ROOT.resolve()
        if any(path == repo or repo in path.parents for path in (output_dir, usage_path)):
            raise EvaluationIntegrityError("raw run artifacts and usage ledger must stay outside the repository")
        if output_dir.exists():
            raise EvaluationIntegrityError("private output already exists; refusing to rerun this pool")
        if usage_path.exists():
            for record in h8._read_jsonl(usage_path):
                if record.get("pool_id") == pool["pool_id"]:
                    raise EvaluationIntegrityError("selected train pool already has a private use record")
        if args.preflight_only:
            print(json.dumps({"status": "PREFLIGHT_PASS", **identity, "seed_ids_emitted": False}, sort_keys=True))
            return 0

        output_dir.mkdir(parents=True, exist_ok=False)
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        h8._append_jsonl(usage_path, {
            "record_type": f"{hypothesis}_{stage}_stage_start",
            **identity,
            "seed_ids": list(seeds),
            "episodes_planned": 2 * len(seeds) + 1,
            "status": "RUNNING",
        })
        checkpoint = h8._materialize_g7_checkpoint(
            archive,
            output_dir / "pinned-g7" / "offline-champion.pt",
        )
        if _sha256(checkpoint) != G7_SHA256:
            raise EvaluationIntegrityError("materialized parent checkpoint is not pinned G7")

        sts = h8._load_sts(args.module_dir.resolve())
        policy = h8.ArmGNoncombatPolicy(root=args.armg_root.resolve(), weight_path=checkpoint)
        parent_outcomes: list[str] = []
        candidate_outcomes: list[str] = []
        paired_rows: list[dict[str, Any]] = []
        trace_totals = h8._empty_trace_coverage()
        safety_results: list[dict[str, Any]] = []
        validated_trace_count = 0
        candidate_overrides = 0
        candidate_eligible = 0
        candidate_reasons: dict[str, int] = {}
        retention_floor_lower = 0
        retention_hp_lower = 0
        trace_off_passed: bool | None = None
        episode_count = 0

        for index, seed in enumerate(seeds):
            pair_index = index + 1
            pair_metadata = {
                **identity,
                "episode_index": pair_index,
                "pair_index": pair_index,
                "seed_id": seed,
                "seed_purpose": seed_stage,
            }
            print(f"{hypothesis.upper()} paired episode starting {pair_index} / {len(seeds)}")
            parent_result, parent_path, parent_trace_path, parent_trace_stats = _run_episode(
                sts=sts,
                policy=policy,
                seed=seed,
                seed_pool=seeds,
                seed_stage=seed_stage,
                output_dir=output_dir,
                pair_index=pair_index,
                arm="parent",
                candidate=False,
                trace_enabled=True,
                metadata={**pair_metadata, "arm": "parent", "trace_mode": "on"},
                hypothesis=hypothesis,
            )
            episode_count += 1
            candidate_result, candidate_path, candidate_trace_path, candidate_trace_stats = _run_episode(
                sts=sts,
                policy=policy,
                seed=seed,
                seed_pool=seeds,
                seed_stage=seed_stage,
                output_dir=output_dir,
                pair_index=pair_index,
                arm="candidate",
                candidate=True,
                trace_enabled=True,
                metadata={**pair_metadata, "arm": "candidate", "trace_mode": "on"},
                hypothesis=hypothesis,
            )
            episode_count += 1
            if parent_trace_path is None or parent_trace_stats is None:
                raise EvaluationIntegrityError("parent-arm trace validation is missing")
            h8._accumulate_trace_coverage(trace_totals, parent_trace_stats)
            validated_trace_count += 1
            if candidate_trace_path is None or candidate_trace_stats is None:
                raise EvaluationIntegrityError("candidate-arm trace validation is missing")
            validated_trace_count += 1
            safety_results.extend((parent_result, candidate_result))
            parent_outcomes.append(str(parent_result["outcome"]))
            candidate_outcomes.append(str(candidate_result["outcome"]))
            candidate_overrides += int(candidate_result.get("card_reward_skip_override_count", 0))
            candidate_eligible += int(candidate_result.get("card_reward_skip_eligible_count", 0))
            for reason, count in candidate_result.get("card_reward_skip_reason_counts", {}).items():
                candidate_reasons[str(reason)] = candidate_reasons.get(str(reason), 0) + int(count)
            h8._accumulate_trace_coverage(trace_totals, candidate_trace_stats)

            parent_view = h3._summary_view(parent_result)
            candidate_view = h3._summary_view(candidate_result)
            if parent_result.get("final_floor") is None or candidate_result.get("final_floor") is None:
                raise EvaluationIntegrityError("terminal floor is missing from a complete episode")
            if parent_result.get("final_hp") is None or candidate_result.get("final_hp") is None:
                raise EvaluationIntegrityError("final HP is missing from a complete episode")
            floor_lower = int(candidate_result["final_floor"]) < int(parent_result["final_floor"])
            hp_lower = int(candidate_result["final_hp"]) < int(parent_result["final_hp"])
            retention_floor_lower += int(floor_lower)
            retention_hp_lower += int(hp_lower)
            paired_rows.append({
                "pair_index": pair_index,
                "seed_id": seed,
                "parent_outcome": parent_result["outcome"],
                "candidate_outcome": candidate_result["outcome"],
                "parent_summary": parent_view,
                "candidate_summary": candidate_view,
                "candidate_skip_override_count": candidate_result.get("card_reward_skip_override_count"),
                "candidate_skip_eligible_count": candidate_result.get("card_reward_skip_eligible_count"),
                "retention_floor_lower": floor_lower,
                "retention_hp_lower": hp_lower,
                "parent_evidence": str(parent_path),
                "candidate_evidence": str(candidate_path),
                "candidate_trace": str(candidate_trace_path),
            })
            h8._append_jsonl(output_dir / "paired_results.private.jsonl", paired_rows[-1])

            if index == 0:
                trace_off_metadata = {**pair_metadata, "arm": "candidate", "trace_mode": "off"}
                trace_off_result, trace_off_path, _, _ = _run_episode(
                    sts=sts,
                    policy=policy,
                    seed=seed,
                    seed_pool=seeds,
                    seed_stage=seed_stage,
                    output_dir=output_dir,
                    pair_index=pair_index,
                    arm="candidate",
                    candidate=True,
                    trace_enabled=False,
                    metadata=trace_off_metadata,
                    hypothesis=hypothesis,
                )
                episode_count += 1
                safety_results.append(trace_off_result)
                trace_off_passed = (
                    h3._summary_view(trace_off_result) == candidate_view
                    and trace_off_result.get("card_reward_skip_override_count")
                    == candidate_result.get("card_reward_skip_override_count")
                    and trace_off_result.get("card_reward_skip_eligible_count")
                    == candidate_result.get("card_reward_skip_eligible_count")
                    and h3._action_signature(trace_off_path) == h3._action_signature(candidate_path)
                )
                if not trace_off_passed:
                    raise EvaluationIntegrityError("candidate trace-on/off invariance failed")
                h8._append_jsonl(output_dir / "trace_invariance.private.jsonl", {
                    "pair_index": pair_index,
                    "seed_id": seed,
                    "candidate_trace_on_off_equal": True,
                    "trace_off_evidence": str(trace_off_path),
                })

            print(f"{hypothesis.upper()} paired episode completed {pair_index} / {len(seeds)}")
            if sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()) > h8.MAX_TOTAL_BYTES:
                raise EvaluationIntegrityError("private artifact set exceeded its registered total bound")

        paired = h3._paired_summary_for_stage(parent_outcomes, candidate_outcomes)
        expected_episodes = 2 * len(seeds) + 1
        expected_traces = 2 * len(seeds)
        if (
            episode_count != expected_episodes
            or len(safety_results) != episode_count
            or validated_trace_count != expected_traces
        ):
            raise EvaluationIntegrityError("planned stage episodes or traced legal-action coverage is incomplete")
        safety = _aggregate_safety(safety_results)
        safety_passed = bool(
            safety["illegal_actions"] == 0
            and safety["crashes"] == 0
            and safety["timeouts"] == 0
            and safety["communication_errors"] == 0
            and safety["all_terminal_records_complete"] is True
            and safety["legal_actions_complete"] is True
            and safety["validated_episode_count"] == expected_episodes
        )
        eligible = candidate_eligible > 0
        retention_passed = retention_floor_lower == 0 and retention_hp_lower == 0
        stage_net_nonnegative = int(paired["net_wins"]) >= 0
        progress_allowed = bool(
            eligible and retention_passed and stage_net_nonnegative and trace_off_passed and safety_passed
        )
        if stage == "train":
            status = "COMPLETE_PROBE_ALLOWED" if progress_allowed else "COMPLETE_NO_PROBE"
        elif stage == "probe":
            status = "COMPLETE_DEV_ALLOWED" if progress_allowed else "COMPLETE_NO_DEV"
        else:
            status = (
                "COMPLETE_DEV_POSITIVE_SIGNAL"
                if eligible and retention_passed and trace_off_passed and safety_passed and int(paired["net_wins"]) > 0
                else "COMPLETE_DEV_NO_POSITIVE_SIGNAL"
            )
        stage_summary = {
            "record_type": f"{hypothesis}_{stage}_stage_summary",
            **identity,
            "status": status,
            "seed_count": len(seeds),
            "episodes": episode_count,
            "seed_contract": "training_internal" if seed_stage == "train" else "heldout_internal",
            "parent_outcomes": h8._outcome_counts(parent_outcomes),
            "candidate_outcomes": h8._outcome_counts(candidate_outcomes),
            "paired": paired,
            "card_reward_skip_override_count": candidate_overrides,
            "card_reward_skip_eligible_count": candidate_eligible,
            "card_reward_skip_reason_counts": candidate_reasons,
            "trace_coverage": trace_totals,
            "trace_on_off_invariance": trace_off_passed,
            "retention": {
                "candidate_lower_terminal_floor_pairs": retention_floor_lower,
                "candidate_lower_final_hp_pairs": retention_hp_lower,
                "strict_no_floor_or_hp_regression_passed": retention_passed,
            },
            "safety": safety,
            "stage_gate": {
                "eligible_action_coverage": eligible,
                "stage_net_nonnegative": stage_net_nonnegative,
                "retention_passed": retention_passed,
                "trace_on_off_passed": bool(trace_off_passed),
                "safety_passed": safety_passed,
                "allowed_next_stage": progress_allowed if stage in {"train", "probe"} else False,
            },
        }
        h8._append_jsonl(args.private_usage_ledger.resolve(), stage_summary)
        manifest, manifest_sha = _manifest_private_artifacts(output_dir)
        manifest_record = {**manifest, "manifest_sha256": manifest_sha}
        (output_dir / "artifact-manifest.json").write_text(
            json.dumps(manifest_record, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": status,
            "round_id": ROUND_ID,
            "hypothesis": hypothesis,
            "stage": str(spec["stage"]) if stage == "train" else stage,
            "episodes": episode_count,
            "parent_outcomes": stage_summary["parent_outcomes"],
            "candidate_outcomes": stage_summary["candidate_outcomes"],
            "paired": paired,
            "card_reward_skip_override_count": candidate_overrides,
            "card_reward_skip_eligible_count": candidate_eligible,
            "retention": stage_summary["retention"],
            "progression_allowed": progress_allowed,
            "trace_coverage": trace_totals,
            "private_artifact_count": manifest["file_count"],
            "private_artifact_bytes": manifest["total_bytes"],
            "private_artifact_manifest_sha256": manifest_sha,
            "seed_ids_emitted": False,
        }, sort_keys=True))
        return 0
    except Exception as exc:
        try:
            if args.private_usage_ledger:
                h8._append_jsonl(args.private_usage_ledger.resolve(), {
                    "record_type": f"{args.hypothesis}_failure",
                    "round_id": ROUND_ID,
                    "hypothesis": args.hypothesis,
                    "stage": args.stage,
                    "status": "NOT_VERIFIED",
                    "exception_type": type(exc).__name__,
                    "candidate_commit": args.candidate_commit,
                })
        except Exception:
            pass
        print(json.dumps({"status": "NOT_VERIFIED", "exception_type": type(exc).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
