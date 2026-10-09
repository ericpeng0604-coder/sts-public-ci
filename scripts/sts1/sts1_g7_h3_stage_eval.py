"""Evaluate the frozen H3 policy on fresh Probe10 or conditional Dev30 pools."""

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

import sts1_g7_h3_emergency_potion_eval as h3  # noqa: E402
from roguelike_ai.sts1_phase3 import simulator as simulator_module  # noqa: E402
from roguelike_ai.sts1_phase3.simulator import (  # noqa: E402
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)
from sts1_g7_seed_ledger import MAX_SEED, sha256_json, validate_inventory  # noqa: E402


FROZEN_CANDIDATE_COMMIT = "009e233bb38f71a2df3aea611db0b8ab9b664aa1"
FROZEN_SIMULATOR_SOURCE_SHA256 = "26779cc79d30e91a9da37630aa72fc5d006dcd6ae5238070e0ff87dcecf076ee"
G7_SHA256 = h3.G7_SHA256
SIMULATOR_COMMIT = h3.SIMULATOR_COMMIT
SIMULATOR_BINDING_SHA256 = h3.SIMULATOR_BINDING_SHA256
ARMG_SOURCE_SHA256 = "7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b"
MCTS_SIMS = 2000
HP_RATIO_THRESHOLD = 0.5

STAGES: dict[str, dict[str, Any]] = {
    "probe": {
        "pool_key": "probe",
        "pool_file": "probe.json",
        "pool_id": "round-003-20261009-probe",
        "purpose": "probe",
        "manifest_sha256": "d6ed4a75d4fca4b0df414572e1effe8585778b2be6cb7f88995b002e3b7ba477",
        "seed_count": 10,
    },
    "dev": {
        "pool_key": "dev",
        "pool_file": "dev.json",
        "pool_id": "round-003-20261009-dev",
        "purpose": "dev",
        "manifest_sha256": "1c5902006c427aee215e7b79eca65e58cf489ef125b18113a4b99c72e007d856",
        "seed_count": 30,
    },
}


class StageEvaluationError(RuntimeError):
    """A frozen held-out evaluation failed a precondition or guard."""


def _stage_spec(stage: str) -> dict[str, Any]:
    try:
        return STAGES[stage]
    except KeyError as exc:
        raise StageEvaluationError("only the pre-registered probe or dev stage is accepted") from exc


def _seed_contract_kwargs(stage: str, seeds: tuple[int, ...]) -> dict[str, Any]:
    if stage not in STAGES or not seeds:
        raise StageEvaluationError("held-out seed contract requires a registered stage and nonempty pool")
    return {"heldout_seeds": seeds, "training_seeds": None}


def _paired_summary(parent: list[str], candidate: list[str]) -> dict[str, Any]:
    if not parent or len(parent) != len(candidate):
        raise StageEvaluationError("paired held-out evaluation requires equal nonempty outcome lists")
    if any(value not in {"victory", "defeat"} for value in parent + candidate):
        raise StageEvaluationError("unknown outcomes cannot be counted as losses")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pool(
    stage: str,
    pool_path: Path,
    pools_dir: Path,
    inventory_path: Path,
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    spec = _stage_spec(stage)
    if pool_path.name != spec["pool_file"]:
        raise StageEvaluationError("pool file does not match the registered stage")
    pool = h3.shared._read_json(pool_path)
    payload = {key: value for key, value in pool.items() if key != "manifest_sha256"}
    if (
        pool.get("pool_id") != spec["pool_id"]
        or pool.get("purpose") != spec["purpose"]
        or pool.get("status") != "GENERATED_NOT_RUN"
        or pool.get("manifest_sha256") != spec["manifest_sha256"]
        or sha256_json(payload) != spec["manifest_sha256"]
    ):
        raise StageEvaluationError("held-out pool identity, status, or manifest hash mismatch")
    seed_values = pool.get("seed_ids")
    if not isinstance(seed_values, list) or len(seed_values) != spec["seed_count"]:
        raise StageEvaluationError("held-out seed count does not match the preregistered stage")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or not 1 <= seed <= MAX_SEED
        for seed in seed_values
    ) or len(set(seed_values)) != len(seed_values):
        raise StageEvaluationError("held-out pool contains invalid or duplicate simulator seeds")
    seeds = tuple(seed_values)

    inventory = h3.shared._read_json(inventory_path)
    excluded, source_audit = validate_inventory(inventory)
    inventory_hash = sha256_json(inventory)
    if (
        pool.get("inventory_id") != inventory.get("inventory_id")
        or pool.get("inventory_sha256") != inventory_hash
        or pool.get("source_audit_sha256") != sha256_json(source_audit)
        or pool.get("source_audit") != source_audit
    ):
        raise StageEvaluationError("held-out pool does not match the verified exclusion inventory")

    ledger = h3.shared._read_json(pools_dir / "ledger.json")
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if sha256_json(ledger_payload) != ledger.get("ledger_sha256"):
        raise StageEvaluationError("round seed ledger hash mismatch")
    selected_entry = ledger.get("pools", {}).get(spec["pool_key"])
    if (
        not isinstance(selected_entry, dict)
        or selected_entry.get("manifest_sha256") != spec["manifest_sha256"]
        or selected_entry.get("seed_ids") != seed_values
    ):
        raise StageEvaluationError("round ledger and selected held-out pool differ")

    seen: set[int] = set()
    expected_files = {
        "train_hypothesis_1": "train_hypothesis_1.json",
        "train_hypothesis_2": "train_hypothesis_2.json",
        "train_hypothesis_3": "train_hypothesis_3.json",
        "probe": "probe.json",
        "dev": "dev.json",
    }
    ledger_pools = ledger.get("pools", {})
    for key, filename in expected_files.items():
        other = h3.shared._read_json(pools_dir / filename)
        other_payload = {name: value for name, value in other.items() if name != "manifest_sha256"}
        other_hash = other.get("manifest_sha256")
        other_seeds = other.get("seed_ids")
        ledger_entry = ledger_pools.get(key)
        if (
            sha256_json(other_payload) != other_hash
            or not isinstance(ledger_entry, dict)
            or ledger_entry.get("manifest_sha256") != other_hash
            or ledger_entry.get("seed_ids") != other_seeds
            or not isinstance(other_seeds, list)
        ):
            raise StageEvaluationError("a round pool differs from its frozen ledger entry")
        normalized = set(other_seeds)
        if len(normalized) != len(other_seeds) or normalized & seen or normalized & excluded:
            raise StageEvaluationError("round pools overlap each other or the protected inventory")
        seen.update(normalized)
    if set(seeds) & excluded:
        raise StageEvaluationError("held-out pool overlaps the protected exclusion inventory")

    return pool, seeds, {
        "heldout_seed_count": len(seeds),
        "exclusion_seed_count": len(excluded),
        "pool_manifest_count": len(expected_files),
        "inventory_sha256": inventory_hash,
        "disjoint_from_exclusion_inventory": True,
        "round_pools_pairwise_disjoint": True,
        "seed_contract": "heldout_internal",
    }


def _check_private_paths(stage: str, output_dir: Path, usage_ledger: Path) -> None:
    repo = REPO_ROOT.resolve()
    for path in (output_dir.resolve(), usage_ledger.resolve()):
        if path == repo or repo in path.parents:
            raise StageEvaluationError("raw held-out evidence must stay outside the repository")
    if output_dir.exists():
        raise StageEvaluationError("private output directory exists; never rerun this pool")
    spec = _stage_spec(stage)
    if usage_ledger.exists() and any(
        event.get("pool_id") == spec["pool_id"] for event in h3.shared._read_jsonl(usage_ledger)
    ):
        raise StageEvaluationError("held-out pool already has a private use record")


def _validate_probe_prerequisite(stage: str, summary_path: Path | None) -> dict[str, Any] | None:
    if stage == "probe":
        if summary_path is not None:
            raise StageEvaluationError("Probe10 must not depend on a prior probe summary")
        return None
    if summary_path is None or not summary_path.is_file():
        raise StageEvaluationError("Dev30 requires the completed private Probe10 summary")
    summary = h3.shared._read_json(summary_path)
    paired = summary.get("paired")
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("pool_id") != STAGES["probe"]["pool_id"]
        or summary.get("pool_manifest_sha256") != STAGES["probe"]["manifest_sha256"]
        or summary.get("identities", {}).get("candidate_git_commit_sha") != FROZEN_CANDIDATE_COMMIT
        or summary.get("trace_invariance") != "PASS"
        or not isinstance(paired, dict)
        or paired.get("net_wins", -1) < 0
        or summary.get("candidate_emergency_potion_overrides", 0) < 1
        or any(int(summary.get(key, -1)) != 0 for key in ("illegal_actions", "crashes", "timeouts"))
    ):
        raise StageEvaluationError("Probe10 failed the preregistered gate for Dev30")
    return {key: summary.get(key) for key in (
        "pool_id", "pool_manifest_sha256", "trace_invariance", "paired",
        "candidate_emergency_potion_overrides", "illegal_actions", "crashes", "timeouts",
    )}


def _run_one(
    *, sts: Any, policy: ArmGNoncombatPolicy, stage: str, seeds: tuple[int, ...], seed: int,
    output_dir: Path, pair_index: int, arm: str, candidate: bool,
    trace_enabled: bool, metadata: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any] | None]:
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
        **_seed_contract_kwargs(stage, seeds),
        collect_ppo=False,
        collect_teacher=False,
        diagnostic_trace_path=trace_path,
        diagnostic_metadata=metadata,
    )
    h3.shared._check_evidence(evidence_path, result, candidate=False)
    trace_stats = None
    if trace_path is not None:
        trace_stats = h3._check_trace(
            trace_path,
            candidate=candidate,
            expected_overrides=int(result.get("emergency_potion_override_count", 0)),
        )
    return result, evidence_path, trace_stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--pools-dir", type=Path, required=True)
    parser.add_argument("--exclusion-inventory", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--g7-checkpoint", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--private-usage-ledger", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        spec = _stage_spec(args.stage)
        pool, seeds, preflight = _validate_pool(
            args.stage, args.pool_file.resolve(), args.pools_dir.resolve(), args.exclusion_inventory.resolve()
        )
        _check_private_paths(args.stage, args.private_output_dir, args.private_usage_ledger)
        probe_prerequisite = _validate_probe_prerequisite(args.stage, args.probe_summary)
        checkpoint_sha = _sha256(args.g7_checkpoint)
        if checkpoint_sha != G7_SHA256 or Path(str(args.g7_checkpoint) + ".adapter.pt").exists():
            raise StageEvaluationError("evaluation must use the unmodified G7 parent checkpoint")
        binding = args.module_dir / "slaythespire.cp312-win_amd64.pyd"
        if _sha256(binding) != SIMULATOR_BINDING_SHA256:
            raise StageEvaluationError("native simulator binding hash mismatch")
        armg_source = args.armg_root / "armG_train.py"
        if not armg_source.is_file() or _sha256(armg_source) != ARMG_SOURCE_SHA256:
            raise StageEvaluationError("pinned ArmG source is missing or changed")
        if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
            raise StageEvaluationError("contextual reranking must be disabled for fixed comparisons")

        simulator_source = Path(simulator_module.__file__).resolve()
        current_policy_sha = _sha256(simulator_source)
        frozen_policy = subprocess.check_output(
            ["git", "show", f"{FROZEN_CANDIDATE_COMMIT}:src/roguelike_ai/sts1_phase3/simulator.py"],
            cwd=REPO_ROOT,
        )
        if (
            current_policy_sha != FROZEN_SIMULATOR_SOURCE_SHA256
            or hashlib.sha256(frozen_policy).hexdigest() != FROZEN_SIMULATOR_SOURCE_SHA256
        ):
            raise StageEvaluationError("current or frozen H3 policy source differs from the registered commit")

        sts = _load_sts(args.module_dir)
        parent_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.g7_checkpoint)
        candidate_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.g7_checkpoint)
        metadata = {
            "stage": args.stage,
            "pool_id": spec["pool_id"],
            "pool_manifest_sha256": spec["manifest_sha256"],
            "candidate_git_commit_sha": FROZEN_CANDIDATE_COMMIT,
            "simulator_commit": SIMULATOR_COMMIT,
            "simulator_python_source_sha256": current_policy_sha,
            "runner_source_sha256": _sha256(Path(__file__).resolve()),
            "simulator_binding_sha256": SIMULATOR_BINDING_SHA256,
            "armg_source_sha256": ARMG_SOURCE_SHA256,
            "g7_checkpoint_sha256": checkpoint_sha,
            "mcts_sims": MCTS_SIMS,
            "seed_contract": "heldout_internal",
            "candidate_intervention": "at HP/maxHP <= 0.5, use the lowest-slot exact legal potion; otherwise preserve the parent MCTS action",
        }
        if args.preflight_only:
            print(json.dumps({"preflight": "PASS", **preflight, **metadata, "probe_prerequisite": probe_prerequisite}, sort_keys=True))
            return 0

        args.private_output_dir.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        h3.shared._append_jsonl(args.private_usage_ledger, {
            "pool_id": spec["pool_id"], "manifest_sha256": spec["manifest_sha256"],
            "stage": args.stage, "status": "RUNNING", "started_at_utc": started,
            "expected_paired_episodes": 2 * len(seeds), "trace_invariance_episodes": 1,
            "seed_contract": "heldout_internal", "candidate_policy_commit": FROZEN_CANDIDATE_COMMIT,
        })
        completed = 0
        parent_outcomes: list[str] = []
        candidate_outcomes: list[str] = []
        rows: list[dict[str, Any]] = []
        try:
            for pair_index, seed in enumerate(seeds):
                print(f"H3 {args.stage} pair {pair_index + 1}/{len(seeds)}: parent")
                parent, _, parent_trace = _run_one(
                    sts=sts, policy=parent_policy, stage=args.stage, seeds=seeds, seed=seed,
                    output_dir=args.private_output_dir, pair_index=pair_index, arm="parent",
                    candidate=False, trace_enabled=True, metadata=metadata,
                )
                completed += 1
                print(f"H3 {args.stage} pair {pair_index + 1}/{len(seeds)}: candidate")
                candidate, candidate_evidence, candidate_trace = _run_one(
                    sts=sts, policy=candidate_policy, stage=args.stage, seeds=seeds, seed=seed,
                    output_dir=args.private_output_dir, pair_index=pair_index, arm="candidate",
                    candidate=True, trace_enabled=True, metadata=metadata,
                )
                completed += 1
                trace_invariance = None
                if pair_index == 0:
                    trace_off, trace_off_evidence, _ = _run_one(
                        sts=sts, policy=candidate_policy, stage=args.stage, seeds=seeds, seed=seed,
                        output_dir=args.private_output_dir, pair_index=pair_index, arm="candidate",
                        candidate=True, trace_enabled=False, metadata=metadata,
                    )
                    completed += 1
                    if (
                        trace_off.get("outcome") != candidate.get("outcome")
                        or h3._summary_view(trace_off) != h3._summary_view(candidate)
                        or h3._action_signature(trace_off_evidence) != h3._action_signature(candidate_evidence)
                    ):
                        raise StageEvaluationError("trace on/off changed the fixed-seed H3 trajectory")
                    trace_invariance = "PASS"
                parent_outcomes.append(str(parent["outcome"]))
                candidate_outcomes.append(str(candidate["outcome"]))
                rows.append({
                    "parent": h3._summary_view(parent), "candidate": h3._summary_view(candidate),
                    "parent_trace": parent_trace, "candidate_trace": candidate_trace,
                    "trace_invariance": trace_invariance,
                })
                h3.shared._append_jsonl(args.private_output_dir / "paired-runs.ndjson", rows[-1])

            paired = _paired_summary(parent_outcomes, candidate_outcomes)
            overrides = sum(int(row["candidate"].get("emergency_potion_override_count") or 0) for row in rows)
            opportunities = sum(int(row["candidate_trace"]["low_hp_potion_opportunities"]) for row in rows)
            safety = {
                key: sum(int(row[arm].get(key) or 0) for row in rows for arm in ("parent", "candidate"))
                for key in ("illegal_action_count", "crash_count", "timeout_count")
            }
            if any(safety.values()):
                raise StageEvaluationError("held-out stage failed a safety guard")
            final = {
                "schema_version": "sts1-g7-h3-heldout-stage-v1", "status": "COMPLETE",
                "stage": args.stage, "pool_id": spec["pool_id"],
                "pool_manifest_sha256": spec["manifest_sha256"],
                "paired_seed_count": len(seeds), "episode_count": completed,
                "trace_invariance": "PASS", "preflight": preflight,
                "probe_prerequisite": probe_prerequisite, "identities": metadata,
                "paired": paired, "candidate_combat_decisions": sum(
                    int(row["candidate_trace"]["combat_decisions"]) for row in rows
                ),
                "candidate_low_hp_potion_opportunities": opportunities,
                "candidate_emergency_potion_overrides": overrides,
                "illegal_actions": safety["illegal_action_count"],
                "crashes": safety["crash_count"], "timeouts": safety["timeout_count"],
                "communication_errors": "N/A_LOCAL_SIMULATOR",
            }
            (args.private_output_dir / "paired-summary.json").write_text(
                json.dumps(final, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            h3.shared._append_jsonl(args.private_usage_ledger, {
                "pool_id": spec["pool_id"], "manifest_sha256": spec["manifest_sha256"],
                "stage": args.stage, "status": "COMPLETE", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "episode_count": completed, "trace_invariance": "PASS",
                "seed_contract": "heldout_internal", "candidate_policy_commit": FROZEN_CANDIDATE_COMMIT,
            })
            print(json.dumps({
                "status": final["status"], "stage": args.stage, "paired_seed_count": len(seeds),
                "episode_count": completed, **paired, "candidate_low_hp_potion_opportunities": opportunities,
                "candidate_emergency_potion_overrides": overrides, "illegal_actions": safety["illegal_action_count"],
                "crashes": safety["crash_count"], "timeouts": safety["timeout_count"],
                "communication_errors": final["communication_errors"],
            }, sort_keys=True))
            return 0
        except BaseException as exc:
            h3.shared._append_jsonl(args.private_usage_ledger, {
                "pool_id": spec["pool_id"], "manifest_sha256": spec["manifest_sha256"],
                "stage": args.stage, "status": "NOT_VERIFIED", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "episode_count": completed, "error_type": type(exc).__name__,
                "seed_contract": "heldout_internal", "candidate_policy_commit": FROZEN_CANDIDATE_COMMIT,
            })
            print(json.dumps({"status": "NOT_VERIFIED", "stage": args.stage, "episode_count": completed, "error_type": type(exc).__name__}, sort_keys=True))
            return 2
    except (StageEvaluationError, h3.EvaluationIntegrityError, OSError, ValueError) as exc:
        print(json.dumps({"status": "PREFLIGHT_FAILED", "error_type": type(exc).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
