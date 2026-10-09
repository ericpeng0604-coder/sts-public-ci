"""Run the preregistered Round008 H16 paired lethal-intent Defend evaluation.

Raw seed IDs, episodes, traces, and usage records must remain in the private
Temp evaluation directory. This runner never tunes the registered rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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

EvaluationIntegrityError = h2.EvaluationIntegrityError
ROUND_ID = "round-008-20261009"
POOL_ROLE = {"train": "train_hypothesis_2", "probe": "probe", "dev": "dev"}
POOL_COUNTS = {"train": 10, "probe": 10, "dev": 30}
POOL_FILES = {"train": "train_hypothesis_2.json", "probe": "probe.json", "dev": "dev.json"}
EXPECTED_H16_POOL_SHA256 = "a71b491d582dfd358c9fc48c0a3f3e9f2a907e3623a447aff598f19417eb2190"
G7_SHA256 = "8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0"
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


def _round008_assets(
    *, stage: str, pool_file: Path, pools_dir: Path, inventory_path: Path
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    role = POOL_ROLE[stage]
    expected_pool = (pools_dir / POOL_FILES[stage]).resolve()
    if pool_file.resolve() != expected_pool:
        raise EvaluationIntegrityError("selected H16 pool path does not match the registered stage")

    # H15's audited loader validates the full Round008 ledger, all five pool
    # manifests, the ID-only exclusion inventory, and pairwise disjointness.
    # It requires the H1 path as its anchor; H16 selects only H2 below and
    # never executes H1.
    _, validated_pools, round_preflight = h15._validate_round008_assets(
        pools_dir / "train_hypothesis_1.json", pools_dir, inventory_path
    )
    pool = h15._read_json(expected_pool)
    expected_pool_sha = (
        EXPECTED_H16_POOL_SHA256 if stage == "train" else pool.get("manifest_sha256")
    )
    if pool.get("manifest_sha256") != expected_pool_sha:
        raise EvaluationIntegrityError("selected H16 pool manifest hash is not registered")
    seeds = validated_pools.get(role)
    if not isinstance(seeds, tuple) or len(seeds) != POOL_COUNTS[stage]:
        raise EvaluationIntegrityError("selected H16 pool count or validation is invalid")
    preflight = {
        **round_preflight,
        "pool_id": pool.get("pool_id"),
        "pool_manifest_sha256": pool.get("manifest_sha256"),
        "stage": stage,
        "seed_count": len(seeds),
    }
    return pool, seeds, preflight


def _validate_private_paths(pools_dir: Path, stage: str, output_dir: Path, usage_path: Path) -> None:
    pools_dir = pools_dir.resolve()
    expected_output = pools_dir.parent.parent / f"round-008-h16-{stage}-20261009"
    expected_usage = pools_dir.parent / "h16-usage-private.jsonl"
    if output_dir.resolve() != expected_output.resolve() or usage_path.resolve() != expected_usage.resolve():
        raise EvaluationIntegrityError("H16 requires its canonical private output and usage paths")
    repo = REPO_ROOT.resolve()
    for path in (output_dir.resolve(), usage_path.resolve()):
        if path == repo or repo in path.parents:
            raise EvaluationIntegrityError("raw H16 records must remain outside the repository")
    if output_dir.exists():
        raise EvaluationIntegrityError("H16 stage output already exists; refusing to reuse it")


def _validate_identity(module_dir: Path, armg_root: Path, checkpoint: Path) -> dict[str, str]:
    simulator_source = REPO_ROOT / "src" / "roguelike_ai" / "sts1_phase3" / "simulator.py"
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


def _stage_summary(events: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    rows = [
        event
        for event in events
        if event.get("record_type") == "h16_stage_summary" and event.get("stage") == stage
    ]
    return rows[-1] if rows else None


def _validate_transition(
    *, stage: str, events: list[dict[str, Any]], candidate_commit: str, identities: dict[str, str]
) -> None:
    if any(event.get("record_type") == "h16_stage_start" and event.get("stage") == stage for event in events):
        raise EvaluationIntegrityError("H16 stage has already been attempted; its pool cannot be rerun")
    if stage == "train":
        if events:
            raise EvaluationIntegrityError("H16 train stage requires a new private usage ledger")
        return

    predecessor = "train" if stage == "probe" else "probe"
    prior = _stage_summary(events, predecessor)
    if not isinstance(prior, dict) or prior.get("status") != "COMPLETE":
        raise EvaluationIntegrityError("required preceding H16 stage is not complete")
    if prior.get("candidate_commit") != candidate_commit:
        raise EvaluationIntegrityError("H16 candidate commit changed between stages")
    for key in (
        "simulator_policy_source_sha256",
        "candidate_evaluator_sha256",
        "simulator_binding_sha256",
        "armg_source_sha256",
        "armg_vocab_sha256",
        "g7_checkpoint_sha256",
    ):
        if prior.get(key) != identities.get(key):
            raise EvaluationIntegrityError("H16 policy, evaluator, simulator, or parent changed between stages")
    if prior.get("advance_eligible") is not True:
        raise EvaluationIntegrityError("preceding H16 paired result did not pass its preregistered gate")


def _check_pair_integrity(path: Path, result: dict[str, Any]) -> None:
    h2._check_evidence(path, result, candidate=False)
    if result.get("outcome") not in {"victory", "defeat"} or result.get("error") is not None:
        raise EvaluationIntegrityError("H16 episode has no complete terminal outcome")
    for key in ("illegal_action_count", "timeout_count", "crash_count"):
        value = result.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise EvaluationIntegrityError("H16 safety counter was nonzero or unavailable")
    communication_errors = result.get("communication_error_count", 0)
    if communication_errors not in (0, None):
        raise EvaluationIntegrityError("H16 local simulator communication error counter was nonzero")
    if result.get("potion_inventory_snapshot_complete") is not True:
        raise EvaluationIntegrityError("H16 complete potion inventory was not recorded")


def _empty_coverage() -> dict[str, int]:
    return {key: 0 for key in TRACE_COVERAGE_FIELDS}


def _accumulate_coverage(total: dict[str, int], stats: dict[str, int]) -> None:
    if set(stats) != set(TRACE_COVERAGE_FIELDS):
        raise EvaluationIntegrityError("H16 trace coverage fields are incomplete")
    for key, value in stats.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvaluationIntegrityError("H16 trace coverage counter is invalid")
        total[key] += value


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
) -> tuple[dict[str, Any], Path, dict[str, int], int, dict[str, int]]:
    stem = f"pair-{pair_index:02d}-{arm}"
    evidence_path = output_dir / f"{stem}.evidence.ndjson"
    trace_path = output_dir / f"{stem}.trace.ndjson"
    if evidence_path.exists() or trace_path.exists():
        raise EvaluationIntegrityError("H16 episode output already exists; refusing to reuse a seed")
    metadata = {
        **identity,
        "stage": identity["stage"],
        "pair_index": pair_index,
        "episode_index": (pair_index - 1) * 2 + (2 if candidate else 1),
        "arm": arm,
        "trace_mode": "on",
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
        training_seeds=stage_seeds,
        collect_ppo=False,
        collect_teacher=False,
        diagnostic_trace_path=trace_path,
        diagnostic_metadata=metadata,
    )
    _check_pair_integrity(evidence_path, result)
    if result.get("lethal_intent_defend_rescue_enabled") is not candidate:
        raise EvaluationIntegrityError("H16 policy activation flag differs from the requested arm")
    trace_stats = h7._validate_trace(trace_path, result, metadata)
    events = list(h7._read_jsonl(trace_path))
    combat_events = [event for event in events if event.get("type") == "combat_decision_trace_v1"]
    override_events = [event for event in combat_events if event.get("lethal_intent_defend_override") is True]
    if len(override_events) != result.get("lethal_intent_defend_override_count"):
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
    for artifact in (evidence_path, trace_path):
        if artifact.stat().st_size > MAX_EPISODE_BYTES:
            raise EvaluationIntegrityError("H16 private episode artifact exceeded its bound")
    return result, evidence_path, trace_stats, len(override_events), dict(reasons)


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
    parser.add_argument("--stage", choices=("train", "probe", "dev"), default="train")
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
    stage = args.stage
    output_dir = args.private_output_dir.resolve()
    usage_path = args.private_usage_ledger.resolve()
    try:
        pool, seeds, pool_preflight = _round008_assets(
            stage=stage,
            pool_file=args.pool_file.resolve(),
            pools_dir=args.pools_dir.resolve(),
            inventory_path=args.exclusion_inventory.resolve(),
        )
        _validate_private_paths(args.pools_dir.resolve(), stage, output_dir, usage_path)
        if len(args.candidate_commit) != 40 or args.candidate_commit != _git_head():
            raise EvaluationIntegrityError("candidate commit does not match the frozen local HEAD")
        identities = _validate_identity(
            args.module_dir.resolve(), args.armg_root.resolve(), args.g7_checkpoint.resolve()
        )
        events = _read_jsonl(usage_path)
        _validate_transition(
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
        "round_id": ROUND_ID,
        "hypothesis": "h16",
        "stage": stage,
        "pool_id": pool["pool_id"],
        "pool_manifest_sha256": pool["manifest_sha256"],
        "candidate_commit": args.candidate_commit,
        "mcts_sims": MCTS_SIMS,
        "g7_checkpoint_sha256": G7_SHA256,
        **identities,
        **pool_preflight,
    }
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", **identity}, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    expected_episodes = 2 * len(seeds)
    _append_jsonl(usage_path, {
        "record_type": "h16_stage_start",
        **identity,
        "status": "RUNNING",
        "started_at_utc": started_at,
        "expected_episodes": expected_episodes,
        "seed_ids": list(seeds),
        "private_output_dir": str(output_dir),
    })

    parent_outcomes: list[str] = []
    candidate_outcomes: list[str] = []
    parent_trace_totals = _empty_coverage()
    candidate_trace_totals = _empty_coverage()
    candidate_overrides = 0
    candidate_reasons: Counter[str] = Counter()
    floor_relations: Counter[str] = Counter()
    hp_relations: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    episodes_attempted = 0
    try:
        for index, seed in enumerate(seeds):
            pair_index = index + 1
            print(f"H16 {stage} pair {pair_index}/{len(seeds)}: parent")
            parent, parent_path, parent_stats, parent_overrides, parent_reasons = _run_episode(
                sts=sts,
                policy=policy,
                seed=seed,
                stage_seeds=seeds,
                output_dir=output_dir,
                pair_index=pair_index,
                arm="parent",
                candidate=False,
                identity=identity,
            )
            episodes_attempted += 1
            _accumulate_coverage(parent_trace_totals, parent_stats)
            if parent_overrides != 0 or parent.get("lethal_intent_defend_override_count") != 0:
                raise EvaluationIntegrityError("G7 parent arm had a registered H16 action override")

            print(f"H16 {stage} pair {pair_index}/{len(seeds)}: candidate")
            candidate, candidate_path, candidate_stats, overrides, reasons = _run_episode(
                sts=sts,
                policy=policy,
                seed=seed,
                stage_seeds=seeds,
                output_dir=output_dir,
                pair_index=pair_index,
                arm="candidate",
                candidate=True,
                identity=identity,
            )
            episodes_attempted += 1
            _accumulate_coverage(candidate_trace_totals, candidate_stats)
            candidate_overrides += overrides
            candidate_reasons.update(reasons)
            parent_outcomes.append(str(parent["outcome"]))
            candidate_outcomes.append(str(candidate["outcome"]))

            parent_floor = parent.get("final_floor")
            candidate_floor = candidate.get("final_floor")
            parent_hp = parent.get("final_hp")
            candidate_hp = candidate.get("final_hp")
            values = (parent_floor, candidate_floor, parent_hp, candidate_hp)
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                raise EvaluationIntegrityError("paired final floor or HP is unavailable")
            floor_relations["lower" if candidate_floor < parent_floor else "higher" if candidate_floor > parent_floor else "equal"] += 1
            hp_relations["lower" if candidate_hp < parent_hp else "higher" if candidate_hp > parent_hp else "equal"] += 1
            _append_jsonl(output_dir / "paired-results.private.jsonl", {
                "pair_index": pair_index,
                "seed_id": seed,
                "parent_outcome": parent["outcome"],
                "candidate_outcome": candidate["outcome"],
                "parent_summary": h3._summary_view(parent),
                "candidate_summary": h3._summary_view(candidate),
                "candidate_override_count": overrides,
                "candidate_override_reasons": reasons,
                "parent_evidence": str(parent_path),
                "candidate_evidence": str(candidate_path),
            })
            _append_jsonl(usage_path, {
                "record_type": "h16_stage_progress",
                "stage": stage,
                "candidate_commit": args.candidate_commit,
                "completed_pairs": pair_index,
                "episodes_attempted": episodes_attempted,
                "candidate_overrides": candidate_overrides,
            })
            rows.append({
                "pair_index": pair_index,
                "seed_id": seed,
                "parent": h3._summary_view(parent),
                "candidate": h3._summary_view(candidate),
                "candidate_override_count": overrides,
            })

        paired = h3._paired_summary_for_stage(parent_outcomes, candidate_outcomes)
        retention = {
            "terminal_floor_pairs": dict(sorted(floor_relations.items())),
            "final_hp_pairs": dict(sorted(hp_relations.items())),
            "lower_floor_pairs": floor_relations["lower"],
            "lower_final_hp_pairs": hp_relations["lower"],
            "passed": floor_relations["lower"] == 0 and hp_relations["lower"] == 0,
        }
        advance_eligible = bool(
            candidate_overrides > 0 and paired.get("net_wins", -1) >= 0 and retention["passed"]
        )
        safety = {
            "illegal_actions": 0,
            "crashes": 0,
            "timeouts": 0,
            "communication_errors": "N/A_LOCAL_SIMULATOR",
            "complete_terminal_records": episodes_attempted,
            "legal_trace_records": episodes_attempted,
            "trace_mode": "on_for_both_arms",
        }
        artifact_entries, artifact_bytes = _artifact_manifest(output_dir)
        summary = {
            "record_type": "h16_stage_summary",
            **identity,
            "status": "COMPLETE",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed_count": len(seeds),
            "episodes": episodes_attempted,
            "parent_outcomes": {"victory": parent_outcomes.count("victory"), "defeat": parent_outcomes.count("defeat")},
            "candidate_outcomes": {"victory": candidate_outcomes.count("victory"), "defeat": candidate_outcomes.count("defeat")},
            "paired": paired,
            "candidate_override_count": candidate_overrides,
            "candidate_override_reason_counts": dict(sorted(candidate_reasons.items())),
            "parent_trace_coverage": parent_trace_totals,
            "candidate_trace_coverage": candidate_trace_totals,
            "retention_guard": retention,
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
            "candidate_override_reason_counts": dict(sorted(candidate_reasons.items())),
            "retention_guard": retention,
            "advance_eligible": advance_eligible,
            "safety": safety,
            "candidate_commit": args.candidate_commit,
            "simulator_policy_source_sha256": identities["simulator_policy_source_sha256"],
            "candidate_evaluator_sha256": identities["candidate_evaluator_sha256"],
            "pool_manifest_sha256": pool["manifest_sha256"],
            "private_summary_sha256": summary["private_summary_sha256"],
            "artifact_bytes": artifact_bytes,
        }, sort_keys=True))
        return 0
    except Exception as exc:
        _append_jsonl(usage_path, {
            "record_type": "h16_stage_failure",
            **identity,
            "status": "NOT_VERIFIED",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_attempted": episodes_attempted,
            "completed_pairs": len(parent_outcomes),
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
