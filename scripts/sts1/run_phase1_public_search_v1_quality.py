#!/usr/bin/env python3
"""Run the first frozen multi-state quality benchmark for native PublicStateSearch V1.

This is a narrow-slice quality experiment, not the full Phase-1 gate. The
formal Search receives public state only. The diagnostic oracle may inspect
private state inside independently reconstructed sample contexts, but oracle
scores/ties are never fed back into Search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
BUILD = ROOT / "external" / "sts_lightspeed" / "source" / "build-probe"
for path in (SRC, BUILD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import slaythespire as sts

from roguelike_ai.sts1_teacher.baselines import deterministic_random_legal, simple_public_heuristic
from roguelike_ai.sts1_teacher.benchmark import conservative_tie_agreement, oracle_ties, single_action_agreement
from roguelike_ai.sts1_teacher.contract import canonical_json
from roguelike_ai.sts1_teacher.native_rollout import NativePublicJawWormRolloutV1
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator, public_sample
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig

SCHEMA = "sts1-phase1-public-search-v1-quality-v1"
ORACLE_MCTS_SIMS = 2000
ORACLE_SAMPLE_COUNT = 4
SEARCH_SAMPLE_COUNT = 8
SEARCH_SAMPLING_SEED = 20260830


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected_json_object:{path}")
    return value


def digest_rows(rows: list[dict]) -> str:
    payload = "\n".join(canonical_json(row) for row in rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def oracle_scores_for_case(context, state, backend, config) -> dict[str, float]:
    semantic_groups: dict[str, list] = {}
    for action in context.legal_actions:
        semantic_groups.setdefault(action.semantic_key, []).append(action)

    totals = {action.action_id: 0.0 for action in context.legal_actions}
    for sample_index in range(ORACLE_SAMPLE_COUNT):
        sample = public_sample(context, sample_index=sample_index, config=config)
        for semantic_key in sorted(semantic_groups):
            members = sorted(semantic_groups[semantic_key], key=lambda item: item.action_id)
            representative = members[0]
            # Fresh reconstructed context per judged representative prevents
            # cross-action mutation while preserving the same paired sample.
            bc = backend._build(context, sample)
            native_action = backend._native_action(bc, representative)
            judged = sts.judge_branch_action(bc, native_action, ORACLE_MCTS_SIMS)
            score = float(judged["score"])
            for member in members:
                totals[member.action_id] += score

    return {
        action_id: total / ORACLE_SAMPLE_COUNT
        for action_id, total in totals.items()
    }


def run_once(suite: dict) -> tuple[list[dict], dict]:
    config = SearchConfig(
        samples_per_semantic_action=SEARCH_SAMPLE_COUNT,
        rollout_budget=256,
        node_budget=16384,
        max_depth=30,
        timeout_ms=60_000,
        tie_tolerance=1e-9,
        sampling_seed=SEARCH_SAMPLING_SEED,
    )
    rows: list[dict] = []
    counts = {"search": 0, "simple": 0, "random": 0}
    unresolved = 0

    for record in suite["cases"]:
        state = record["public_state"]
        context = require_public_reconstruction(state)
        if context.decision_signature != record["decision_signature"]:
            raise RuntimeError(f"suite_signature_drift:{record['case_id']}")

        backend = NativePublicJawWormRolloutV1(state, sts)
        evaluator = PairedPublicSampleEvaluator(backend)
        search_result = PublicStateSearch(evaluator, config).run(context)
        if not search_result.resolved or search_result.timed_out or search_result.unresolved_action_ids:
            unresolved += 1

        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError(f"baseline_missing_action:{record['case_id']}")

        oracle_scores = oracle_scores_for_case(context, state, backend, config)
        oracle_tie_ids = oracle_ties(oracle_scores)
        search_ties = tuple(sorted(search_result.tie_action_ids))

        agreement = {
            "search": conservative_tie_agreement(search_ties, oracle_tie_ids),
            "simple": single_action_agreement(simple.action_id, oracle_tie_ids),
            "random": single_action_agreement(random_action.action_id, oracle_tie_ids),
        }
        for key in counts:
            counts[key] += int(agreement[key])

        rows.append(
            {
                "schema_version": SCHEMA,
                "case_id": record["case_id"],
                "source_seed_provenance_only": record["source_seed_provenance_only"],
                "decision_signature": context.decision_signature,
                "legal_action_ids": sorted(action.action_id for action in context.legal_actions),
                "search_tie_ids": list(search_ties),
                "search_unique_best_action_id": search_result.unique_best_action_id,
                "search_candidate_scores": dict(sorted(search_result.candidate_scores.items())),
                "search_evidence_hash": search_result.evidence_hash,
                "search_rollout_count": search_result.rollout_count,
                "simple_action_id": simple.action_id,
                "random_action_id": random_action.action_id,
                "oracle_scores": dict(sorted(oracle_scores.items())),
                "oracle_tie_ids": list(oracle_tie_ids),
                "agreement": agreement,
            }
        )

    total = len(rows)
    rates = {name: counts[name] / total for name in counts}
    quality_reasons: list[str] = []
    if unresolved:
        quality_reasons.append(f"search_unresolved:{unresolved}")
    if not rates["search"] > rates["random"]:
        quality_reasons.append("search_not_strictly_better_than_random")
    if not rates["search"] > rates["simple"]:
        quality_reasons.append("search_not_strictly_better_than_simple")

    summary = {
        "schema_version": SCHEMA,
        "scope": "opening-turn Jaw Worm V1 quality slice only",
        "phase1_gate_claimed": False,
        "case_count": total,
        "search_samples_per_semantic_action": SEARCH_SAMPLE_COUNT,
        "oracle_samples": ORACLE_SAMPLE_COUNT,
        "oracle_mcts_sims_per_action_per_sample": ORACLE_MCTS_SIMS,
        "agreement_counts": counts,
        "agreement_rates": rates,
        "search_unresolved": unresolved,
        "quality_verdict": "PASS" if not quality_reasons else "FAIL",
        "quality_reasons": quality_reasons,
        "rows_digest": digest_rows(rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not hasattr(sts, "judge_branch_action"):
        raise RuntimeError("diagnostic_oracle_binding_missing")

    suite = read_json(args.suite)
    if suite.get("case_count") != 24 or suite.get("frozen_before_quality_results") is not True:
        raise RuntimeError("suite_freeze_contract_failed")

    rows1, summary1 = run_once(suite)
    rows2, summary2 = run_once(suite)
    deterministic = rows1 == rows2 and summary1 == summary2
    if not deterministic:
        raise RuntimeError("public_search_v1_quality_not_deterministic")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "decisions.ndjson").write_text("".join(canonical_json(row) + "\n" for row in rows1), encoding="utf-8")
    final_summary = dict(summary1)
    final_summary.update(
        {
            "full_repeats": 2,
            "deterministic": True,
            "suite_digest": suite["suite_digest"],
        }
    )
    (out / "summary.json").write_text(json.dumps(final_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(final_summary, indent=2, sort_keys=True))
    print("PUBLIC_SEARCH_V1_QUALITY_EXECUTION = PASS")
    print(f"QUALITY_VERDICT = {final_summary['quality_verdict']}")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
