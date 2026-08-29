#!/usr/bin/env python3
"""Evaluate the frozen midturn V2 candidate once on a fresh disjoint holdout."""
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
from roguelike_ai.sts1_teacher.midturn_v2 import FROZEN_MARGIN, POLICY_ID, choose_midturn_v2
from roguelike_ai.sts1_teacher.native_rollout import NativePublicJawWormRolloutV1
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator, public_sample
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig

SCHEMA = "sts1-phase1-public-search-v2-jaw-worm-turn-one-midturn-holdout-quality-v1"
SUITE_SCHEMA = "sts1-phase1-public-search-v2-jaw-worm-turn-one-midturn-holdout-v1"
SEARCH_SAMPLE_COUNT = 8
ORACLE_SAMPLE_COUNT = 4
ORACLE_MCTS_SIMS = 2000
SEARCH_SAMPLING_SEED = 20260830


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("holdout_json_not_object")
    return value


def rows_digest(rows: list[dict]) -> str:
    payload = "\n".join(canonical_json(row) for row in rows) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def oracle_scores(context, backend, config) -> dict[str, float]:
    groups: dict[str, list] = {}
    for action in context.legal_actions:
        groups.setdefault(action.semantic_key, []).append(action)
    totals = {action.action_id: 0.0 for action in context.legal_actions}
    for sample_index in range(ORACLE_SAMPLE_COUNT):
        sample = public_sample(context, sample_index=sample_index, config=config)
        for semantic_key in sorted(groups):
            members = sorted(groups[semantic_key], key=lambda x: x.action_id)
            representative = members[0]
            bc = backend._build(context, sample)
            native_action = backend._native_action(bc, representative)
            score = float(sts.judge_branch_action(bc, native_action, ORACLE_MCTS_SIMS)["score"])
            for member in members:
                totals[member.action_id] += score
    return {k: v / ORACLE_SAMPLE_COUNT for k, v in totals.items()}


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
    counts = {"v2": 0, "search": 0, "simple": 0, "random": 0}
    fallback_count = 0
    unresolved = 0
    rows: list[dict] = []

    for record in suite["cases"]:
        context = require_public_reconstruction(record["public_state"])
        if context.decision_signature != record["decision_signature"]:
            raise RuntimeError(f"holdout_signature_drift:{record['case_id']}")
        backend = NativePublicJawWormRolloutV1(record["public_state"], sts)
        search_result = PublicStateSearch(PairedPublicSampleEvaluator(backend), config).run(context)
        if not search_result.resolved or search_result.timed_out or search_result.unresolved_action_ids:
            unresolved += 1
        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError(f"holdout_baseline_missing:{record['case_id']}")
        v2 = choose_midturn_v2(context, search_result)
        fallback_count += int(v2.used_simple_fallback)
        scores = oracle_scores(context, backend, config)
        ties = oracle_ties(scores)
        agreement = {
            "v2": single_action_agreement(v2.action_id, ties),
            "search": conservative_tie_agreement(tuple(sorted(search_result.tie_action_ids)), ties),
            "simple": single_action_agreement(simple.action_id, ties),
            "random": single_action_agreement(random_action.action_id, ties),
        }
        for key in counts:
            counts[key] += int(agreement[key])
        rows.append({
            "case_id": record["case_id"],
            "source_seed_provenance_only": record["source_seed_provenance_only"],
            "decision_signature": context.decision_signature,
            "public_intent": str(context.state["enemies"][0].get("intent", "")),
            "public_block": int(context.state.get("block", 0)),
            "v2_policy_id": POLICY_ID,
            "v2_margin": FROZEN_MARGIN,
            "v2_selected_action_id": v2.action_id,
            "v2_used_simple_fallback": v2.used_simple_fallback,
            "v2_observed_semantic_margin": v2.semantic_margin,
            "raw_search_tie_ids": list(sorted(search_result.tie_action_ids)),
            "simple_action_id": simple.action_id,
            "random_action_id": random_action.action_id,
            "oracle_tie_ids": list(ties),
            "agreement": agreement,
        })

    reasons: list[str] = []
    if unresolved:
        reasons.append(f"search_unresolved:{unresolved}")
    for baseline in ("search", "simple", "random"):
        if not counts["v2"] > counts[baseline]:
            reasons.append(f"v2_not_strictly_better_than_{baseline}")
    summary = {
        "schema_version": SCHEMA,
        "scope": "fresh 440xxx Jaw Worm turn-1 one-card midturn holdout",
        "phase1_gate_claimed": False,
        "policy_id": POLICY_ID,
        "frozen_margin": FROZEN_MARGIN,
        "case_count": len(rows),
        "agreement_counts": counts,
        "agreement_rates": {key: counts[key] / len(rows) for key in counts},
        "fallback_count": fallback_count,
        "search_unresolved": unresolved,
        "quality_verdict": "PASS" if not reasons else "FAIL",
        "quality_reasons": reasons,
        "rows_digest": rows_digest(rows),
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
    if suite.get("schema_version") != SUITE_SCHEMA:
        raise RuntimeError("unexpected_holdout_suite_schema")
    if suite.get("case_count") != 24 or suite.get("frozen_before_holdout_results") is not True:
        raise RuntimeError("fresh_holdout_freeze_contract_failed")
    if suite.get("preselected_policy_id") != POLICY_ID or float(suite.get("preselected_margin")) != FROZEN_MARGIN:
        raise RuntimeError("holdout_policy_was_not_preselected")
    if suite.get("quality_results_observed_during_selection") != 0:
        raise RuntimeError("holdout_selection_observed_quality")

    rows1, summary1 = run_once(suite)
    rows2, summary2 = run_once(suite)
    if rows1 != rows2 or summary1 != summary2:
        raise RuntimeError("midturn_v2_fresh_holdout_not_deterministic")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    (out / "decisions.ndjson").write_text("".join(canonical_json(row) + "\n" for row in rows1))
    final = dict(summary1)
    final.update({
        "full_repeats": 2,
        "deterministic": True,
        "suite_digest": suite["suite_digest"],
        "source_seed_file_sha256": suite["source_seed_file_sha256"],
        "retired_420xxx_rows_read": 0,
        "discovery_430xxx_rows_read_during_holdout_evaluation": 0,
    })
    (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True))
    print(f"MIDTURN_V2_FRESH_HOLDOUT = {final['quality_verdict']}")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
