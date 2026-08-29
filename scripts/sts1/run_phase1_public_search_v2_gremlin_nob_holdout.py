#!/usr/bin/env python3
"""Evaluate the frozen Gremlin Nob Search V2 candidate on a fresh 360xxx holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from roguelike_ai.sts1_teacher.native_rollout import NativePublicGremlinNobRolloutV1
from roguelike_ai.sts1_teacher.nob_search_v2 import (
    GREMLIN_NOB_V2_MARGIN_THRESHOLD,
    GREMLIN_NOB_V2_POLICY_ID,
    select_gremlin_nob_search_v2,
)
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator, public_sample
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig

SCHEMA = "sts1-phase1-public-search-v2-gremlin-nob-holdout-v1"
EXPECTED_SUITE_SCHEMA = "sts1-phase1-public-search-v2-gremlin-nob-holdout-suite-v1"
SEARCH_SAMPLE_COUNT = 8
ORACLE_SAMPLE_COUNT = 4
ORACLE_MCTS_SIMS = 2000
SEARCH_SAMPLING_SEED = 20260830


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("nob_v2_holdout_expected_json_object")
    return value


def digest_rows(rows: list[dict]) -> str:
    payload = "\n".join(canonical_json(row) for row in rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def oracle_scores_for_case(context, backend, config) -> dict[str, float]:
    groups: dict[str, list] = {}
    for action in context.legal_actions:
        groups.setdefault(action.semantic_key, []).append(action)
    totals = {action.action_id: 0.0 for action in context.legal_actions}
    for sample_index in range(ORACLE_SAMPLE_COUNT):
        sample = public_sample(context, sample_index=sample_index, config=config)
        for semantic_key in sorted(groups):
            members = sorted(groups[semantic_key], key=lambda item: item.action_id)
            representative = members[0]
            bc = backend._build(context, sample)
            native_action = backend._native_action(bc, representative)
            score = float(sts.judge_branch_action(bc, native_action, ORACLE_MCTS_SIMS)["score"])
            for member in members:
                totals[member.action_id] += score
    return {action_id: total / ORACLE_SAMPLE_COUNT for action_id, total in totals.items()}


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
    sources: dict[str, int] = {}
    unresolved = 0
    rows: list[dict] = []

    for record in suite["cases"]:
        state = record["public_state"]
        context = require_public_reconstruction(state)
        if context.decision_signature != record["decision_signature"]:
            raise RuntimeError(f"nob_v2_holdout_signature_drift:{record['case_id']}")
        enemy = state["enemies"][0]
        if state.get("turn") != 0 or enemy.get("name") != "GREMLIN_NOB" or enemy.get("intent") != "GREMLIN_NOB_BELLOW":
            raise RuntimeError(f"nob_v2_holdout_scope_drift:{record['case_id']}")

        backend = NativePublicGremlinNobRolloutV1(state, sts)
        search_result = PublicStateSearch(PairedPublicSampleEvaluator(backend), config).run(context)
        if not search_result.resolved or search_result.timed_out or search_result.unresolved_action_ids:
            unresolved += 1
        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError(f"nob_v2_holdout_baseline_missing:{record['case_id']}")
        v2 = select_gremlin_nob_search_v2(context, search_result, simple)
        sources[v2.source] = sources.get(v2.source, 0) + 1

        oracle_scores = oracle_scores_for_case(context, backend, config)
        ties = oracle_ties(oracle_scores)
        agreement = {
            "v2": single_action_agreement(v2.action_id, ties),
            "search": conservative_tie_agreement(tuple(sorted(search_result.tie_action_ids)), ties),
            "simple": single_action_agreement(simple.action_id, ties),
            "random": single_action_agreement(random_action.action_id, ties),
        }
        for key in counts:
            counts[key] += int(agreement[key])

        margin = v2.semantic_margin
        rows.append(
            {
                "schema_version": SCHEMA,
                "case_id": record["case_id"],
                "source_seed_provenance_only": record["source_seed_provenance_only"],
                "decision_signature": context.decision_signature,
                "policy_id": v2.policy_id,
                "v2_action_id": v2.action_id,
                "v2_source": v2.source,
                "v2_semantic_margin": margin if margin is None or math.isfinite(margin) else "inf",
                "search_tie_ids": list(sorted(search_result.tie_action_ids)),
                "simple_action_id": simple.action_id,
                "random_action_id": random_action.action_id,
                "oracle_tie_ids": list(ties),
                "agreement": agreement,
            }
        )

    reasons: list[str] = []
    if unresolved:
        reasons.append(f"search_unresolved:{unresolved}")
    if not counts["v2"] > counts["search"]:
        reasons.append("v2_not_strictly_better_than_raw_search")
    if not counts["v2"] > counts["simple"]:
        reasons.append("v2_not_strictly_better_than_simple")
    if not counts["v2"] > counts["random"]:
        reasons.append("v2_not_strictly_better_than_random")

    total = len(rows)
    summary = {
        "schema_version": SCHEMA,
        "scope": "fresh opening-turn Gremlin Nob V2 holdout only",
        "phase1_gate_claimed": False,
        "policy_id": GREMLIN_NOB_V2_POLICY_ID,
        "margin_threshold": GREMLIN_NOB_V2_MARGIN_THRESHOLD,
        "case_count": total,
        "agreement_counts": counts,
        "agreement_rates": {key: value / total for key, value in counts.items()},
        "selection_sources": dict(sorted(sources.items())),
        "search_unresolved": unresolved,
        "quality_verdict": "PASS" if not reasons else "FAIL",
        "quality_reasons": reasons,
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
    if (
        suite.get("schema_version") != EXPECTED_SUITE_SCHEMA
        or suite.get("case_count") != 24
        or suite.get("frozen_before_quality_results") is not True
        or suite.get("threshold_frozen_before_holdout_results") != GREMLIN_NOB_V2_MARGIN_THRESHOLD
    ):
        raise RuntimeError("nob_v2_holdout_freeze_contract_failed")

    rows1, summary1 = run_once(suite)
    rows2, summary2 = run_once(suite)
    if rows1 != rows2 or summary1 != summary2:
        raise RuntimeError("nob_v2_holdout_not_deterministic")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "decisions.ndjson").write_text("".join(canonical_json(row) + "\n" for row in rows1), encoding="utf-8")
    final = dict(summary1)
    final.update({"full_repeats": 2, "deterministic": True, "suite_digest": suite["suite_digest"]})
    (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True))
    print("PUBLIC_SEARCH_V2_GREMLIN_NOB_HOLDOUT_EXECUTION = PASS")
    print(f"GREMLIN_NOB_V2_HOLDOUT_VERDICT = {final['quality_verdict']}")
    print("DISCOVERY_350XXX_READ = 0")
    print("PRIOR_HOLDOUT_340XXX_READ = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
