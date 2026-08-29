#!/usr/bin/env python3
"""Evaluate the frozen Anger+Finesse candidate on 530xxx discovery only."""
from __future__ import annotations

import argparse
from collections import Counter
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

from roguelike_ai.sts1_teacher.anger_finesse_composition_policy_v1 import (
    POLICY_ID,
    choose_anger_finesse_composition_candidate,
)
from roguelike_ai.sts1_teacher.baselines import deterministic_random_legal, simple_public_heuristic
from roguelike_ai.sts1_teacher.benchmark import conservative_tie_agreement, oracle_ties, single_action_agreement
from roguelike_ai.sts1_teacher.contract import canonical_json
from roguelike_ai.sts1_teacher.native_rollout import NativePublicJawWormRolloutV1
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator, public_sample
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig

SCHEMA = "sts1-phase1-anger-finesse-composition-discovery-v1"
SUITE_SCHEMA = "sts1-phase1-anger-finesse-composition-discovery-suite-v1"
SAMPLING_CONTRACT = "anger-finesse-composition-discovery-fixed-256-first24-v1"
ORACLE_MCTS_SIMS = 2000
ORACLE_SAMPLE_COUNT = 4
SEARCH_SAMPLE_COUNT = 8
SEARCH_ROLLOUT_BUDGET = 256
SEARCH_NODE_BUDGET = 16384
SEARCH_MAX_DEPTH = 30
SEARCH_TIMEOUT_MS = 60_000
SEARCH_SAMPLING_SEED = 20260830
DISCOVERY_GATE_ID = "anger-finesse-discovery-candidate-ge-raw-gt-simple-random-v1"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected_json_object:{path}")
    return value


def digest_rows(rows: list[dict]) -> str:
    payload = "\n".join(canonical_json(row) for row in rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def oracle_scores_for_case(context, backend, config) -> dict[str, float]:
    semantic_groups: dict[str, list] = {}
    for action in context.legal_actions:
        semantic_groups.setdefault(action.semantic_key, []).append(action)
    totals = {action.action_id: 0.0 for action in context.legal_actions}
    for sample_index in range(ORACLE_SAMPLE_COUNT):
        sample = public_sample(context, sample_index=sample_index, config=config)
        for semantic_key in sorted(semantic_groups):
            members = sorted(semantic_groups[semantic_key], key=lambda item: item.action_id)
            representative = members[0]
            bc = backend._build(context, sample)
            native_action = backend._native_action(bc, representative)
            judged = sts.judge_branch_action(bc, native_action, ORACLE_MCTS_SIMS)
            score = float(judged["score"])
            for member in members:
                totals[member.action_id] += score
    return {action_id: total / ORACLE_SAMPLE_COUNT for action_id, total in totals.items()}


def run_once(suite: dict) -> tuple[list[dict], dict]:
    config = SearchConfig(
        samples_per_semantic_action=SEARCH_SAMPLE_COUNT,
        rollout_budget=SEARCH_ROLLOUT_BUDGET,
        node_budget=SEARCH_NODE_BUDGET,
        max_depth=SEARCH_MAX_DEPTH,
        timeout_ms=SEARCH_TIMEOUT_MS,
        tie_tolerance=1e-9,
        sampling_seed=SEARCH_SAMPLING_SEED,
    )
    names = ("candidate", "raw_search", "simple", "random")
    counts = {name: 0 for name in names}
    rows: list[dict] = []
    unresolved = 0
    illegal = 0
    failsafes = 0
    top_tie_breaks = 0
    top_semantic_counts: Counter[int] = Counter()

    for record in suite["cases"]:
        state = record["public_state"]
        aux = record["reconstruction_aux"]
        context = require_public_reconstruction(state)
        if context.decision_signature != record["decision_signature"]:
            raise RuntimeError(f"suite_signature_drift:{record['case_id']}")
        if record.get("order_observation_equivalent") is not True:
            raise RuntimeError(f"order_equivalence_not_proven:{record['case_id']}")

        backend = NativePublicJawWormRolloutV1(state, sts, reconstruction_aux=aux)
        evaluator = PairedPublicSampleEvaluator(backend)
        search_result = PublicStateSearch(evaluator, config).run(context)
        if not search_result.resolved or search_result.timed_out or search_result.unresolved_action_ids:
            unresolved += 1

        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError(f"baseline_missing_action:{record['case_id']}")

        candidate = choose_anger_finesse_composition_candidate(context, search_result, simple.action_id)
        failsafes += int(candidate.used_simple_failsafe)
        top_tie_breaks += int(candidate.used_simple_top_tie_break)
        top_semantic_counts[candidate.top_semantic_count] += 1

        legal_ids = {action.action_id for action in context.legal_actions}
        for name, action_id in {
            "candidate": candidate.action_id,
            "simple": simple.action_id,
            "random": random_action.action_id,
        }.items():
            if action_id not in legal_ids:
                illegal += 1
                raise RuntimeError(f"illegal_{name}_action:{record['case_id']}:{action_id}")

        oracle_scores = oracle_scores_for_case(context, backend, config)
        oracle_tie_ids = oracle_ties(oracle_scores)
        search_ties = tuple(sorted(search_result.tie_action_ids))
        agreement = {
            "candidate": single_action_agreement(candidate.action_id, oracle_tie_ids),
            "raw_search": conservative_tie_agreement(search_ties, oracle_tie_ids),
            "simple": single_action_agreement(simple.action_id, oracle_tie_ids),
            "random": single_action_agreement(random_action.action_id, oracle_tie_ids),
        }
        for name in names:
            counts[name] += int(agreement[name])

        rows.append(
            {
                "schema_version": SCHEMA,
                "case_id": record["case_id"],
                "source_seed_provenance_only": record["source_seed_provenance_only"],
                "decision_signature": context.decision_signature,
                "public_intent": str(context.state["enemies"][0].get("intent", "UNKNOWN")),
                "candidate_action_id": candidate.action_id,
                "candidate_used_simple_failsafe": candidate.used_simple_failsafe,
                "candidate_used_simple_top_tie_break": candidate.used_simple_top_tie_break,
                "candidate_top_semantic_count": candidate.top_semantic_count,
                "raw_search_tie_ids": list(search_ties),
                "simple_action_id": simple.action_id,
                "random_action_id": random_action.action_id,
                "oracle_tie_ids": list(oracle_tie_ids),
                "agreement": agreement,
            }
        )

    total = len(rows)
    reasons: list[str] = []
    if total < 24:
        reasons.append(f"case_count_below_24:{total}")
    if unresolved:
        reasons.append(f"search_unresolved:{unresolved}")
    if illegal:
        reasons.append(f"illegal:{illegal}")
    if counts["candidate"] < counts["raw_search"]:
        reasons.append("candidate_worse_than_raw_search")
    if counts["candidate"] <= counts["simple"]:
        reasons.append("candidate_not_strictly_better_than_simple")
    if counts["candidate"] <= counts["random"]:
        reasons.append("candidate_not_strictly_better_than_random")

    summary = {
        "schema_version": SCHEMA,
        "purpose": "DISCOVERY_ONLY_NOT_HOLDOUT",
        "discovery_gate_id": DISCOVERY_GATE_ID,
        "candidate_policy_id": POLICY_ID,
        "case_count": total,
        "agreement_counts": counts,
        "agreement_rates": {name: counts[name] / total for name in names},
        "candidate_simple_failsafe_count": failsafes,
        "candidate_simple_top_tie_break_count": top_tie_breaks,
        "candidate_top_semantic_count_histogram": dict(sorted(top_semantic_counts.items())),
        "search_unresolved": unresolved,
        "illegal": illegal,
        "oracle_samples": ORACLE_SAMPLE_COUNT,
        "oracle_mcts_sims_per_action_per_sample": ORACLE_MCTS_SIMS,
        "search_samples_per_semantic_action": SEARCH_SAMPLE_COUNT,
        "search_rollout_budget": SEARCH_ROLLOUT_BUDGET,
        "search_node_budget": SEARCH_NODE_BUDGET,
        "search_max_depth": SEARCH_MAX_DEPTH,
        "search_timeout_ms": SEARCH_TIMEOUT_MS,
        "search_sampling_seed": SEARCH_SAMPLING_SEED,
        "discovery_verdict": "PASS" if not reasons else "FAIL",
        "discovery_reasons": reasons,
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
    if suite.get("schema_version") != SUITE_SCHEMA:
        raise RuntimeError(f"unexpected_suite_schema:{suite.get('schema_version')}")
    if suite.get("sampling_contract_version") != SAMPLING_CONTRACT:
        raise RuntimeError(f"unexpected_sampling_contract:{suite.get('sampling_contract_version')}")
    if suite.get("purpose") != "DISCOVERY_ONLY_NOT_HOLDOUT":
        raise RuntimeError("discovery_purpose_missing")
    if suite.get("quality_results_observed_during_selection") != 0:
        raise RuntimeError("discovery_selection_was_not_blind")
    for key in ("prior_500xxx_quality_rows_read", "prior_510xxx_quality_rows_read", "prior_520xxx_quality_rows_read"):
        if suite.get(key) != 0:
            raise RuntimeError(f"prior_quality_rows_leaked:{key}")
    for key in (
        "source_battle_context_exported",
        "source_hidden_rng_exported",
        "source_move_history_exported",
        "source_counter_history_exported",
    ):
        if suite.get(key) != 0:
            raise RuntimeError(f"hidden_generation_state_exported:{key}")

    rows1, summary1 = run_once(suite)
    rows2, summary2 = run_once(suite)
    if rows1 != rows2 or summary1 != summary2:
        raise RuntimeError("anger_finesse_discovery_not_deterministic")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "decisions.ndjson").write_text(
        "".join(canonical_json(row) + "\n" for row in rows1), encoding="utf-8"
    )
    final = dict(summary1)
    final.update(
        {
            "full_repeats": 2,
            "deterministic": True,
            "suite_digest": suite["suite_digest"],
            "source_seed_file_sha256": suite["source_seed_file_sha256"],
            "source_seed_range": suite["source_seed_range"],
            "sampling_contract_version": suite["sampling_contract_version"],
            "order_observation_equivalence_required": suite["order_observation_equivalence_required"],
            "phase1_status": "IN_PROGRESS",
            "phase1_gate_claimed": False,
            "phase2_locked": True,
        }
    )
    (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True))
    print("ANGER_FINESSE_DISCOVERY_EXECUTION = PASS")
    print(f"ANGER_FINESSE_DISCOVERY_VERDICT = {final['discovery_verdict']}")
    print("PHASE1_STATUS = IN_PROGRESS")
    print("PHASE1_GATE_CLAIMED = 0")
    print("PHASE2 = LOCKED")


if __name__ == "__main__":
    main()
