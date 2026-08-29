#!/usr/bin/env python3
"""Formal fresh quality gate for frozen Anger+Pommel V2 on 520xxx."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from roguelike_ai.sts1_teacher.anger_pommel_composition_policy_v2 import (
    POLICY_ID,
    choose_anger_pommel_composition_candidate_v2,
)
from roguelike_ai.sts1_teacher.baselines import deterministic_random_legal, simple_public_heuristic
from roguelike_ai.sts1_teacher.benchmark import conservative_tie_agreement, oracle_ties, single_action_agreement
from roguelike_ai.sts1_teacher.contract import canonical_json
from roguelike_ai.sts1_teacher.native_rollout import NativePublicJawWormRolloutV1
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator, public_sample
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig

SCHEMA = "sts1-phase1-anger-pommel-composition-v2-holdout-quality-v1"
SUITE_SCHEMA = "sts1-phase1-anger-pommel-composition-fresh-holdout-suite-v1"
SAMPLING_CONTRACT = "anger-pommel-composition-v2-holdout-fixed-256-first24-v1"
GATE_ID = "anger-pommel-composition-v2-holdout-ge-search-gt-simple-random"
ORACLE_MCTS_SIMS = 2000
ORACLE_SAMPLE_COUNT = 4
SEARCH_SAMPLE_COUNT = 8
SEARCH_SAMPLING_SEED = 20260829


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected_json_object:{path}")
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
            judged = sts.judge_branch_action(bc, native_action, ORACLE_MCTS_SIMS)
            score = float(judged["score"])
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
    names = ("v2", "raw_search", "simple", "random")
    counts = {name: 0 for name in names}
    rows: list[dict] = []
    unresolved = 0
    illegal = 0
    failsafe_count = 0
    top_tie_break_count = 0
    intent_counts: Counter[str] = Counter()
    intent_agreement: dict[str, Counter[str]] = defaultdict(Counter)

    for record in suite["cases"]:
        state = record["public_state"]
        aux = record["reconstruction_aux"]
        context = require_public_reconstruction(state)
        if context.decision_signature != record["decision_signature"]:
            raise RuntimeError(f"suite_signature_drift:{record['case_id']}")
        if state.get("reconstruction", {}).get("anger_pommel_composition_complete") is not True:
            raise RuntimeError(f"suite_composition_admission_drift:{record['case_id']}")

        backend = NativePublicJawWormRolloutV1(state, sts, reconstruction_aux=aux)
        search_result = PublicStateSearch(PairedPublicSampleEvaluator(backend), config).run(context)
        if not search_result.resolved or search_result.timed_out or search_result.unresolved_action_ids:
            unresolved += 1

        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError(f"baseline_missing_action:{record['case_id']}")
        candidate = choose_anger_pommel_composition_candidate_v2(context, search_result, simple.action_id)
        failsafe_count += int(candidate.used_simple_failsafe)
        top_tie_break_count += int(candidate.used_simple_top_tie_break)

        legal_ids = {action.action_id for action in context.legal_actions}
        for name, action_id in {
            "v2": candidate.action_id,
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
            "v2": single_action_agreement(candidate.action_id, oracle_tie_ids),
            "raw_search": conservative_tie_agreement(search_ties, oracle_tie_ids),
            "simple": single_action_agreement(simple.action_id, oracle_tie_ids),
            "random": single_action_agreement(random_action.action_id, oracle_tie_ids),
        }
        intent = str(context.state["enemies"][0].get("intent", "UNKNOWN"))
        intent_counts[intent] += 1
        for name in names:
            counts[name] += int(agreement[name])
            intent_agreement[intent][name] += int(agreement[name])

        rows.append(
            {
                "schema_version": SCHEMA,
                "case_id": record["case_id"],
                "source_seed_provenance_only": record["source_seed_provenance_only"],
                "decision_signature": context.decision_signature,
                "public_intent": intent,
                "v2_action_id": candidate.action_id,
                "v2_used_simple_failsafe": candidate.used_simple_failsafe,
                "v2_used_simple_top_tie_break": candidate.used_simple_top_tie_break,
                "v2_top_semantic_count": candidate.top_semantic_count,
                "raw_search_tie_ids": list(search_ties),
                "raw_search_unique_best_action_id": search_result.unique_best_action_id,
                "simple_action_id": simple.action_id,
                "random_action_id": random_action.action_id,
                "oracle_tie_ids": list(oracle_tie_ids),
                "agreement": agreement,
            }
        )

    total = len(rows)
    reasons: list[str] = []
    if total != 24:
        reasons.append(f"case_count_not_24:{total}")
    if unresolved:
        reasons.append(f"search_unresolved:{unresolved}")
    if illegal:
        reasons.append(f"illegal:{illegal}")
    if counts["v2"] < counts["raw_search"]:
        reasons.append("v2_worse_than_raw_search")
    if counts["v2"] <= counts["simple"]:
        reasons.append("v2_not_strictly_better_than_simple")
    if counts["v2"] <= counts["random"]:
        reasons.append("v2_not_strictly_better_than_random")

    summary = {
        "schema_version": SCHEMA,
        "scope": "reachable Jaw Worm second-player-turn post-Anger+Pommel public composition states only",
        "gate_id": GATE_ID,
        "candidate_policy_id": POLICY_ID,
        "case_count": total,
        "agreement_counts": counts,
        "agreement_rates": {name: counts[name] / total for name in names},
        "candidate_simple_failsafe_count": failsafe_count,
        "candidate_simple_top_tie_break_count": top_tie_break_count,
        "search_unresolved": unresolved,
        "illegal": illegal,
        "quality_verdict": "PASS" if not reasons else "FAIL",
        "quality_reasons": reasons,
        "public_intent_breakdown": {
            intent: {"cases": intent_counts[intent], **{name: intent_agreement[intent][name] for name in names}}
            for intent in sorted(intent_counts)
        },
        "search_samples_per_semantic_action": SEARCH_SAMPLE_COUNT,
        "oracle_samples": ORACLE_SAMPLE_COUNT,
        "oracle_mcts_sims_per_action_per_sample": ORACLE_MCTS_SIMS,
        "rows_digest": digest_rows(rows),
        "phase1_gate_claimed": False,
        "phase2_locked": True,
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
    if suite.get("purpose") != "V2_FORMAL_FRESH_HOLDOUT":
        raise RuntimeError("formal_holdout_purpose_missing")
    if suite.get("case_count") != 24 or suite.get("frozen_before_quality_results") is not True:
        raise RuntimeError("suite_freeze_contract_failed")
    if suite.get("quality_results_observed_during_selection") != 0:
        raise RuntimeError("suite_selection_was_not_blind")
    if suite.get("prior_500xxx_quality_rows_read") != 0:
        raise RuntimeError("v1_holdout_rows_leaked")
    if suite.get("prior_510xxx_discovery_rows_used_for_selection") != 0:
        raise RuntimeError("discovery_rows_leaked_into_holdout_selection")
    for key in (
        "source_battle_context_exported",
        "source_hidden_rng_exported",
        "source_move_history_exported",
        "source_counter_history_exported",
    ):
        if suite.get(key) != 0:
            raise RuntimeError(f"suite_hidden_source_export:{key}:{suite.get(key)}")

    rows1, summary1 = run_once(suite)
    rows2, summary2 = run_once(suite)
    if rows1 != rows2 or summary1 != summary2:
        raise RuntimeError("composition_v2_holdout_not_deterministic")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "decisions.ndjson").write_text("".join(canonical_json(row) + "\n" for row in rows1), encoding="utf-8")
    final = dict(summary1)
    final.update(
        {
            "full_repeats": 2,
            "deterministic": True,
            "suite_digest": suite["suite_digest"],
            "source_seed_file_sha256": suite["source_seed_file_sha256"],
            "source_seed_range": suite["source_seed_range"],
            "sampling_contract_version": suite["sampling_contract_version"],
            "public_contract_version": "sts1-public-state-v1",
            "reconstruction_aux_version": "sts1-public-reconstruction-aux-v1",
        }
    )
    (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True))
    print("ANGER_POMMEL_V2_HOLDOUT_EXECUTION = PASS")
    print(f"ANGER_POMMEL_V2_HOLDOUT_VERDICT = {final['quality_verdict']}")
    print("PHASE1_GATE_CLAIMED = 0")
    print("PHASE2 = LOCKED")


if __name__ == "__main__":
    main()
