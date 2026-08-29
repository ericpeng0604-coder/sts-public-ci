#!/usr/bin/env python3
"""Run the predeclared fresh quality gate for Anger+Pommel composition states."""
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

from roguelike_ai.sts1_teacher.anger_pommel_composition_policy_v1 import (
    MARGIN_THRESHOLD,
    POLICY_ID,
    choose_anger_pommel_composition_candidate,
)
from roguelike_ai.sts1_teacher.baselines import deterministic_random_legal, simple_public_heuristic
from roguelike_ai.sts1_teacher.benchmark import conservative_tie_agreement, oracle_ties, single_action_agreement
from roguelike_ai.sts1_teacher.contract import canonical_json
from roguelike_ai.sts1_teacher.native_rollout import NativePublicJawWormRolloutV1
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator, public_sample
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig

SCHEMA = "sts1-phase1-anger-pommel-composition-fresh-quality-v1"
SUITE_SCHEMA = "sts1-phase1-anger-pommel-composition-fresh-holdout-suite-v1"
SAMPLING_CONTRACT = "anger-pommel-composition-fresh-fixed-256-first24-v1"
ORACLE_MCTS_SIMS = 2000
ORACLE_SAMPLE_COUNT = 4
SEARCH_SAMPLE_COUNT = 8
SEARCH_SAMPLING_SEED = 20260829

# Frozen before any 500xxx quality result:
# candidate >= Raw Search, candidate > Simple, candidate > Random,
# unresolved=0, illegal=0, exact two-repeat determinism, exactly 24 fresh cases.
GATE_ID = "anger-pommel-composition-fresh-v1-candidate-ge-search-gt-simple-random"


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
        rollout_budget=256,
        node_budget=16384,
        max_depth=30,
        timeout_ms=60_000,
        tie_tolerance=1e-9,
        sampling_seed=SEARCH_SAMPLING_SEED,
    )
    rows: list[dict] = []
    counts = {"candidate": 0, "raw_search": 0, "simple": 0, "random": 0}
    unresolved = 0
    illegal = 0
    fallback_count = 0
    intent_counts: Counter[str] = Counter()
    intent_agreement: dict[str, Counter[str]] = defaultdict(Counter)

    for record in suite["cases"]:
        state = record["public_state"]
        aux = record["reconstruction_aux"]
        context = require_public_reconstruction(state)
        if context.decision_signature != record["decision_signature"]:
            raise RuntimeError(f"suite_signature_drift:{record['case_id']}")
        marker = state.get("reconstruction", {})
        if marker.get("anger_pommel_composition_complete") is not True:
            raise RuntimeError(f"suite_composition_admission_drift:{record['case_id']}")
        if context.state.get("turn") != 1 or len(context.state.get("hand", [])) != 4:
            raise RuntimeError(f"suite_midturn_drift:{record['case_id']}")

        backend = NativePublicJawWormRolloutV1(state, sts, reconstruction_aux=aux)
        evaluator = PairedPublicSampleEvaluator(backend)
        search_result = PublicStateSearch(evaluator, config).run(context)
        if not search_result.resolved or search_result.timed_out or search_result.unresolved_action_ids:
            unresolved += 1

        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError(f"baseline_missing_action:{record['case_id']}")
        candidate = choose_anger_pommel_composition_candidate(context, search_result, simple.action_id)
        fallback_count += int(candidate.used_simple_fallback)

        legal_ids = {action.action_id for action in context.legal_actions}
        chosen = {
            "candidate": candidate.action_id,
            "simple": simple.action_id,
            "random": random_action.action_id,
        }
        for name, action_id in chosen.items():
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
        intent = str(context.state["enemies"][0].get("intent", "UNKNOWN"))
        intent_counts[intent] += 1
        for key in counts:
            counts[key] += int(agreement[key])
            intent_agreement[intent][key] += int(agreement[key])

        rows.append(
            {
                "schema_version": SCHEMA,
                "case_id": record["case_id"],
                "source_seed_provenance_only": record["source_seed_provenance_only"],
                "decision_signature": context.decision_signature,
                "public_intent": intent,
                "legal_action_ids": sorted(legal_ids),
                "candidate_action_id": candidate.action_id,
                "candidate_used_simple_fallback": candidate.used_simple_fallback,
                "candidate_semantic_margin": candidate.semantic_margin,
                "candidate_policy_id": POLICY_ID,
                "raw_search_tie_ids": list(search_ties),
                "raw_search_unique_best_action_id": search_result.unique_best_action_id,
                "raw_search_evidence_hash": search_result.evidence_hash,
                "raw_search_rollout_count": search_result.rollout_count,
                "simple_action_id": simple.action_id,
                "random_action_id": random_action.action_id,
                "oracle_scores": dict(sorted(oracle_scores.items())),
                "oracle_tie_ids": list(oracle_tie_ids),
                "agreement": agreement,
            }
        )

    total = len(rows)
    rates = {name: counts[name] / total for name in counts}
    reasons: list[str] = []
    if total != 24:
        reasons.append(f"case_count_not_24:{total}")
    if unresolved:
        reasons.append(f"search_unresolved:{unresolved}")
    if illegal:
        reasons.append(f"illegal:{illegal}")
    if not rates["candidate"] >= rates["raw_search"]:
        reasons.append("candidate_worse_than_raw_search")
    if not rates["candidate"] > rates["simple"]:
        reasons.append("candidate_not_strictly_better_than_simple")
    if not rates["candidate"] > rates["random"]:
        reasons.append("candidate_not_strictly_better_than_random")

    intent_breakdown = {
        intent: {
            "cases": intent_counts[intent],
            **{name: intent_agreement[intent][name] for name in counts},
        }
        for intent in sorted(intent_counts)
    }
    summary = {
        "schema_version": SCHEMA,
        "scope": "reachable Jaw Worm second-player-turn post-Anger+Pommel public composition states only",
        "phase1_gate_claimed": False,
        "phase2_locked": True,
        "gate_id": GATE_ID,
        "candidate_policy_id": POLICY_ID,
        "candidate_margin_threshold": MARGIN_THRESHOLD,
        "case_count": total,
        "search_samples_per_semantic_action": SEARCH_SAMPLE_COUNT,
        "oracle_samples": ORACLE_SAMPLE_COUNT,
        "oracle_mcts_sims_per_action_per_sample": ORACLE_MCTS_SIMS,
        "agreement_counts": counts,
        "agreement_rates": rates,
        "candidate_fallback_count": fallback_count,
        "public_intent_breakdown": intent_breakdown,
        "search_unresolved": unresolved,
        "illegal": illegal,
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
    if suite.get("schema_version") != SUITE_SCHEMA:
        raise RuntimeError(f"unexpected_suite_schema:{suite.get('schema_version')}")
    if suite.get("sampling_contract_version") != SAMPLING_CONTRACT:
        raise RuntimeError(f"unexpected_sampling_contract:{suite.get('sampling_contract_version')}")
    if suite.get("case_count") != 24 or suite.get("frozen_before_quality_results") is not True:
        raise RuntimeError("suite_freeze_contract_failed")
    if suite.get("quality_results_observed_during_selection") != 0:
        raise RuntimeError("suite_selection_was_not_blind")
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
        raise RuntimeError("anger_pommel_composition_fresh_quality_not_deterministic")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "decisions.ndjson").write_text(
        "".join(canonical_json(row) + "\n" for row in rows1),
        encoding="utf-8",
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
            "public_contract_version": "sts1-public-state-v1",
            "reconstruction_aux_version": "sts1-public-reconstruction-aux-v1",
        }
    )
    (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True))
    print("ANGER_POMMEL_COMPOSITION_FRESH_QUALITY_EXECUTION = PASS")
    print(f"ANGER_POMMEL_COMPOSITION_FRESH_QUALITY_VERDICT = {final['quality_verdict']}")
    print("PHASE1_GATE_CLAIMED = 0")
    print("PHASE2 = LOCKED")


if __name__ == "__main__":
    main()
