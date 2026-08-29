#!/usr/bin/env python3
"""Run frozen quality benchmark for audited Jaw Worm two-card midturn states."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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

SCHEMA = "sts1-phase1-public-search-v1-jaw-worm-turn-one-two-card-midturn-quality-v1"
SUITE_SCHEMA = "sts1-phase1-public-search-v1-jaw-worm-turn-one-two-card-midturn-suite-v1"
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
    counts = {"search": 0, "simple": 0, "random": 0}
    unresolved = 0
    intent_counts: Counter[str] = Counter()
    intent_agreement: dict[str, Counter[str]] = defaultdict(Counter)
    sequence_counts: Counter[str] = Counter()
    sequence_agreement: dict[str, Counter[str]] = defaultdict(Counter)

    for record in suite["cases"]:
        state = record["public_state"]
        context = require_public_reconstruction(state)
        if context.decision_signature != record["decision_signature"]:
            raise RuntimeError(f"suite_signature_drift:{record['case_id']}")
        if context.state.get("turn") != 1 or len(context.state.get("hand", [])) != 3:
            raise RuntimeError(f"suite_two_card_midturn_drift:{record['case_id']}")
        if context.state.get("draw_pile") != [] or len(context.state.get("discard_pile", [])) != 7:
            raise RuntimeError(f"suite_two_card_pile_drift:{record['case_id']}")

        enemy = context.state["enemies"][0]
        intent = str(enemy.get("intent", "UNKNOWN"))
        sequence = str(record.get("generation_sequence", "UNKNOWN"))
        intent_counts[intent] += 1
        sequence_counts[sequence] += 1

        backend = NativePublicJawWormRolloutV1(state, sts)
        evaluator = PairedPublicSampleEvaluator(backend)
        search_result = PublicStateSearch(evaluator, config).run(context)
        if not search_result.resolved or search_result.timed_out or search_result.unresolved_action_ids:
            unresolved += 1

        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError(f"baseline_missing_action:{record['case_id']}")

        oracle_scores = oracle_scores_for_case(context, backend, config)
        oracle_tie_ids = oracle_ties(oracle_scores)
        search_ties = tuple(sorted(search_result.tie_action_ids))
        agreement = {
            "search": conservative_tie_agreement(search_ties, oracle_tie_ids),
            "simple": single_action_agreement(simple.action_id, oracle_tie_ids),
            "random": single_action_agreement(random_action.action_id, oracle_tie_ids),
        }
        for key in counts:
            counts[key] += int(agreement[key])
            intent_agreement[intent][key] += int(agreement[key])
            sequence_agreement[sequence][key] += int(agreement[key])

        rows.append(
            {
                "schema_version": SCHEMA,
                "case_id": record["case_id"],
                "source_seed_provenance_only": record["source_seed_provenance_only"],
                "generation_first_card": record.get("generation_first_card"),
                "generation_second_card": record.get("generation_second_card"),
                "generation_sequence": sequence,
                "decision_signature": context.decision_signature,
                "public_intent": intent,
                "legal_action_ids": sorted(action.action_id for action in context.legal_actions),
                "search_tie_ids": list(search_ties),
                "search_unique_best_action_id": search_result.unique_best_action_id,
                "search_candidate_scores": [
                    {
                        "action_id": item.action_id,
                        "semantic_key": item.semantic_key,
                        "score": item.score,
                        "samples": item.samples,
                        "unresolved": item.unresolved,
                    }
                    for item in search_result.candidate_scores
                ],
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

    intent_breakdown = {
        intent: {
            "cases": intent_counts[intent],
            "search": intent_agreement[intent]["search"],
            "simple": intent_agreement[intent]["simple"],
            "random": intent_agreement[intent]["random"],
        }
        for intent in sorted(intent_counts)
    }
    sequence_breakdown = {
        sequence: {
            "cases": sequence_counts[sequence],
            "search": sequence_agreement[sequence]["search"],
            "simple": sequence_agreement[sequence]["simple"],
            "random": sequence_agreement[sequence]["random"],
        }
        for sequence in sorted(sequence_counts)
    }
    summary = {
        "schema_version": SCHEMA,
        "scope": "Jaw Worm second-player-turn audited exactly-two-starter-card midturn public states only",
        "phase1_gate_claimed": False,
        "hand_size_shortcut_generalized": False,
        "nonstarter_cards_admitted": 0,
        "case_count": total,
        "search_samples_per_semantic_action": SEARCH_SAMPLE_COUNT,
        "oracle_samples": ORACLE_SAMPLE_COUNT,
        "oracle_mcts_sims_per_action_per_sample": ORACLE_MCTS_SIMS,
        "agreement_counts": counts,
        "agreement_rates": rates,
        "public_intent_breakdown": intent_breakdown,
        "generation_sequence_breakdown": sequence_breakdown,
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
    if suite.get("schema_version") != SUITE_SCHEMA:
        raise RuntimeError(f"unexpected_suite_schema:{suite.get('schema_version')}")
    if suite.get("case_count") != 24 or suite.get("frozen_before_quality_results") is not True:
        raise RuntimeError("suite_freeze_contract_failed")
    if suite.get("quality_results_observed_during_selection") != 0:
        raise RuntimeError("suite_selection_was_not_blind")
    if suite.get("source_counter_history_exported") != 0 or suite.get("source_move_history_exported") != 0:
        raise RuntimeError("suite_exported_hidden_history")
    if suite.get("hand_size_shortcut_generalized") is not False:
        raise RuntimeError("suite_generalized_hand_size_shortcut")
    if suite.get("nonstarter_cards_admitted") != 0:
        raise RuntimeError("suite_admitted_nonstarter_cards")

    rows1, summary1 = run_once(suite)
    rows2, summary2 = run_once(suite)
    if rows1 != rows2 or summary1 != summary2:
        raise RuntimeError("jaw_worm_two_card_midturn_quality_not_deterministic")

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
            "source_seed_file_sha256": suite["source_seed_file_sha256"],
        }
    )
    (out / "summary.json").write_text(json.dumps(final_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    print("PUBLIC_SEARCH_V1_JAW_WORM_TURN_ONE_TWO_CARD_MIDTURN_QUALITY_EXECUTION = PASS")
    print(f"JAW_WORM_TURN_ONE_TWO_CARD_MIDTURN_QUALITY_VERDICT = {final_summary['quality_verdict']}")
    print("MIDTURN_HAND_SIZE_SHORTCUT_GENERALIZED = 0")
    print("NONSTARTER_CARDS_ADMITTED = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
