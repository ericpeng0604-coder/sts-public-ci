#!/usr/bin/env python3
"""Blind 540xxx fresh holdout for the frozen Anger + Finesse candidate.

This file selects cases without Search/oracle quality labels. It deliberately
reuses the already-frozen public generation/admission helper from 530 discovery
so only the source seeds and holdout metadata change. The two play orders must
still collapse to the exact same formal public DecisionContext or the seed is
rejected. No discovery decision/evidence file is read.
"""
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

from roguelike_ai.sts1_teacher.contract import canonical_json

# Import code only: this does not read any 530xxx generated suite/quality rows.
from scripts.sts1.generate_phase1_anger_finesse_composition_discovery import (
    _legal_observation,
    _run_order,
    aux,
)

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_anger_finesse_composition_fresh_holdout_seeds_256.txt"
SUITE_SCHEMA = "sts1-phase1-anger-finesse-composition-fresh-holdout-suite-v1"
SAMPLING_CONTRACT = "anger-finesse-composition-fresh-holdout-fixed-256-first24-v1"
EXPECTED_SOURCE_SEEDS = 256
MIN_SEED = 540001
MAX_SEED = 540256
TARGET_CASES = 24


def load_seeds() -> list[int]:
    seeds = [
        int(line.strip())
        for line in SEED_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != EXPECTED_SOURCE_SEEDS or len(set(seeds)) != EXPECTED_SOURCE_SEEDS:
        raise RuntimeError(f"source_seed_contract:{len(seeds)}:{len(set(seeds))}")
    if seeds != list(range(MIN_SEED, MAX_SEED + 1)):
        raise RuntimeError(f"unexpected_source_seed_sequence:{seeds[:1]}:{seeds[-1:]}")
    return seeds


def suite_digest(records: list[dict]) -> str:
    payload = "\n".join(canonical_json(record) for record in records) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate() -> dict:
    records: list[dict] = []
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    intents: Counter[str] = Counter()
    equivalent_orders = 0

    for source_seed in load_seeds():
        try:
            af_state, af_context, af_actions, af_intent = _run_order(source_seed, ("ANGER", "FINESSE"))
            fa_state, fa_context, fa_actions, fa_intent = _run_order(source_seed, ("FINESSE", "ANGER"))
            if af_intent != fa_intent:
                raise RuntimeError("order_public_intent_mismatch")
            if canonical_json(af_context.state) != canonical_json(fa_context.state):
                raise RuntimeError("order_formal_state_mismatch")
            if canonical_json(_legal_observation(af_context)) != canonical_json(_legal_observation(fa_context)):
                raise RuntimeError("order_legal_actions_mismatch")
            if af_context.decision_signature != fa_context.decision_signature:
                raise RuntimeError("order_decision_signature_mismatch")
            equivalent_orders += 1
            if af_context.decision_signature in seen:
                rejected["duplicate_signature"] += 1
                continue
        except Exception as exc:
            reason = str(exc).split(":", 1)[0] or exc.__class__.__name__
            rejected[reason] += 1
            continue

        seen.add(af_context.decision_signature)
        intents[af_intent] += 1
        records.append(
            {
                "case_id": f"anger-finesse-composition-fresh-holdout-v1-{len(records):02d}",
                "source_seed_provenance_only": source_seed,
                "generation_rule": (
                    "add one normal Anger and one normal Finesse before combat; end opening player turn without cards; "
                    "require both in the second-turn public hand; replay both public play orders from the same seed; "
                    "retain only if their formal public observations and legal actions are identical"
                ),
                "anger_then_finesse_public_action_ids": af_actions,
                "finesse_then_anger_public_action_ids": fa_actions,
                "order_observation_equivalent": True,
                "decision_signature": af_context.decision_signature,
                "public_state": af_state,
                "reconstruction_aux": aux(),
            }
        )
        if len(records) == TARGET_CASES:
            break

    if len(records) != TARGET_CASES:
        raise RuntimeError(
            f"insufficient_unique_admitted_holdout_cases:{len(records)}/{TARGET_CASES}:rejected={dict(rejected)}"
        )

    return {
        "schema_version": SUITE_SCHEMA,
        "sampling_contract_version": SAMPLING_CONTRACT,
        "purpose": "FRESH_HOLDOUT_ONLY",
        "frozen_before_quality_results": True,
        "selection_rule": (
            "first 24 unique reachable admitted Anger+Finesse formal public signatures in committed 540001..540256 "
            "ascending order; selection uses only public rich-card presence, exact formal admission, and cross-order "
            "formal-observation equivalence, never Search/oracle quality"
        ),
        "source_seed_file": str(SEED_FILE.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "source_seed_range": [MIN_SEED, MAX_SEED],
        "quality_results_observed_during_selection": 0,
        "discovery_decision_rows_read": 0,
        "prior_500xxx_quality_rows_read": 0,
        "prior_510xxx_quality_rows_read": 0,
        "prior_520xxx_quality_rows_read": 0,
        "prior_530xxx_quality_rows_read": 0,
        "source_battle_context_exported": 0,
        "source_hidden_rng_exported": 0,
        "source_move_history_exported": 0,
        "source_counter_history_exported": 0,
        "order_observation_equivalence_required": True,
        "order_equivalent_source_count_before_stop": equivalent_orders,
        "case_count": len(records),
        "public_intent_counts": dict(sorted(intents.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "suite_digest": suite_digest(records),
        "cases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ANGER_FINESSE_FRESH_HOLDOUT_SUITE_FREEZE = PASS")
    print(f"SAMPLING_CONTRACT = {suite['sampling_contract_version']}")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"PUBLIC_INTENT_COUNTS = {suite['public_intent_counts']}")
    print(f"REJECTED_COUNTS = {suite['rejected_counts']}")
    print(f"ORDER_EQUIVALENT_SOURCE_COUNT_BEFORE_STOP = {suite['order_equivalent_source_count_before_stop']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("DISCOVERY_DECISION_ROWS_READ = 0")
    print("PRIOR_530XXX_QUALITY_ROWS_READ = 0")
    print("SOURCE_HIDDEN_GENERATION_STATE_EXPORTED = 0")


if __name__ == "__main__":
    main()
