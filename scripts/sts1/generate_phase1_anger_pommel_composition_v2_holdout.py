#!/usr/bin/env python3
"""Blind formal 520xxx holdout suite for frozen Anger+Pommel V2 policy.

This reuses the already-audited reachable-state mechanics. Only the committed
source seed supplier and holdout labels change. V2 policy/search/oracle settings
are not consulted during selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_phase1_anger_pommel_composition_fresh_holdout as base

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_anger_pommel_composition_v2_holdout_seeds_256.txt"
MIN_SEED = 520001
MAX_SEED = 520256
EXPECTED_SOURCE_SEEDS = 256
SAMPLING_CONTRACT = "anger-pommel-composition-v2-holdout-fixed-256-first24-v1"


def generate() -> dict:
    base.SEED_FILE = SEED_FILE
    base.MIN_SEED = MIN_SEED
    base.MAX_SEED = MAX_SEED
    base.EXPECTED_SOURCE_SEEDS = EXPECTED_SOURCE_SEEDS
    base.SAMPLING_CONTRACT = SAMPLING_CONTRACT
    suite = base.generate()
    suite["sampling_contract_version"] = SAMPLING_CONTRACT
    suite["selection_rule"] = (
        "first 24 unique reachable admitted Anger+Pommel post-composition public signatures in committed "
        "520001..520256 ascending order; selection uses only public second-turn rich-card presence and formal "
        "reconstruction admission, never Search/oracle quality"
    )
    suite["source_seed_file"] = str(SEED_FILE.relative_to(ROOT))
    suite["source_seed_file_sha256"] = hashlib.sha256(SEED_FILE.read_bytes()).hexdigest()
    suite["source_seed_range"] = [MIN_SEED, MAX_SEED]
    suite["quality_results_observed_during_selection"] = 0
    suite["prior_500xxx_quality_rows_read"] = 0
    suite["prior_510xxx_discovery_rows_used_for_selection"] = 0
    suite["purpose"] = "V2_FORMAL_FRESH_HOLDOUT"
    return suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ANGER_POMMEL_V2_HOLDOUT_SUITE_FREEZE = PASS")
    print(f"SAMPLING_CONTRACT = {suite['sampling_contract_version']}")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"PUBLIC_INTENT_COUNTS = {suite['public_intent_counts']}")
    print(f"REJECTED_COUNTS = {suite['rejected_counts']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("PRIOR_500XXX_QUALITY_ROWS_READ = 0")
    print("PRIOR_510XXX_DISCOVERY_ROWS_USED_FOR_SELECTION = 0")


if __name__ == "__main__":
    main()
