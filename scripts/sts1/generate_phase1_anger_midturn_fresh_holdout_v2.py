#!/usr/bin/env python3
"""Second frozen Anger holdout collection after 480xxx was insufficient.

Sampling contract was frozen before any 490xxx quality result:
- exact source seeds 490001..490128
- fixed ascending order
- take the first 24 unique reachable/admitted post-Anger public states
- selection never consults Search/oracle quality

This wraps the already-audited v1 reachable-state generator without changing
its gameplay/reconstruction selection logic.  The retired 480xxx collection
remains immutable and is not read here.
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

import generate_phase1_anger_midturn_fresh_holdout as base

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_anger_midturn_fresh_holdout_v2_seeds_128.txt"
EXPECTED_SOURCE_SEEDS = 128
MIN_SEED = 490001
MAX_SEED = 490128
SAMPLING_CONTRACT = "anger-midturn-fresh-fixed-128-first24-v2"


def load_v2_seeds() -> list[int]:
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


def generate() -> dict:
    # Reuse the already-tested reachable-state mechanics, replacing only the
    # frozen source-seed supplier.  No quality code is imported or consulted.
    base.SEED_FILE = SEED_FILE
    base.EXPECTED_SOURCE_SEEDS = EXPECTED_SOURCE_SEEDS
    base.load_seeds = load_v2_seeds
    suite = base.generate()
    suite["sampling_contract_version"] = SAMPLING_CONTRACT
    suite["selection_rule"] = (
        "first 24 unique reachable admitted Anger post-play public signatures in committed "
        "490001..490128 ascending order; selection uses only public second-turn Anger presence "
        "and reconstruction admission, never Search/oracle quality"
    )
    suite["source_seed_file"] = str(SEED_FILE.relative_to(ROOT))
    suite["source_seed_file_sha256"] = hashlib.sha256(SEED_FILE.read_bytes()).hexdigest()
    suite["source_seed_range"] = [MIN_SEED, MAX_SEED]
    suite["quality_results_observed_during_selection"] = 0
    suite["prior_480xxx_quality_rows_read"] = 0
    suite["prior_480xxx_cases_used"] = 0
    return suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ANGER_MIDTURN_FRESH_HOLDOUT_V2_SUITE_FREEZE = PASS")
    print(f"SAMPLING_CONTRACT = {suite['sampling_contract_version']}")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"PUBLIC_INTENT_COUNTS = {suite['public_intent_counts']}")
    print(f"REJECTED_COUNTS = {suite['rejected_counts']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("PRIOR_480XXX_QUALITY_ROWS_READ = 0")
    print("PRIOR_480XXX_CASES_USED = 0")


if __name__ == "__main__":
    main()
