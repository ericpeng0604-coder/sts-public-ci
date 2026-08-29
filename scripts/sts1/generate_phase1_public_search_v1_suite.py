#!/usr/bin/env python3
"""Freeze the first multi-state public-search V1 quality suite.

Selection is deliberately blind to Search/oracle quality: scan the precommitted
source seeds in file order and keep the first 24 unique opening Jaw Worm public
decision signatures that pass the exact V1 reconstruction admission gate.
Source simulator seed is provenance only and is never placed inside public_state.
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

from roguelike_ai.sts1_teacher.contract import canonical_json
from roguelike_ai.sts1_teacher.public_run_state import simulator_public_run_state
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v1_source_seeds_64.txt"
SUITE_SCHEMA = "sts1-phase1-public-search-v1-quality-suite-v1"
EXPECTED_SOURCE_SEEDS = 64
TARGET_CASES = 24


def load_seeds() -> list[int]:
    seeds = [
        int(line.strip())
        for line in SEED_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != EXPECTED_SOURCE_SEEDS or len(set(seeds)) != EXPECTED_SOURCE_SEEDS:
        raise RuntimeError(f"source_seed_contract:{len(seeds)}:{len(set(seeds))}")
    return seeds


def public_run_state(gc) -> dict:
    run = simulator_public_run_state(gc)
    # Explicit encounter is the first Act-1 monster floor for this frozen V1
    # slice. Character/ascension are constructor inputs, not hidden state.
    run.update(
        {
            "floor": 1,
            "act": 1,
            "character": "IRONCLAD",
            "ascension_level": 0,
            "room": "COMBAT",
            "screen_type": "NONE",
            "screen_choices": [],
            "rewards": [],
            "map_choices": [],
        }
    )
    return run


def suite_digest(records: list[dict]) -> str:
    payload = "\n".join(canonical_json(record) for record in records) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate() -> dict:
    adapter = SimulatorCombatAdapter()
    records: list[dict] = []
    seen_signatures: set[str] = set()
    rejected: list[dict] = []

    for source_seed in load_seeds():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, source_seed, 0)
        gc.floor_num = 1
        gc.cur_room = sts.Room.MONSTER

        bc = sts.BattleContext()
        bc.init_encounter(gc, sts.MonsterEncounter.JAW_WORM)
        native_actions = list(sts.get_legal_actions(bc))
        if not native_actions:
            rejected.append({"source_seed": source_seed, "reason": "no_native_legal_actions"})
            continue

        run = public_run_state(gc)
        projected = adapter.adapt(bc, legal_actions=native_actions, run_state=run)
        admitted_state = attach_reconstruction_capabilities(projected, run_state=run)
        try:
            context = require_public_reconstruction(admitted_state)
        except Exception as exc:
            rejected.append({"source_seed": source_seed, "reason": str(exc)})
            continue

        if admitted_state.get("turn") != 0:
            raise RuntimeError(f"opening_turn_not_zero:{source_seed}:{admitted_state.get('turn')}")
        enemy = admitted_state["enemies"][0]
        if enemy.get("name") != "JAW_WORM" or enemy.get("intent") != "JAW_WORM_CHOMP":
            raise RuntimeError(f"unexpected_opening_jaw_worm:{source_seed}:{enemy}")
        if context.decision_signature in seen_signatures:
            continue

        seen_signatures.add(context.decision_signature)
        records.append(
            {
                "case_id": f"jaw-worm-v1-{len(records):02d}",
                "source_seed_provenance_only": source_seed,
                "decision_signature": context.decision_signature,
                "public_state": admitted_state,
            }
        )
        if len(records) == TARGET_CASES:
            break

    if len(records) != TARGET_CASES:
        raise RuntimeError(
            f"insufficient_unique_admitted_cases:{len(records)}/{TARGET_CASES}:rejected={rejected}"
        )

    digest = suite_digest(records)
    return {
        "schema_version": SUITE_SCHEMA,
        "frozen_before_quality_results": True,
        "selection_rule": "first 24 unique fully V1-admitted opening Jaw Worm public decision signatures in committed source-seed order",
        "source_seed_file": str(SEED_FILE.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "case_count": len(records),
        "suite_digest": digest,
        "cases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PUBLIC_SEARCH_V1_SUITE_FREEZE = PASS")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")


if __name__ == "__main__":
    main()
