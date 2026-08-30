#!/usr/bin/env python3
"""Freeze disjoint Looter turn-0 discovery/fresh-holdout suites.

Selection is blind to Search/oracle quality. For each precommitted source-seed
file, scan seeds in file order and retain the first 24 unique fully admitted
Looter turn-0 public decision signatures. Source seeds are provenance only.
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

TARGET_CASES = 24
SUITES = {
    "discovery": {
        "seed_file": ROOT / "tests" / "data" / "sts1_phase1_looter_discovery_seeds_64.txt",
        "min_seed": 590001,
        "max_seed": 590064,
        "schema": "sts1-phase1-looter-turn0-discovery-suite-v1",
    },
    "holdout": {
        "seed_file": ROOT / "tests" / "data" / "sts1_phase1_looter_fresh_holdout_seeds_64.txt",
        "min_seed": 600001,
        "max_seed": 600064,
        "schema": "sts1-phase1-looter-turn0-fresh-holdout-suite-v1",
    },
}


def load_seeds(spec: dict) -> list[int]:
    path = spec["seed_file"]
    seeds = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != 64 or len(set(seeds)) != 64:
        raise RuntimeError(f"looter_source_seed_contract:{len(seeds)}:{len(set(seeds))}")
    if min(seeds) != spec["min_seed"] or max(seeds) != spec["max_seed"]:
        raise RuntimeError("looter_source_seed_range_drift")
    return seeds


def public_run_state(gc) -> dict:
    run = simulator_public_run_state(gc)
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


def generate(kind: str) -> dict:
    spec = SUITES[kind]
    adapter = SimulatorCombatAdapter()
    records: list[dict] = []
    seen_signatures: set[str] = set()
    rejected: list[dict] = []

    for source_seed in load_seeds(spec):
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, source_seed, 0)
        gc.floor_num = 1
        gc.cur_room = sts.Room.MONSTER
        bc = sts.BattleContext()
        bc.init_encounter(gc, sts.MonsterEncounter.LOOTER)
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
            raise RuntimeError(f"looter_opening_turn_not_zero:{source_seed}")
        enemy = admitted_state["enemies"][0]
        if enemy.get("name") != "LOOTER" or enemy.get("intent") != "LOOTER_MUG":
            raise RuntimeError(f"unexpected_opening_looter:{source_seed}:{enemy}")
        if context.decision_signature in seen_signatures:
            continue

        seen_signatures.add(context.decision_signature)
        records.append(
            {
                "case_id": f"looter-{kind}-v1-{len(records):02d}",
                "source_seed_provenance_only": source_seed,
                "decision_signature": context.decision_signature,
                "public_state": admitted_state,
            }
        )
        if len(records) == TARGET_CASES:
            break

    if len(records) != TARGET_CASES:
        raise RuntimeError(
            f"insufficient_unique_admitted_looter_cases:{kind}:{len(records)}/{TARGET_CASES}:rejected={rejected}"
        )

    path = spec["seed_file"]
    return {
        "schema_version": spec["schema"],
        "suite_kind": kind,
        "frozen_before_quality_results": True,
        "selection_rule": f"first 24 unique fully admitted Looter turn-0 public decision signatures in committed {spec['min_seed']}..{spec['max_seed']} source-seed order",
        "source_seed_file": str(path.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "case_count": len(records),
        "suite_digest": suite_digest(records),
        "cases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-kind", choices=sorted(SUITES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = generate(args.suite_kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"LOOTER_{args.suite_kind.upper()}_SUITE_FREEZE = PASS")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")


if __name__ == "__main__":
    main()
