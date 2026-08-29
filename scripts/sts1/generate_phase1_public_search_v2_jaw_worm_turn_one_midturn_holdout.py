#!/usr/bin/env python3
"""Freeze fresh 24-case holdout for the preselected midturn V2 policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
BUILD = ROOT / "external" / "sts_lightspeed" / "source" / "build-probe"
for path in (SRC, BUILD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import slaythespire as sts

from roguelike_ai.sts1_teacher.contract import DecisionContext, canonical_json
from roguelike_ai.sts1_teacher.public_run_state import simulator_public_run_state
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v2_jaw_worm_turn_one_midturn_holdout_seeds_64.txt"
RETIRED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v1_jaw_worm_turn_one_midturn_source_seeds_64.txt"
DISCOVERY_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v2_jaw_worm_turn_one_midturn_discovery_seeds_64.txt"
SUITE_SCHEMA = "sts1-phase1-public-search-v2-jaw-worm-turn-one-midturn-holdout-v1"
TARGET_CASES = 24


def load(path: Path) -> list[int]:
    return [int(x.strip()) for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]


def seeds() -> list[int]:
    values = load(SEED_FILE)
    if len(values) != 64 or len(set(values)) != 64 or min(values) != 440001 or max(values) != 440064:
        raise RuntimeError("midturn_v2_holdout_seed_contract_failed")
    forbidden = set(load(RETIRED_FILE)) | set(load(DISCOVERY_FILE))
    if forbidden.intersection(values):
        raise RuntimeError("midturn_v2_holdout_seed_overlap")
    return values


def run_state(gc) -> dict:
    run = simulator_public_run_state(gc)
    run.update({
        "floor": 1, "act": 1, "character": "IRONCLAD", "ascension_level": 0,
        "room": "COMBAT", "screen_type": "NONE", "screen_choices": [],
        "rewards": [], "map_choices": [],
    })
    return run


def enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name if isinstance(name, str) else value).rsplit(".", 1)[-1].upper()


def end_first_turn(bc) -> None:
    action = next((a for a in sts.get_legal_actions(bc) if enum_name(a.action_type) == "END_TURN"), None)
    if action is None:
        raise RuntimeError("midturn_v2_holdout_end_turn_missing")
    action.execute(bc)


def project(adapter, bc, run: dict) -> dict:
    return adapter.adapt(bc, legal_actions=list(sts.get_legal_actions(bc)), run_state=run)


def choose_public_card(context: DecisionContext):
    cards = sorted((a for a in context.legal_actions if a.payload.get("kind") == "play_card"), key=lambda a: a.action_id)
    if not cards:
        raise RuntimeError("midturn_v2_holdout_no_generation_card")
    return cards[int(context.decision_signature[:16], 16) % len(cards)]


def execute_public_card(bc, action) -> str:
    source_idx = int(action.payload["hand_index"]) - 1
    target = action.payload.get("target_index")
    name = str(list(bc.hand)[source_idx].name).upper()
    for native in sts.get_legal_actions(bc):
        if enum_name(native.action_type) != "CARD" or int(native.source_idx) != source_idx:
            continue
        if target is not None and int(native.target_idx) != int(target):
            continue
        native.execute(bc)
        return name
    raise RuntimeError("midturn_v2_holdout_native_generation_action_missing")


def digest(records: list[dict]) -> str:
    payload = "\n".join(canonical_json(row) for row in records) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def generate() -> dict:
    adapter = SimulatorCombatAdapter()
    records: list[dict] = []
    seen: set[str] = set()
    intent_counts: Counter[str] = Counter()
    played_counts: Counter[str] = Counter()
    for seed in seeds():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
        gc.floor_num = 1
        gc.cur_room = sts.Room.MONSTER
        bc = sts.BattleContext()
        bc.init_encounter(gc, sts.MonsterEncounter.JAW_WORM)
        end_first_turn(bc)
        run = run_state(gc)
        boundary = attach_reconstruction_capabilities(project(adapter, bc, run), run_state=run)
        boundary_context = require_public_reconstruction(boundary)
        generation_action = choose_public_card(boundary_context)
        played = execute_public_card(bc, generation_action)
        state = attach_reconstruction_capabilities(project(adapter, bc, run), run_state=run)
        context = require_public_reconstruction(state)
        if context.state.get("turn") != 1 or len(context.state.get("hand", [])) != 4:
            raise RuntimeError(f"midturn_v2_holdout_scope_drift:{seed}")
        if context.state["enemies"][0].get("name") != "JAW_WORM":
            raise RuntimeError(f"midturn_v2_holdout_enemy_drift:{seed}")
        if context.decision_signature in seen:
            continue
        seen.add(context.decision_signature)
        intent = str(context.state["enemies"][0].get("intent", "UNKNOWN"))
        intent_counts[intent] += 1
        played_counts[played] += 1
        records.append({
            "case_id": f"jaw-midturn-v2-holdout-{len(records):02d}",
            "source_seed_provenance_only": seed,
            "generation_rule": "one card selected from fresh-boundary public decision signature only",
            "generation_played_card_diagnostic_only": played,
            "decision_signature": context.decision_signature,
            "public_state": state,
        })
        if len(records) == TARGET_CASES:
            break
    if len(records) != TARGET_CASES:
        raise RuntimeError(f"midturn_v2_holdout_insufficient_cases:{len(records)}")
    return {
        "schema_version": SUITE_SCHEMA,
        "frozen_before_holdout_results": True,
        "preselected_policy_id": "jaw-worm-turn1-midturn-margin1-else-simple-v1",
        "preselected_margin": 1.0,
        "quality_results_observed_during_selection": 0,
        "retired_420xxx_rows_read": 0,
        "discovery_430xxx_rows_read_during_holdout_selection": 0,
        "source_move_history_exported": 0,
        "source_counter_history_exported": 0,
        "source_seed_file": str(SEED_FILE.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "case_count": len(records),
        "public_intent_counts": dict(sorted(intent_counts.items())),
        "generation_played_card_counts": dict(sorted(played_counts.items())),
        "suite_digest": digest(records),
        "cases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    print("MIDTURN_V2_FRESH_HOLDOUT_FREEZE = PASS")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"PUBLIC_INTENT_COUNTS = {suite['public_intent_counts']}")
    print(f"GENERATION_PLAYED_CARD_COUNTS = {suite['generation_played_card_counts']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("HOLDOUT_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("RETIRED_420XXX_ROWS_READ = 0")
    print("DISCOVERY_430XXX_ROWS_READ_DURING_SELECTION = 0")


if __name__ == "__main__":
    main()
