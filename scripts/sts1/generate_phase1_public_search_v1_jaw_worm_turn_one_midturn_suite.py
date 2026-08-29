#!/usr/bin/env python3
"""Freeze a public-search quality suite for Jaw Worm turn-1 midturn states.

Selection is blind to Search/oracle quality.  For each precommitted source seed
we reach the already-audited second-player-turn start boundary, build its public
DecisionContext, choose exactly one card using only that public decision
signature, execute the corresponding source action for state generation, and
then admit the resulting public midturn observation.

The source seed and source BattleContext are provenance/generation tools only;
formal Teacher/Search receives only the reconstructed public state.
"""
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

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v1_jaw_worm_turn_one_midturn_source_seeds_64.txt"
SUITE_SCHEMA = "sts1-phase1-public-search-v1-jaw-worm-turn-one-midturn-suite-v1"
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
    if min(seeds) != 420001 or max(seeds) != 420064:
        raise RuntimeError(f"unexpected_source_seed_range:{min(seeds)}:{max(seeds)}")
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


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name if isinstance(name, str) else value).rsplit(".", 1)[-1].upper()


def _end_first_player_turn_without_cards(bc) -> None:
    end_turn = next(
        (action for action in sts.get_legal_actions(bc) if _enum_name(action.action_type) == "END_TURN"),
        None,
    )
    if end_turn is None:
        raise RuntimeError("opening_end_turn_missing")
    end_turn.execute(bc)


def _project(adapter, bc, run: dict) -> dict:
    return adapter.adapt(bc, legal_actions=list(sts.get_legal_actions(bc)), run_state=run)


def _choose_one_public_card(context: DecisionContext):
    cards = sorted(
        (action for action in context.legal_actions if action.payload.get("kind") == "play_card"),
        key=lambda action: action.action_id,
    )
    if not cards:
        raise RuntimeError("turn_one_no_public_card_action")
    index = int(context.decision_signature[:16], 16) % len(cards)
    return cards[index]


def _execute_public_card_on_source(bc, public_action) -> str:
    hand_index = int(public_action.payload["hand_index"])
    target = public_action.payload.get("target_index")
    source_idx = hand_index - 1
    hand = list(bc.hand)
    chosen_card = str(hand[source_idx].name).upper()
    for native in sts.get_legal_actions(bc):
        if _enum_name(native.action_type) != "CARD":
            continue
        if int(native.source_idx) != source_idx:
            continue
        if target is not None and int(native.target_idx) != int(target):
            continue
        native.execute(bc)
        return chosen_card
    raise RuntimeError(f"public_native_generation_action_mismatch:{public_action.action_id}")


def _assert_midturn(state: dict, *, source_seed: int) -> str:
    if state.get("turn") != 1:
        raise RuntimeError(f"midturn_not_turn_one:{source_seed}:{state.get('turn')}")
    piles = {
        "hand": len(state.get("hand", [])),
        "draw": len(state.get("draw_pile", [])),
        "discard": len(state.get("discard_pile", [])),
        "exhaust": len(state.get("exhaust_pile", [])),
    }
    if piles != {"hand": 4, "draw": 0, "discard": 6, "exhaust": 0}:
        raise RuntimeError(f"midturn_one_card_pile_mismatch:{source_seed}:{piles}")
    if state.get("energy") not in {1, 2}:
        raise RuntimeError(f"midturn_energy_mismatch:{source_seed}:{state.get('energy')}")
    if state.get("block") not in {0, 5}:
        raise RuntimeError(f"midturn_block_mismatch:{source_seed}:{state.get('block')}")
    if state.get("powers") != []:
        raise RuntimeError(f"midturn_player_powers_not_empty:{source_seed}:{state.get('powers')}")
    enemies = state.get("enemies", [])
    if len(enemies) != 1 or enemies[0].get("name") != "JAW_WORM":
        raise RuntimeError(f"midturn_enemy_mismatch:{source_seed}:{enemies}")
    intent = str(enemies[0].get("intent", ""))
    if not intent.startswith("JAW_WORM_"):
        raise RuntimeError(f"midturn_public_intent_missing:{source_seed}:{intent}")
    return intent


def generate() -> dict:
    adapter = SimulatorCombatAdapter()
    records: list[dict] = []
    seen_signatures: set[str] = set()
    rejected: list[dict] = []
    intent_counts: Counter[str] = Counter()
    first_card_counts: Counter[str] = Counter()

    for source_seed in load_seeds():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, source_seed, 0)
        gc.floor_num = 1
        gc.cur_room = sts.Room.MONSTER
        bc = sts.BattleContext()
        bc.init_encounter(gc, sts.MonsterEncounter.JAW_WORM)
        if int(bc.turn) != 0:
            rejected.append({"source_seed": source_seed, "reason": f"opening_turn:{bc.turn}"})
            continue

        _end_first_player_turn_without_cards(bc)
        run = public_run_state(gc)
        boundary = _project(adapter, bc, run)
        admitted_boundary = attach_reconstruction_capabilities(boundary, run_state=run)
        try:
            boundary_context = require_public_reconstruction(admitted_boundary)
            generation_action = _choose_one_public_card(boundary_context)
            played_card = _execute_public_card_on_source(bc, generation_action)
            projected = _project(adapter, bc, run)
            intent = _assert_midturn(projected, source_seed=source_seed)
            admitted_state = attach_reconstruction_capabilities(projected, run_state=run)
            context = require_public_reconstruction(admitted_state)
        except Exception as exc:
            rejected.append({"source_seed": source_seed, "reason": str(exc)})
            continue

        if context.decision_signature in seen_signatures:
            continue
        seen_signatures.add(context.decision_signature)
        intent_counts[intent] += 1
        first_card_counts[played_card] += 1
        records.append(
            {
                "case_id": f"jaw-worm-turn-one-midturn-v1-{len(records):02d}",
                "source_seed_provenance_only": source_seed,
                "generation_rule": "one card selected from public boundary decision signature only",
                "generation_public_action_id": generation_action.action_id,
                "generation_played_card": played_card,
                "decision_signature": context.decision_signature,
                "public_state": admitted_state,
            }
        )
        if len(records) == TARGET_CASES:
            break

    if len(records) != TARGET_CASES:
        raise RuntimeError(
            f"insufficient_unique_admitted_midturn_cases:{len(records)}/{TARGET_CASES}:rejected={rejected}"
        )

    return {
        "schema_version": SUITE_SCHEMA,
        "frozen_before_quality_results": True,
        "selection_rule": (
            "first 24 unique fully admitted Jaw Worm turn-1 public signatures after one card "
            "chosen solely from the fresh-boundary public decision signature, in committed "
            "420001..420064 source-seed order"
        ),
        "source_seed_file": str(SEED_FILE.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "quality_results_observed_during_selection": 0,
        "source_move_history_exported": 0,
        "source_counter_history_exported": 0,
        "case_count": len(records),
        "public_intent_counts": dict(sorted(intent_counts.items())),
        "generation_played_card_counts": dict(sorted(first_card_counts.items())),
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
    print("PUBLIC_SEARCH_V1_JAW_WORM_TURN_ONE_MIDTURN_SUITE_FREEZE = PASS")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"PUBLIC_INTENT_COUNTS = {suite['public_intent_counts']}")
    print(f"GENERATION_PLAYED_CARD_COUNTS = {suite['generation_played_card_counts']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("SOURCE_MOVE_HISTORY_EXPORTED = 0")
    print("SOURCE_COUNTER_HISTORY_EXPORTED = 0")


if __name__ == "__main__":
    main()
