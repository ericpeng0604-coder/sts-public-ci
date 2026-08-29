#!/usr/bin/env python3
"""Freeze a public-search quality suite at the audited Jaw Worm turn-1 boundary.

Selection is blind to Search/oracle quality. For each precommitted source seed we
create the opening Jaw Worm encounter only to generate a public observation,
end the first player turn without playing a card, then project the resulting
second-player-turn state through the public adapter. Hidden source seed/history
is provenance only and never enters the formal Teacher state.
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

from roguelike_ai.sts1_teacher.contract import canonical_json
from roguelike_ai.sts1_teacher.public_run_state import simulator_public_run_state
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v1_jaw_worm_turn_one_source_seeds_64.txt"
SUITE_SCHEMA = "sts1-phase1-public-search-v1-jaw-worm-turn-one-suite-v1"
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
    if min(seeds) != 410001 or max(seeds) != 410064:
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


def _end_first_player_turn_without_cards(bc) -> None:
    legal = list(sts.get_legal_actions(bc))
    end_turn = next(
        (action for action in legal if action.action_type == sts.SearchActionType.END_TURN),
        None,
    )
    if end_turn is None:
        raise RuntimeError("opening_end_turn_missing")
    end_turn.execute(bc)


def _assert_public_turn_one_boundary(state: dict, *, source_seed: int) -> str:
    if state.get("turn") != 1:
        raise RuntimeError(f"turn_one_not_reached:{source_seed}:{state.get('turn')}")
    if state.get("energy") != 3 or state.get("block") != 0:
        raise RuntimeError(
            f"turn_one_player_boundary_mismatch:{source_seed}:"
            f"energy={state.get('energy')}:block={state.get('block')}"
        )
    piles = {
        "hand": len(state.get("hand", [])),
        "draw": len(state.get("draw_pile", [])),
        "discard": len(state.get("discard_pile", [])),
        "exhaust": len(state.get("exhaust_pile", [])),
    }
    if piles != {"hand": 5, "draw": 0, "discard": 5, "exhaust": 0}:
        raise RuntimeError(f"turn_one_pile_boundary_mismatch:{source_seed}:{piles}")
    if state.get("powers") != []:
        raise RuntimeError(f"turn_one_player_powers_not_empty:{source_seed}:{state.get('powers')}")

    enemies = state.get("enemies", [])
    if len(enemies) != 1 or enemies[0].get("name") != "JAW_WORM":
        raise RuntimeError(f"turn_one_enemy_mismatch:{source_seed}:{enemies}")
    intent = str(enemies[0].get("intent", ""))
    if not intent.startswith("JAW_WORM_"):
        raise RuntimeError(f"turn_one_public_intent_missing:{source_seed}:{intent}")
    return intent


def generate() -> dict:
    adapter = SimulatorCombatAdapter()
    records: list[dict] = []
    seen_signatures: set[str] = set()
    rejected: list[dict] = []
    intent_counts: Counter[str] = Counter()

    for source_seed in load_seeds():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, source_seed, 0)
        gc.floor_num = 1
        gc.cur_room = sts.Room.MONSTER

        # SOURCE-STATE GENERATION ONLY. Formal Search never receives this source
        # BattleContext and never copies its RNG or hidden move history.
        bc = sts.BattleContext()
        bc.init_encounter(gc, sts.MonsterEncounter.JAW_WORM)
        if int(bc.turn) != 0:
            rejected.append({"source_seed": source_seed, "reason": f"opening_turn:{bc.turn}"})
            continue

        _end_first_player_turn_without_cards(bc)
        native_actions = list(sts.get_legal_actions(bc))
        if not native_actions:
            rejected.append({"source_seed": source_seed, "reason": "turn_one_no_native_legal_actions"})
            continue

        run = public_run_state(gc)
        projected = adapter.adapt(bc, legal_actions=native_actions, run_state=run)
        try:
            intent = _assert_public_turn_one_boundary(projected, source_seed=source_seed)
        except Exception as exc:
            rejected.append({"source_seed": source_seed, "reason": str(exc)})
            continue

        admitted_state = attach_reconstruction_capabilities(projected, run_state=run)
        try:
            context = require_public_reconstruction(admitted_state)
        except Exception as exc:
            rejected.append({"source_seed": source_seed, "reason": str(exc)})
            continue

        if context.decision_signature in seen_signatures:
            continue

        seen_signatures.add(context.decision_signature)
        intent_counts[intent] += 1
        records.append(
            {
                "case_id": f"jaw-worm-turn-one-v1-{len(records):02d}",
                "source_seed_provenance_only": source_seed,
                "decision_signature": context.decision_signature,
                "public_state": admitted_state,
            }
        )
        if len(records) == TARGET_CASES:
            break

    if len(records) != TARGET_CASES:
        raise RuntimeError(
            f"insufficient_unique_admitted_turn_one_cases:{len(records)}/{TARGET_CASES}:"
            f"rejected={rejected}"
        )

    digest = suite_digest(records)
    return {
        "schema_version": SUITE_SCHEMA,
        "frozen_before_quality_results": True,
        "selection_rule": (
            "first 24 unique fully admitted public decision signatures after ending "
            "Jaw Worm turn 0 without playing any card, in committed 410001..410064 seed order"
        ),
        "source_seed_file": str(SEED_FILE.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "quality_results_observed_during_selection": 0,
        "case_count": len(records),
        "public_intent_counts": dict(sorted(intent_counts.items())),
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
    print("PUBLIC_SEARCH_V1_JAW_WORM_TURN_ONE_SUITE_FREEZE = PASS")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"PUBLIC_INTENT_COUNTS = {suite['public_intent_counts']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("SOURCE_MOVE_HISTORY_EXPORTED = 0")


if __name__ == "__main__":
    main()
