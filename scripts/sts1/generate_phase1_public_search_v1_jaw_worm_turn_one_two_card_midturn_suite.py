#!/usr/bin/env python3
"""Freeze a public-search suite after exactly two audited starter-card plays.

Selection is blind to Search/oracle quality. For each precommitted source seed we
reach the audited Jaw Worm second-player-turn boundary, choose the first card
using only that public DecisionContext, project again, choose the second card
using only the new public DecisionContext, and keep the resulting public state.

The source simulator is provenance/state generation only. Formal Search receives
only the admitted public state. This is NOT a general hand-size history rule:
only the already-admitted starter-only Strike/Defend/Bash slice is eligible.
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

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v1_jaw_worm_turn_one_two_card_midturn_source_seeds_64.txt"
SUITE_SCHEMA = "sts1-phase1-public-search-v1-jaw-worm-turn-one-two-card-midturn-suite-v1"
EXPECTED_SOURCE_SEEDS = 64
TARGET_CASES = 24
STARTER_IDS = frozenset({"STRIKE_RED", "DEFEND_RED", "BASH"})


def load_seeds() -> list[int]:
    seeds = [
        int(line.strip())
        for line in SEED_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != EXPECTED_SOURCE_SEEDS or len(set(seeds)) != EXPECTED_SOURCE_SEEDS:
        raise RuntimeError(f"source_seed_contract:{len(seeds)}:{len(set(seeds))}")
    if min(seeds) != 450001 or max(seeds) != 450064:
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


def _choose_public_card(context: DecisionContext, *, ordinal: int):
    cards = sorted(
        (action for action in context.legal_actions if action.payload.get("kind") == "play_card"),
        key=lambda action: action.action_id,
    )
    if not cards:
        raise RuntimeError(f"turn_one_no_public_card_action:{ordinal}")
    # Only the current public decision signature and the fixed play ordinal are
    # used. No source RNG/history and no Search/oracle score participates.
    selector = hashlib.sha256(
        f"two-card-generation-v1:{ordinal}:{context.decision_signature}".encode("utf-8")
    ).hexdigest()
    return cards[int(selector[:16], 16) % len(cards)]


def _execute_public_card_on_source(bc, public_action) -> str:
    hand_index = int(public_action.payload["hand_index"])
    target = public_action.payload.get("target_index")
    source_idx = hand_index - 1
    hand = list(bc.hand)
    chosen_card = str(hand[source_idx].name).upper()
    if chosen_card not in {"STRIKE", "DEFEND", "BASH"}:
        raise RuntimeError(f"generation_nonstarter_card:{chosen_card}")
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


def _admitted_context(adapter, bc, run: dict) -> tuple[dict, DecisionContext]:
    projected = _project(adapter, bc, run)
    admitted_state = attach_reconstruction_capabilities(projected, run_state=run)
    return admitted_state, require_public_reconstruction(admitted_state)


def _assert_two_card_midturn(state: dict, *, source_seed: int) -> str:
    if state.get("turn") != 1:
        raise RuntimeError(f"two_card_midturn_not_turn_one:{source_seed}:{state.get('turn')}")
    piles = {
        "hand": len(state.get("hand", [])),
        "draw": len(state.get("draw_pile", [])),
        "discard": len(state.get("discard_pile", [])),
        "exhaust": len(state.get("exhaust_pile", [])),
    }
    if piles != {"hand": 3, "draw": 0, "discard": 7, "exhaust": 0}:
        raise RuntimeError(f"two_card_midturn_pile_mismatch:{source_seed}:{piles}")
    if state.get("energy") not in {0, 1}:
        raise RuntimeError(f"two_card_midturn_energy_mismatch:{source_seed}:{state.get('energy')}")
    if state.get("block") not in {0, 5, 10}:
        raise RuntimeError(f"two_card_midturn_block_mismatch:{source_seed}:{state.get('block')}")
    if state.get("powers") != []:
        raise RuntimeError(f"two_card_midturn_player_powers_not_empty:{source_seed}:{state.get('powers')}")
    all_cards = [
        *state.get("hand", []),
        *state.get("draw_pile", []),
        *state.get("discard_pile", []),
        *state.get("exhaust_pile", []),
    ]
    ids = {str(card.get("id", "")).upper() for card in all_cards if isinstance(card, dict)}
    if not ids or not ids.issubset(STARTER_IDS):
        raise RuntimeError(f"two_card_midturn_nonstarter_surface:{source_seed}:{sorted(ids)}")
    enemies = state.get("enemies", [])
    if len(enemies) != 1 or enemies[0].get("name") != "JAW_WORM":
        raise RuntimeError(f"two_card_midturn_enemy_mismatch:{source_seed}:{enemies}")
    intent = str(enemies[0].get("intent", ""))
    if not intent.startswith("JAW_WORM_"):
        raise RuntimeError(f"two_card_midturn_public_intent_missing:{source_seed}:{intent}")
    return intent


def generate() -> dict:
    adapter = SimulatorCombatAdapter()
    records: list[dict] = []
    seen_signatures: set[str] = set()
    rejected: list[dict] = []
    intent_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()

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
        try:
            _, boundary_context = _admitted_context(adapter, bc, run)
            first_action = _choose_public_card(boundary_context, ordinal=1)
            first_card = _execute_public_card_on_source(bc, first_action)

            _, after_first_context = _admitted_context(adapter, bc, run)
            second_action = _choose_public_card(after_first_context, ordinal=2)
            second_card = _execute_public_card_on_source(bc, second_action)

            projected = _project(adapter, bc, run)
            intent = _assert_two_card_midturn(projected, source_seed=source_seed)
            admitted_state = attach_reconstruction_capabilities(projected, run_state=run)
            context = require_public_reconstruction(admitted_state)
        except Exception as exc:
            rejected.append({"source_seed": source_seed, "reason": str(exc)})
            continue

        if context.decision_signature in seen_signatures:
            continue
        seen_signatures.add(context.decision_signature)
        intent_counts[intent] += 1
        sequence = f"{first_card}->{second_card}"
        sequence_counts[sequence] += 1
        records.append(
            {
                "case_id": f"jaw-worm-turn-one-two-card-midturn-v1-{len(records):02d}",
                "source_seed_provenance_only": source_seed,
                "generation_rule": (
                    "two sequential starter cards selected only from each current public decision signature"
                ),
                "generation_first_public_action_id": first_action.action_id,
                "generation_second_public_action_id": second_action.action_id,
                "generation_first_card": first_card,
                "generation_second_card": second_card,
                "generation_sequence": sequence,
                "decision_signature": context.decision_signature,
                "public_state": admitted_state,
            }
        )
        if len(records) == TARGET_CASES:
            break

    if len(records) != TARGET_CASES:
        raise RuntimeError(
            f"insufficient_unique_admitted_two_card_midturn_cases:{len(records)}/{TARGET_CASES}:rejected={rejected}"
        )

    return {
        "schema_version": SUITE_SCHEMA,
        "frozen_before_quality_results": True,
        "selection_rule": (
            "first 24 unique fully admitted Jaw Worm turn-1 public signatures after exactly two "
            "starter-card plays selected solely from successive public decision signatures, in "
            "committed 450001..450064 source-seed order"
        ),
        "source_seed_file": str(SEED_FILE.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "quality_results_observed_during_selection": 0,
        "source_move_history_exported": 0,
        "source_counter_history_exported": 0,
        "hand_size_shortcut_generalized": False,
        "nonstarter_cards_admitted": 0,
        "case_count": len(records),
        "public_intent_counts": dict(sorted(intent_counts.items())),
        "generation_sequence_counts": dict(sorted(sequence_counts.items())),
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
    print("PUBLIC_SEARCH_V1_JAW_WORM_TURN_ONE_TWO_CARD_MIDTURN_SUITE_FREEZE = PASS")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"PUBLIC_INTENT_COUNTS = {suite['public_intent_counts']}")
    print(f"GENERATION_SEQUENCE_COUNTS = {suite['generation_sequence_counts']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("SOURCE_MOVE_HISTORY_EXPORTED = 0")
    print("SOURCE_COUNTER_HISTORY_EXPORTED = 0")
    print("MIDTURN_HAND_SIZE_SHORTCUT_GENERALIZED = 0")
    print("NONSTARTER_CARDS_ADMITTED = 0")


if __name__ == "__main__":
    main()
