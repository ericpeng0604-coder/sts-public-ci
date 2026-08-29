#!/usr/bin/env python3
"""Freeze reachable Anger-rich Jaw Worm midturn states from fresh 480xxx seeds.

Selection is blind to Search/oracle quality.  Each source run genuinely adds
one normal Anger to the Ironclad deck before combat, reaches the audited second
player turn, accepts the seed only when Anger is publicly present in that hand,
plays that public Anger action on the source solely to generate a reachable
post-action state, then stores only the public projection plus bounded
reconstruction auxiliary counters.

The source BattleContext and source RNG are generation/provenance only and are
never stored in formal Teacher state or used by the candidate policy.
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

import slaythespire as sts

from roguelike_ai.sts1_teacher.contract import DecisionContext, canonical_json
from roguelike_ai.sts1_teacher.public_run_state import simulator_public_run_state
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_anger_midturn_fresh_holdout_seeds_64.txt"
SUITE_SCHEMA = "sts1-phase1-anger-midturn-fresh-holdout-suite-v1"
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
    if min(seeds) != 480001 or max(seeds) != 480064:
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


def aux() -> dict:
    return {
        "schema_version": "sts1-public-reconstruction-aux-v1",
        "source": "communicationmod_command_trace_v1",
        "turn": 1,
        "complete": True,
        "cards_played_this_turn": 1,
        "attacks_played_this_turn": 1,
        "skills_played_this_turn": 0,
        "cards_discarded_this_turn": 0,
    }


def suite_digest(records: list[dict]) -> str:
    payload = "\n".join(canonical_json(record) for record in records) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name if isinstance(name, str) else value).rsplit(".", 1)[-1].upper()


def _end_first_player_turn_without_cards(bc) -> None:
    action = next(
        (a for a in sts.get_legal_actions(bc) if _enum_name(a.action_type) == "END_TURN"),
        None,
    )
    if action is None:
        raise RuntimeError("opening_end_turn_missing")
    action.execute(bc)


def _project(adapter, bc, run: dict) -> dict:
    return adapter.adapt(bc, legal_actions=list(sts.get_legal_actions(bc)), run_state=run)


def _public_anger_action(context: DecisionContext):
    hand = context.state.get("hand", [])
    anger_positions = [
        int(card.get("position", index + 1))
        for index, card in enumerate(hand)
        if str(card.get("id", "")).strip().upper().replace(" ", "_").replace("-", "_") == "ANGER"
    ]
    if len(anger_positions) != 1:
        raise RuntimeError(f"boundary_anger_hand_count:{len(anger_positions)}")
    position = anger_positions[0]
    candidates = [
        action
        for action in context.legal_actions
        if action.payload.get("kind") == "play_card"
        and int(action.payload.get("hand_index", -1)) == position
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"boundary_anger_public_action_count:{len(candidates)}")
    return candidates[0]


def _execute_public_card_on_source(bc, public_action) -> None:
    source_idx = int(public_action.payload["hand_index"]) - 1
    target = public_action.payload.get("target_index")
    for native in sts.get_legal_actions(bc):
        if _enum_name(native.action_type) != "CARD":
            continue
        if int(native.source_idx) != source_idx:
            continue
        if target is not None and int(native.target_idx) != int(target):
            continue
        native.execute(bc)
        return
    raise RuntimeError(f"public_native_generation_action_mismatch:{public_action.action_id}")


def _card_id(card: dict) -> str:
    value = str(card.get("id", "")).strip().upper().replace(" ", "_").replace("-", "_")
    return {"STRIKE_R": "STRIKE_RED", "DEFEND_R": "DEFEND_RED"}.get(value, value)


def _assert_post_anger(state: dict, *, source_seed: int) -> str:
    if state.get("turn") != 1:
        raise RuntimeError(f"anger_midturn_not_turn_one:{source_seed}:{state.get('turn')}")
    shape = {
        "hand": len(state.get("hand", [])),
        "draw": len(state.get("draw_pile", [])),
        "discard": len(state.get("discard_pile", [])),
        "exhaust": len(state.get("exhaust_pile", [])),
    }
    if shape != {"hand": 4, "draw": 1, "discard": 7, "exhaust": 0}:
        raise RuntimeError(f"anger_midturn_pile_shape:{source_seed}:{shape}")
    if state.get("energy") != 3 or state.get("block") != 0:
        raise RuntimeError(
            f"anger_midturn_player_shape:{source_seed}:energy={state.get('energy')}:block={state.get('block')}"
        )
    if state.get("powers") != []:
        raise RuntimeError(f"anger_midturn_player_powers:{source_seed}:{state.get('powers')}")

    piles = [
        state.get("hand", []),
        state.get("draw_pile", []),
        state.get("discard_pile", []),
        state.get("exhaust_pile", []),
    ]
    ids = [_card_id(card) for pile in piles for card in pile]
    expected = Counter({"STRIKE_RED": 5, "DEFEND_RED": 4, "BASH": 1, "ANGER": 2})
    if Counter(ids) != expected:
        raise RuntimeError(f"anger_midturn_card_composition:{source_seed}:{Counter(ids)}")
    discard_angers = sum(_card_id(card) == "ANGER" for card in state.get("discard_pile", []))
    other_angers = sum(
        _card_id(card) == "ANGER"
        for pile_name in ("hand", "draw_pile", "exhaust_pile")
        for card in state.get(pile_name, [])
    )
    if discard_angers != 2 or other_angers != 0:
        raise RuntimeError(
            f"anger_generated_copy_not_publicly_visible:{source_seed}:discard={discard_angers}:other={other_angers}"
        )
    enemies = state.get("enemies", [])
    if len(enemies) != 1 or enemies[0].get("name") != "JAW_WORM":
        raise RuntimeError(f"anger_midturn_enemy:{source_seed}:{enemies}")
    intent = str(enemies[0].get("intent", ""))
    if not intent.startswith("JAW_WORM_"):
        raise RuntimeError(f"anger_midturn_public_intent:{source_seed}:{intent}")
    return intent


def generate() -> dict:
    adapter = SimulatorCombatAdapter()
    records: list[dict] = []
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    intents: Counter[str] = Counter()

    for source_seed in load_seeds():
        try:
            gc = sts.GameContext(sts.CharacterClass.IRONCLAD, source_seed, 0)
            gc.obtain_card(sts.Card(sts.CardId.ANGER))
            deck = list(gc.deck)
            if len(deck) != 11:
                raise RuntimeError(f"deck_size_after_anger:{len(deck)}")
            gc.floor_num = 1
            gc.cur_room = sts.Room.MONSTER
            bc = sts.BattleContext()
            bc.init_encounter(gc, sts.MonsterEncounter.JAW_WORM)
            if int(bc.turn) != 0:
                raise RuntimeError(f"opening_turn:{bc.turn}")

            _end_first_player_turn_without_cards(bc)
            run = public_run_state(gc)
            boundary = _project(adapter, bc, run)
            # This is a source-generation boundary, not a formal rollout root.
            # Select Anger strictly from public observation/legal actions; the
            # post-Anger state must pass the formal reconstruction gate below.
            boundary_context = DecisionContext.from_public_state(boundary)
            anger_action = _public_anger_action(boundary_context)
            _execute_public_card_on_source(bc, anger_action)

            projected = _project(adapter, bc, run)
            intent = _assert_post_anger(projected, source_seed=source_seed)
            admitted_state = attach_reconstruction_capabilities(
                projected,
                run_state=run,
                reconstruction_aux=aux(),
            )
            context = require_public_reconstruction(admitted_state)
            if context.decision_signature in seen:
                rejected["duplicate_signature"] += 1
                continue
        except Exception as exc:
            reason = str(exc).split(":", 1)[0] or exc.__class__.__name__
            rejected[reason] += 1
            continue

        seen.add(context.decision_signature)
        intents[intent] += 1
        records.append(
            {
                "case_id": f"anger-midturn-fresh-v1-{len(records):02d}",
                "source_seed_provenance_only": source_seed,
                "generation_rule": (
                    "add one normal Anger to the source deck before combat; end opening player turn without cards; "
                    "accept only when the second-turn public hand contains exactly one Anger; play that public Anger action"
                ),
                "generation_public_action_id": anger_action.action_id,
                "decision_signature": context.decision_signature,
                "public_state": admitted_state,
                "reconstruction_aux": aux(),
            }
        )
        if len(records) == TARGET_CASES:
            break

    if len(records) != TARGET_CASES:
        raise RuntimeError(
            f"insufficient_unique_admitted_anger_cases:{len(records)}/{TARGET_CASES}:rejected={dict(rejected)}"
        )

    return {
        "schema_version": SUITE_SCHEMA,
        "frozen_before_quality_results": True,
        "selection_rule": (
            "first 24 unique reachable admitted Anger post-play public signatures in committed 480001..480064 order; "
            "selection uses only public second-turn Anger presence and reconstruction admission, never Search/oracle quality"
        ),
        "source_seed_file": str(SEED_FILE.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "source_seed_range": [480001, 480064],
        "quality_results_observed_during_selection": 0,
        "source_battle_context_exported": 0,
        "source_hidden_rng_exported": 0,
        "source_move_history_exported": 0,
        "source_counter_history_exported": 0,
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
    print("ANGER_MIDTURN_FRESH_HOLDOUT_SUITE_FREEZE = PASS")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"PUBLIC_INTENT_COUNTS = {suite['public_intent_counts']}")
    print(f"REJECTED_COUNTS = {suite['rejected_counts']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("SOURCE_BATTLE_CONTEXT_EXPORTED = 0")
    print("SOURCE_HIDDEN_RNG_EXPORTED = 0")
    print("SOURCE_MOVE_HISTORY_EXPORTED = 0")
    print("SOURCE_COUNTER_HISTORY_EXPORTED = 0")


if __name__ == "__main__":
    main()
