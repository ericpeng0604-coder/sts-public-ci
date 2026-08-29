#!/usr/bin/env python3
"""Blind 530xxx discovery suite for the exact Anger + Finesse composition.

Selection is frozen before quality labels. Each source seed adds one normal
Anger and one normal Finesse, ends the opening player turn without cards, and
requires both rich cards in the public second-turn hand. Both public execution
orders are replayed from the same seed. They must produce the same formal
DecisionContext observation, legal actions, and decision signature; otherwise
that source seed fails closed.

Source BattleContext/RNG exist only to generate reachable public states. They
are never exported into the formal Teacher observation or candidate policy.
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

SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_anger_finesse_composition_discovery_seeds_256.txt"
SUITE_SCHEMA = "sts1-phase1-anger-finesse-composition-discovery-suite-v1"
SAMPLING_CONTRACT = "anger-finesse-composition-discovery-fixed-256-first24-v1"
EXPECTED_SOURCE_SEEDS = 256
MIN_SEED = 530001
MAX_SEED = 530256
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
        "cards_played_this_turn": 2,
        "attacks_played_this_turn": 1,
        "skills_played_this_turn": 1,
        "cards_discarded_this_turn": 0,
    }


def suite_digest(records: list[dict]) -> str:
    payload = "\n".join(canonical_json(record) for record in records) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name if isinstance(name, str) else value).rsplit(".", 1)[-1].upper()


def _card_id(card: dict) -> str:
    value = str(card.get("id", "")).strip().upper().replace(" ", "_").replace("-", "_")
    return {"STRIKE_R": "STRIKE_RED", "DEFEND_R": "DEFEND_RED"}.get(value, value)


def _end_first_player_turn_without_cards(bc) -> None:
    action = next((a for a in sts.get_legal_actions(bc) if _enum_name(a.action_type) == "END_TURN"), None)
    if action is None:
        raise RuntimeError("opening_end_turn_missing")
    action.execute(bc)


def _project(adapter, bc, run: dict) -> dict:
    return adapter.adapt(bc, legal_actions=list(sts.get_legal_actions(bc)), run_state=run)


def _public_card_action(context: DecisionContext, wanted_id: str):
    positions = [
        int(card.get("position", index + 1))
        for index, card in enumerate(context.state.get("hand", []))
        if _card_id(card) == wanted_id
    ]
    if len(positions) != 1:
        raise RuntimeError(f"boundary_{wanted_id.lower()}_hand_count:{len(positions)}")
    position = positions[0]
    candidates = [
        action
        for action in context.legal_actions
        if action.payload.get("kind") == "play_card"
        and int(action.payload.get("hand_index", -1)) == position
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"boundary_{wanted_id.lower()}_public_action_count:{len(candidates)}")
    return candidates[0]


def _execute_public_card_on_source(bc, public_action) -> None:
    source_idx = int(public_action.payload["hand_index"]) - 1
    target = public_action.payload.get("target_index")
    for native in sts.get_legal_actions(bc):
        if _enum_name(native.action_type) != "CARD" or int(native.source_idx) != source_idx:
            continue
        if target is not None and int(native.target_idx) != int(target):
            continue
        native.execute(bc)
        return
    raise RuntimeError(f"public_native_generation_action_mismatch:{public_action.action_id}")


def _assert_post_composition(state: dict, *, source_seed: int) -> str:
    if state.get("turn") != 1:
        raise RuntimeError(f"composition_not_turn_one:{source_seed}:{state.get('turn')}")
    shape = {
        "hand": len(state.get("hand", [])),
        "draw": len(state.get("draw_pile", [])),
        "discard": len(state.get("discard_pile", [])),
        "exhaust": len(state.get("exhaust_pile", [])),
    }
    if shape != {"hand": 4, "draw": 1, "discard": 8, "exhaust": 0}:
        raise RuntimeError(f"composition_pile_shape:{source_seed}:{shape}")
    if state.get("energy") != 3 or state.get("block") != 2:
        raise RuntimeError(
            f"composition_player_shape:{source_seed}:energy={state.get('energy')}:block={state.get('block')}"
        )
    if state.get("powers") != []:
        raise RuntimeError(f"composition_player_powers:{source_seed}:{state.get('powers')}")

    piles = {name: state.get(name, []) for name in ("hand", "draw_pile", "discard_pile", "exhaust_pile")}
    ids = [_card_id(card) for pile in piles.values() for card in pile]
    expected = Counter({"STRIKE_RED": 5, "DEFEND_RED": 4, "BASH": 1, "FINESSE": 1, "ANGER": 2})
    if Counter(ids) != expected:
        raise RuntimeError(f"composition_card_composition:{source_seed}:{Counter(ids)}")

    discard_angers = sum(_card_id(card) == "ANGER" for card in piles["discard_pile"])
    other_angers = sum(
        _card_id(card) == "ANGER"
        for name in ("hand", "draw_pile", "exhaust_pile")
        for card in piles[name]
    )
    discard_finesses = sum(_card_id(card) == "FINESSE" for card in piles["discard_pile"])
    other_finesses = sum(
        _card_id(card) == "FINESSE"
        for name in ("hand", "draw_pile", "exhaust_pile")
        for card in piles[name]
    )
    if discard_angers != 2 or other_angers != 0:
        raise RuntimeError(
            f"composition_generated_anger_not_public:{source_seed}:discard={discard_angers}:other={other_angers}"
        )
    if discard_finesses != 1 or other_finesses != 0:
        raise RuntimeError(
            f"composition_finesse_not_public:{source_seed}:discard={discard_finesses}:other={other_finesses}"
        )

    enemies = state.get("enemies", [])
    if len(enemies) != 1 or enemies[0].get("name") != "JAW_WORM":
        raise RuntimeError(f"composition_enemy:{source_seed}:{enemies}")
    intent = str(enemies[0].get("intent", ""))
    if not intent.startswith("JAW_WORM_"):
        raise RuntimeError(f"composition_public_intent:{source_seed}:{intent}")
    return intent


def _run_order(source_seed: int, order: tuple[str, str]) -> tuple[dict, DecisionContext, list[str], str]:
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, source_seed, 0)
    gc.obtain_card(sts.Card(sts.CardId.ANGER))
    gc.obtain_card(sts.Card(sts.CardId.FINESSE))
    if len(list(gc.deck)) != 12:
        raise RuntimeError(f"deck_size_after_rich_cards:{len(list(gc.deck))}")
    gc.floor_num = 1
    gc.cur_room = sts.Room.MONSTER
    bc = sts.BattleContext()
    bc.init_encounter(gc, sts.MonsterEncounter.JAW_WORM)
    if int(bc.turn) != 0:
        raise RuntimeError(f"opening_turn:{bc.turn}")

    _end_first_player_turn_without_cards(bc)
    run = public_run_state(gc)
    boundary = _project(SimulatorCombatAdapter(), bc, run)
    boundary_context = DecisionContext.from_public_state(boundary)
    _public_card_action(boundary_context, "ANGER")
    _public_card_action(boundary_context, "FINESSE")

    action_ids: list[str] = []
    for wanted_id in order:
        projected = _project(SimulatorCombatAdapter(), bc, run)
        context = DecisionContext.from_public_state(projected)
        action = _public_card_action(context, wanted_id)
        action_ids.append(action.action_id)
        _execute_public_card_on_source(bc, action)

    projected = _project(SimulatorCombatAdapter(), bc, run)
    intent = _assert_post_composition(projected, source_seed=source_seed)
    admitted_state = attach_reconstruction_capabilities(
        projected,
        run_state=run,
        reconstruction_aux=aux(),
    )
    if not admitted_state.get("reconstruction", {}).get("anger_finesse_composition_complete"):
        raise RuntimeError("composition_formal_admission_failed")
    context = require_public_reconstruction(admitted_state)
    return admitted_state, context, action_ids, intent


def _legal_observation(context: DecisionContext) -> list[dict]:
    return [
        {"action_id": action.action_id, "semantic_key": action.semantic_key, "payload": action.payload}
        for action in context.legal_actions
    ]


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
                "case_id": f"anger-finesse-composition-discovery-v1-{len(records):02d}",
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
            f"insufficient_unique_admitted_composition_cases:{len(records)}/{TARGET_CASES}:rejected={dict(rejected)}"
        )

    return {
        "schema_version": SUITE_SCHEMA,
        "sampling_contract_version": SAMPLING_CONTRACT,
        "purpose": "DISCOVERY_ONLY_NOT_HOLDOUT",
        "frozen_before_quality_results": True,
        "selection_rule": (
            "first 24 unique reachable admitted Anger+Finesse formal public signatures in committed 530001..530256 "
            "ascending order; selection uses only public rich-card presence, exact formal admission, and cross-order "
            "formal-observation equivalence, never Search/oracle quality"
        ),
        "source_seed_file": str(SEED_FILE.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "source_seed_range": [MIN_SEED, MAX_SEED],
        "quality_results_observed_during_selection": 0,
        "prior_500xxx_quality_rows_read": 0,
        "prior_510xxx_quality_rows_read": 0,
        "prior_520xxx_quality_rows_read": 0,
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
    print("ANGER_FINESSE_DISCOVERY_SUITE_FREEZE = PASS")
    print(f"SAMPLING_CONTRACT = {suite['sampling_contract_version']}")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"PUBLIC_INTENT_COUNTS = {suite['public_intent_counts']}")
    print(f"REJECTED_COUNTS = {suite['rejected_counts']}")
    print(f"ORDER_EQUIVALENT_SOURCE_COUNT_BEFORE_STOP = {suite['order_equivalent_source_count_before_stop']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("PRIOR_500XXX_QUALITY_ROWS_READ = 0")
    print("PRIOR_510XXX_QUALITY_ROWS_READ = 0")
    print("PRIOR_520XXX_QUALITY_ROWS_READ = 0")
    print("SOURCE_BATTLE_CONTEXT_EXPORTED = 0")
    print("SOURCE_HIDDEN_RNG_EXPORTED = 0")
    print("SOURCE_MOVE_HISTORY_EXPORTED = 0")
    print("SOURCE_COUNTER_HISTORY_EXPORTED = 0")


if __name__ == "__main__":
    main()
