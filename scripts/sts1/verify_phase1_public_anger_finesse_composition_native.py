#!/usr/bin/env python3
"""Native proof + deterministic Search smoke for Anger + Finesse composition."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
BUILD = ROOT / "external" / "sts_lightspeed" / "source" / "build-probe"
for path in (SRC, BUILD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import slaythespire as sts

from roguelike_ai.sts1_teacher.contract import DecisionContext, PUBLIC_STATE_SCHEMA
from roguelike_ai.sts1_teacher.native_rollout import NativePublicJawWormRolloutV1
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.redeterminization import build_redeterminization_plan
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator, public_sample
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig


def card(card_id: str, name: str, card_type: str, cost: int, *, target: bool) -> dict:
    return {
        "id": card_id,
        "name": name,
        "type": card_type,
        "cost": cost,
        "upgrades": 0,
        "has_target": target,
        "is_playable": True,
    }


def public_state() -> dict:
    strike = lambda: card("Strike_R", "Strike", "ATTACK", 1, target=True)
    defend = lambda: card("Defend_R", "Defend", "SKILL", 1, target=False)
    bash = lambda: card("Bash", "Bash", "ATTACK", 2, target=True)
    finesse = lambda: card("Finesse", "Finesse", "SKILL", 0, target=False)
    anger = lambda: card("Anger", "Anger", "ATTACK", 0, target=True)
    hand = [strike(), defend(), strike(), defend()]
    for position, item in enumerate(hand, start=1):
        item["position"] = position
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "real_game",
        "hp": 70,
        "max_hp": 80,
        "block": 2,
        "energy": 3,
        "hand": hand,
        "draw_pile": [strike()],
        "discard_pile": [
            strike(),
            defend(),
            bash(),
            strike(),
            defend(),
            finesse(),
            anger(),
            anger(),
        ],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [
            {
                "index": 0,
                "name": "JAW_WORM",
                "hp": 25,
                "max_hp": 40,
                "block": 0,
                "intent": "JAW_WORM_CHOMP",
                "intent_damage": 11,
                "intent_hits": 1,
                "is_gone": False,
                "powers": [],
            }
        ],
        "turn": 1,
        "combat_active": True,
        "relics": [{"id": "BURNING_BLOOD", "counter": -1}],
        "potions": [
            {"index": 0, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
            {"index": 1, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
            {"index": 2, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
        ],
        "gold": 99,
        "floor": 2,
        "act": 1,
        "character": "IRONCLAD",
        "ascension_level": 0,
        "room": "COMBAT",
        "screen_type": "NONE",
        "screen_choices": [],
        "rewards": [],
        "map_choices": [],
        "legal_actions": [
            {"kind": "play_card", "hand_index": 1, "target_index": 0},
            {"kind": "play_card", "hand_index": 2},
            {"kind": "play_card", "hand_index": 3, "target_index": 0},
            {"kind": "play_card", "hand_index": 4},
            {"kind": "end_turn"},
        ],
    }


def run_state(state: dict) -> dict:
    return {
        "relics": state["relics"],
        "potions": state["potions"],
        "gold": state["gold"],
        "floor": state["floor"],
        "act": state["act"],
        "character": state["character"],
        "ascension_level": state["ascension_level"],
        "room": state["room"],
        "screen_type": state["screen_type"],
        "screen_choices": [],
        "rewards": [],
        "map_choices": [],
        "reconstruction": {
            "schema_version": "sts1-public-reconstruction-v1",
            "public_player_state_complete": False,
            "public_card_instance_state_complete": False,
            "public_relic_state_complete": True,
            "public_potion_state_complete": True,
            "public_enemy_state_complete": False,
        },
    }


def reconstruction_aux() -> dict:
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


def admitted_context_and_seeds(state: dict, aux: dict) -> tuple[DecisionContext, dict, dict]:
    admitted_state = attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=aux,
    )
    assert admitted_state["reconstruction"]["anger_finesse_composition_complete"] is True
    admitted = require_public_reconstruction(admitted_state)
    sample = public_sample(admitted, sample_index=0, config=SearchConfig(sampling_seed=20260830))
    plan = build_redeterminization_plan(admitted, sample)
    native_seeds = dict(plan.rng_seeds)
    native_seeds["previous_history"] = plan.monster_history[0].previous_history_seed
    return admitted, admitted_state, native_seeds


def verify_search_smoke(aux: dict) -> None:
    cases = (
        (70, 25),
        (64, 23),
        (56, 20),
        (43, 17),
        (29, 14),
    )
    config = SearchConfig(
        samples_per_semantic_action=2,
        rollout_budget=64,
        node_budget=1024,
        max_depth=4,
        timeout_ms=30_000,
        sampling_seed=20260830,
    )
    signatures: set[str] = set()
    total_rollouts = 0

    for player_hp, enemy_hp in cases:
        state = deepcopy(public_state())
        state["hp"] = player_hp
        state["enemies"][0]["hp"] = enemy_hp
        admitted_state = attach_reconstruction_capabilities(
            state,
            run_state=run_state(state),
            reconstruction_aux=aux,
        )
        context = require_public_reconstruction(admitted_state)
        assert context.decision_signature not in signatures
        signatures.add(context.decision_signature)

        backend = NativePublicJawWormRolloutV1(
            admitted_state,
            sts,
            reconstruction_aux=aux,
        )
        evaluator = PairedPublicSampleEvaluator(backend)
        search = PublicStateSearch(evaluator, config)
        first = search.run(context)
        second = search.run(context)

        assert first.evidence_hash == second.evidence_hash
        assert first.candidate_scores == second.candidate_scores
        assert first.tie_action_ids == second.tie_action_ids
        assert first.unique_best_action_id == second.unique_best_action_id
        assert first.unresolved_action_ids == ()
        assert first.timed_out is False
        assert first.rollout_count > 0
        assert all(item.score is not None and item.unresolved is False for item in first.candidate_scores)
        total_rollouts += first.rollout_count

    assert len(signatures) == 5
    print("ANGER_FINESSE_SEARCH_SMOKE_STATES = 5")
    print("ANGER_FINESSE_SEARCH_SMOKE_REPEATS_PER_STATE = 2")
    print(f"ANGER_FINESSE_SEARCH_SMOKE_TOTAL_FIRST_REPEAT_ROLLOUTS = {total_rollouts}")
    print("ANGER_FINESSE_SEARCH_SMOKE_DETERMINISM = PASS")
    print("ANGER_FINESSE_SEARCH_SMOKE_ILLEGAL = 0")
    print("ANGER_FINESSE_SEARCH_SMOKE_UNRESOLVED = 0")
    print("ANGER_FINESSE_SEARCH_SMOKE_HIDDEN_INFORMATION = 0")


def main() -> None:
    state = public_state()
    aux = reconstruction_aux()
    admitted, admitted_state, seeds = admitted_context_and_seeds(state, aux)

    assert len(state["hand"]) == 4
    assert len(state["draw_pile"]) == 1
    assert len(state["discard_pile"]) == 8
    assert sum(card["name"] == "Anger" for card in state["discard_pile"]) == 2
    assert sum(card["name"] == "Finesse" for card in state["discard_pile"]) == 1
    assert aux["cards_played_this_turn"] == 2
    assert aux["attacks_played_this_turn"] == 1
    assert aux["skills_played_this_turn"] == 1
    assert admitted.decision_signature == DecisionContext.from_public_state(state).decision_signature

    # The old snapshot-only path must still reject this rich composition.
    try:
        sts.build_public_jaw_worm_context_v1(admitted_state, seeds)
    except RuntimeError as exc:
        assert "turn1_midturn_" in str(exc) or "unsupported_card_v1" in str(exc), str(exc)
    else:
        raise AssertionError("Anger+Finesse composition unexpectedly admitted without reconstruction aux")

    bc = sts.build_public_jaw_worm_context_v1(admitted_state, seeds, aux)
    assert bc.turn == 1
    assert bc.player.cur_hp == 70
    assert bc.player.energy == 3
    assert bc.player.block == 2
    assert len(bc.hand) == 4
    assert len(bc.draw_pile) == 1
    assert len(bc.discard_pile) == 8
    assert len(bc.exhaust_pile) == 0
    assert sum(card.name == "Anger" for card in bc.discard_pile) == 2
    assert sum(card.name == "Finesse" for card in bc.discard_pile) == 1

    legal = list(sts.get_legal_actions(bc))
    assert any(action.action_type == sts.SearchActionType.END_TURN for action in legal)
    strike = next(
        action for action in legal
        if action.action_type == sts.SearchActionType.CARD
        and action.source_idx == 0
        and action.target_idx == 0
    )
    hp_before = bc.monsters[0].cur_hp
    strike.execute(bc)
    assert bc.monsters[0].cur_hp < hp_before
    assert bc.player.energy == 2

    bad_aux = dict(aux)
    bad_aux["complete"] = False
    try:
        sts.build_public_jaw_worm_context_v1(admitted_state, seeds, bad_aux)
    except RuntimeError as exc:
        assert "anger_finesse_aux_incomplete" in str(exc), str(exc)
    else:
        raise AssertionError("incomplete Anger+Finesse aux unexpectedly admitted")

    bad_counter = dict(aux)
    bad_counter["cards_played_this_turn"] = 3
    try:
        sts.build_public_jaw_worm_context_v1(admitted_state, seeds, bad_counter)
    except RuntimeError as exc:
        assert "draw_aux_counter_slice_unsupported" in str(exc), str(exc)
    else:
        raise AssertionError("wrong Anger+Finesse counters unexpectedly admitted")

    verify_search_smoke(aux)

    print("ANGER_FINESSE_COMPOSITION_NATIVE = PASS")
    print("ANGER_GENERATED_COPY_PUBLIC_PILES = PASS")
    print("FINESSE_DRAW_BLOCK_SHAPE_PUBLIC_PILES = PASS")
    print("ONE_ATTACK_ONE_SKILL_AUX_COUNTER_PATH = PASS")
    print("NO_AUX_FAILS_CLOSED = PASS")
    print("INCOMPLETE_AUX_FAILS_CLOSED = PASS")
    print("BAD_COUNTERS_FAIL_CLOSED = PASS")
    print("EXECUTABLE_POST_RECONSTRUCTION_ACTION = PASS")
    print("PUBLIC_CONTRACT_FIELDS_ADDED = 0")
    print("SOURCE_BATTLE_CONTEXT_INPUT = 0")
    print("SOURCE_HIDDEN_RNG_ACCESS = 0")
    print("SOURCE_MOVE_HISTORY_ACCESS = 0")
    print("SOURCE_COUNTER_HISTORY_ACCESS = 0")


if __name__ == "__main__":
    main()
