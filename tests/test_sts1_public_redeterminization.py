from roguelike_ai.sts1_teacher.contract import DecisionContext, PUBLIC_STATE_SCHEMA
from roguelike_ai.sts1_teacher.redeterminization import (
    REDETERMINIZATION_SCHEMA,
    build_redeterminization_plan,
)
from roguelike_ai.sts1_teacher.sampling import public_sample
from roguelike_ai.sts1_teacher.search import SearchConfig


def public_state() -> dict:
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "simulator",
        "hp": 70,
        "max_hp": 80,
        "block": 0,
        "energy": 3,
        "hand": [{"position": 1, "id": "STRIKE_RED", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0, "has_target": True}],
        "draw_pile": [{"id": "DEFEND_RED", "name": "Defend", "type": "SKILL", "cost": 1, "upgrades": 0, "has_target": False}],
        "discard_pile": [],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [
            {"index": 0, "name": "Cultist", "hp": 48, "max_hp": 48, "block": 0, "intent": "DARK_STRIKE", "intent_damage": 6, "intent_hits": 1, "is_gone": False, "powers": []},
            {"index": 1, "name": "Jaw Worm", "hp": 40, "max_hp": 40, "block": 0, "intent": "CHOMP", "intent_damage": 11, "intent_hits": 1, "is_gone": False, "powers": []},
        ],
        "turn": 2,
        "combat_active": True,
        "relics": [],
        "potions": [],
        "gold": 0,
        "floor": 1,
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
            {"kind": "play_card", "hand_index": 1, "target_index": 1},
            {"kind": "end_turn"},
        ],
    }


def test_redeterminization_is_exactly_repeatable() -> None:
    context = DecisionContext.from_public_state(public_state())
    config = SearchConfig(sampling_seed=17)
    sample = public_sample(context, sample_index=3, config=config)

    first = build_redeterminization_plan(context, sample)
    second = build_redeterminization_plan(context, sample)

    assert first == second
    assert first.schema_version == REDETERMINIZATION_SCHEMA
    assert first.sample_index == 3
    assert {name for name, _ in first.rng_seeds} == {
        "ai", "card_random", "misc", "monster_hp", "potion", "shuffle", "draw_order"
    }


def test_plan_has_no_candidate_action_input_so_candidates_share_it() -> None:
    context = DecisionContext.from_public_state(public_state())
    sample = public_sample(context, sample_index=0, config=SearchConfig(sampling_seed=123))
    plan = build_redeterminization_plan(context, sample)

    assert len(context.legal_actions) == 3
    assert plan == build_redeterminization_plan(context, sample)
    assert plan.monster_history[0].current_public_intent == "DARK_STRIKE"
    assert plan.monster_history[1].current_public_intent == "CHOMP"


def test_previous_monster_history_is_sampled_not_copied_from_input() -> None:
    state = public_state()
    # The public contract does not contain prior move history at all.
    assert all("move" not in key.lower() for enemy in state["enemies"] for key in enemy)

    context = DecisionContext.from_public_state(state)
    config = SearchConfig(sampling_seed=99)
    sample0 = public_sample(context, sample_index=0, config=config)
    sample1 = public_sample(context, sample_index=1, config=config)
    plan0 = build_redeterminization_plan(context, sample0)
    plan1 = build_redeterminization_plan(context, sample1)

    assert [item.current_public_intent for item in plan0.monster_history] == ["DARK_STRIKE", "CHOMP"]
    assert [item.previous_history_seed for item in plan0.monster_history] != [
        item.previous_history_seed for item in plan1.monster_history
    ]


def test_all_domain_seeds_change_with_public_sample() -> None:
    context = DecisionContext.from_public_state(public_state())
    config = SearchConfig(sampling_seed=7)
    first = build_redeterminization_plan(context, public_sample(context, sample_index=0, config=config))
    second = build_redeterminization_plan(context, public_sample(context, sample_index=1, config=config))

    assert dict(first.rng_seeds) != dict(second.rng_seeds)
    assert all(0 <= value < 2**64 for _, value in first.rng_seeds)
