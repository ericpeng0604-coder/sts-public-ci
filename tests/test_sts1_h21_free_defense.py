from copy import deepcopy
import pytest
from roguelike_ai.sts1_phase3.free_defense_end_turn import select_free_defense_before_end_turn as select
from scripts.sts1 import sts1_g7_h16_lethal_defend_eval as runner


def case(card="INTIMIDATE"):
    state = {"combat_active": True, "energy": 0, "block": 0, "powers": [], "relics": [],
             "hand": [{"id": card, "position": 1, "cost": 0, "upgrades": 0,
                       "is_playable": True, "has_target": False}],
             "enemies": [{"name": "AWAKENED_ONE", "hp": 50, "is_gone": False,
                          "powers": [], "intent_damage": 20, "intent_hits": 1}]}
    actions = [{"kind": "end_turn"}, {"kind": "play_card", "hand_index": 1}]
    return state, actions


@pytest.mark.parametrize("card", ["DEFEND_RED", "GOOD_INSTINCTS", "INTIMIDATE"])
def test_zero_energy_exact_native_ordinal_and_immutable_evidence(card):
    state, actions = case(card)
    original = deepcopy((state, actions))
    choice, _ = select(state, actions + [dict(actions[1])], actions[0], legal_actions_complete=True)
    assert choice.native_action_index == 1 and choice.action == actions[1]
    assert (state, actions) == original


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(powers=[{"name": "NoBlock", "amount": 1}]),
    lambda s: s.update(relics=["VELVET_CHOKER"]),
    lambda s: s.update(relics=["DEAD_BRANCH"]),
    lambda s: s.update(block=30),
    lambda s: s["hand"][0].update(cost=1),
    lambda s: s["hand"][0].update(cost=False),
    lambda s: s["hand"].append({"id": "PAIN", "position": 2}),
    lambda s: s["enemies"][0].update(name="TIME_EATER"),
    lambda s: s["enemies"][0].update(name="GREMLIN_NOB"),
    lambda s: s["enemies"][0].update(powers=[{"name": "Artifact", "amount": 1}]),
    lambda s: s["enemies"][0].update(powers=[{"name": "Weak", "amount": 1}]),
])
def test_risks_and_useless_weak_abstain(mutation):
    state, actions = case()
    mutation(state)
    assert select(state, actions, actions[0], legal_actions_complete=True)[0] is None


def test_completeness_and_ambiguous_action_fail_closed():
    state, actions = case()
    assert select(state, actions, actions[0], legal_actions_complete=False)[0] is None
    assert select(state, actions + [dict(actions[1], target_index=0)], actions[0], legal_actions_complete=True)[0] is None
    assert select(state, actions[1:], actions[0], legal_actions_complete=True)[0] is None


def test_frozen_stage_budget_and_separate_replay_count():
    assert runner._stage_episode_counts("h21", "train", 10)["expected_episodes"] == 21
    assert runner.TRIAL_ROUND_IDS["h21"] == "round-015-20261011"
    assert not runner._trace_replay_matches({"free_defense_override_count": 1},
        {"free_defense_override_count": 2}, "same", "same", 1, 1, "h21")
