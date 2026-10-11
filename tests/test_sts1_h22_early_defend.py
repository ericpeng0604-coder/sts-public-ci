from copy import deepcopy
import pytest
from roguelike_ai.sts1_phase3.early_defend_end_turn import select_early_defend_before_end_turn as select
from scripts.sts1 import sts1_g7_h16_lethal_defend_eval as runner


def case():
    state = {"combat_active": True, "energy": 1, "block": 0, "hp": 50,
             "powers": [], "relics": [],
             "hand": [{"id": "DEFEND_RED", "position": 1, "cost": 1, "upgrades": 0,
                       "is_playable": True, "has_target": False}],
             "enemies": [{"name": "THE_CHAMP", "hp": 100, "is_gone": False,
                          "powers": [], "intent_damage": 10, "intent_hits": 1}]}
    return state, [{"kind": "end_turn"}, {"kind": "play_card", "hand_index": 1}]


def test_nonlethal_hp_preservation_preserves_real_cost_and_exact_ordinal():
    state, actions = case()
    original = deepcopy((state, actions))
    choice, reason = select(state, actions + [dict(actions[1])], actions[0], legal_actions_complete=True)
    assert choice.native_action_index == 1 and choice.action == actions[1]
    assert reason == "legal_early_defend_before_end_turn"
    assert (state, actions) == original


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(energy=0), lambda s: s.update(energy=True),
    lambda s: s.update(hp=10), lambda s: s.update(block=10),
    lambda s: s["hand"][0].update(cost=2), lambda s: s["hand"][0].update(cost=0),
    lambda s: s["hand"][0].update(id="POWER_THROUGH"),
    lambda s: s.update(powers=[{"name": "NoBlock", "amount": 1}]),
    lambda s: s.update(relics=["INK_BOTTLE"]),
    lambda s: s["enemies"][0].update(name="GREMLIN_NOB"),
    lambda s: s["enemies"][0].update(name="TIME_EATER"),
    lambda s: s["hand"].append({"id": "NORMALITY", "position": 2}),
])
def test_unavailable_energy_late_rescue_and_penalties_abstain(mutation):
    state, actions = case()
    mutation(state)
    assert select(state, actions, actions[0], legal_actions_complete=True)[0] is None


def test_h22_does_not_relabel_h21_and_keeps_stage_gate():
    assert runner._stage_episode_counts("h22", "train", 10)["expected_episodes"] == 21
    assert runner.TRIAL_ROUND_IDS["h22"] != runner.TRIAL_ROUND_IDS["h21"]
    assert runner.STAGE_CONFIG["train"]["minimum_override_seed_coverage"] == 3
    state, actions = case()
    assert select(state, actions, actions[0], legal_actions_complete=False)[0] is None
    assert select(state, actions + [dict(actions[1], target_index=0)], actions[0], legal_actions_complete=True)[0] is None


@pytest.mark.parametrize("relic", ["VELVET_CHOKER", "DEAD_BRANCH", "CHARONS_ASHES"])
def test_plain_nonexhausting_defend_uses_native_legality_for_harmless_relics(relic):
    state, actions = case()
    state["relics"] = [relic]
    assert select(state, actions, actions[0], legal_actions_complete=True)[0] is not None
    assert select(state, actions[:1], actions[0], legal_actions_complete=True)[0] is None
