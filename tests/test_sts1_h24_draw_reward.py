from copy import deepcopy
import inspect
import pytest
from roguelike_ai.sts1_phase3.draw_reward import select_draw_reward, validate_override, effective_override
from roguelike_ai.sts1_phase3.simulator import run_simulator_game, SimulatorRunError
from scripts.sts1 import sts1_g7_h16_lethal_defend_eval as runner

TYPES = {"Strike": "ATTACK", "Defend": "SKILL", "Pommel Strike": "ATTACK", "Shrug It Off": "SKILL", "Battle Trance": "SKILL"}
DECK = [{"name": "Strike", "upgrades": 0}, {"name": "Defend", "upgrades": 0}]
CHOICES = [{"kind": "card", "choice": "card", "card_name": "Strike"},
           {"kind": "card", "choice": "card", "card_name": "Shrug It Off"}, {"kind": "card", "choice": "skip"}]


def event(candidate=False):
    selected, detail = select_draw_reward(0, CHOICES, DECK, TYPES)
    return {"type": "noncombat_decision_trace_v1", "game_step": 3, "screen_before": "REWARDS",
        "state_before": {"deck": deepcopy(DECK)}, "state_after": {"deck": deepcopy(DECK) +
            [{"name": "Shrug It Off" if candidate else "Strike", "upgrades": 0}]},
        "legal_choices_complete": True,
        "legal_choices": [{"index": i, "semantics": c} for i, c in enumerate(CHOICES)],
        "recommended_legal_action_index": 0, "selected_legal_action_index": selected if candidate else 0,
        "draw_reward_override": candidate, "draw_reward_intervention": detail if candidate else None,
        "draw_reward_reason": detail["reason"] if candidate else "disabled_or_not_reward"}


@pytest.mark.parametrize("name", ["Pommel Strike", "Shrug It Off", "Battle Trance"])
def test_unique_existing_draw_option(name):
    choices = deepcopy(CHOICES)
    choices[1]["card_name"] = name
    before = deepcopy((choices, DECK, TYPES))
    selected, detail = select_draw_reward(0, choices, DECK, TYPES)
    assert selected == 1 and detail["overridden"]
    assert (choices, DECK, TYPES) == before


@pytest.mark.parametrize("deck,choices,index,complete", [
    ([], CHOICES, 0, True), ([{"name": "unknown"}], CHOICES, 0, True),
    (DECK, CHOICES, 0, False), (DECK, CHOICES, True, True),
    (DECK, CHOICES, 2, True), (DECK, CHOICES, 1, True),
    (DECK, CHOICES + [CHOICES[1]], 0, True),
    (DECK, CHOICES[:1], 0, True),
    (DECK, CHOICES + [{"kind": "card", "choice": "card", "card_name": "unknown"}], 0, True),
    (DECK + [{"name": "Shrug It Off", "upgrades": 0}] * 2, CHOICES, 0, True),
    (DECK + [{"name": "Shrug It Off+", "upgrades": 0}], CHOICES, 0, True),
])
def test_abstention(deck, choices, index, complete):
    selected, detail = select_draw_reward(index, choices, deck, TYPES, legal_choices_complete=complete)
    assert selected == index and not detail["overridden"]


def test_upgraded_draw_counts_and_one_copy_allows_change():
    assert select_draw_reward(0, CHOICES, DECK + [{"name": "Battle Trance+", "upgrades": 1}], TYPES)[0] == 1


def test_override_and_prefix_integrity():
    p, c = event(), event(True)
    validate_override(c, TYPES)
    assert effective_override([p], [c])
    assert runner._pair_has_effective_override({"events": [p]}, {"events": [c]}, "h24")
    bad = deepcopy(c)
    bad["selected_legal_action_index"] = 2
    with pytest.raises(ValueError):
        validate_override(bad, TYPES)
    bad = deepcopy(c)
    bad["state_before"]["gold"] = 1
    assert not effective_override([p], [bad])
    bad = deepcopy(c)
    bad["state_after"] = deepcopy(p["state_after"])
    assert not effective_override([p], [bad])
    prefix = {"type": "combat_decision_trace_v1", "selected_action": "a"}
    assert not effective_override([prefix, p], [{**prefix, "selected_action": "b"}, c])


def test_actual_added_card_not_only_other_state_difference():
    p, c = event(), event(True)
    c["state_after"] = deepcopy(p["state_after"])
    c["state_after"]["hp"] = 99
    assert not effective_override([p], [c])
    c = event(True)
    c["state_after"]["deck"][-1]["name"] = "Battle Trance"
    assert not effective_override([p], [c])
    c = event(True)
    c["state_after"]["deck"][0]["name"] = "Defend"
    assert not effective_override([p], [c])


def test_activation_guards_and_fixed_stages():
    assert inspect.signature(run_simulator_game).parameters["prefer_draw_reward_under_two"].default is False
    with pytest.raises(SimulatorRunError, match="boolean"):
        run_simulator_game(student=None, sts=None, seed=0, prefer_draw_reward_under_two=1)
    with pytest.raises(SimulatorRunError, match="requires ArmG"):
        run_simulator_game(student=None, sts=None, seed=0, prefer_draw_reward_under_two=True)
    assert runner.STAGE_CONFIG["train"]["paired_seed_count"] == 10
    assert runner.STAGE_CONFIG["probe"]["paired_seed_count"] == 10
    assert runner.STAGE_CONFIG["dev"]["paired_seed_count"] == 30
