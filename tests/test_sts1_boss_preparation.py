from copy import deepcopy
import json

import pytest

from roguelike_ai.sts1_phase3.boss_preparation import (
    SAFETY, UNKNOWN, profile_trace, public_summary, resource_facts,
)

EMPTY = ["EMPTY_POTION_SLOT"] * 5
META = {"stage": "train", "purpose": "baseline_live_resource_diagnosis", "seed_id": 123456789}


def run_state():
    return {"act": 3, "floor": 49, "hp": 70, "max_hp": 80, "gold": 30,
            "deck": [{"name": "Inflame", "upgrades": 1}, {"name": "Defend", "upgrades": 0}],
            "relics": ["Burning Blood"], "potions": ["FIRE_POTION", *EMPTY[1:]],
            "potion_inventory_complete": True}


def trace():
    run = run_state()
    return [
        {"type": "diagnostic_trace_header_v1", "trace_schema": "sts1-diagnostic-trace-v1",
         "run_metadata": deepcopy(META), "live_combat_resources_required": True},
        {"type": "noncombat_decision_trace_v1", "game_step": 1,
         "act": 3, "floor": 50, "screen_before": "REST_ROOM",
         "legal_choices_complete": True, "selected_legal_action_index": 1,
         "legal_choices": [{"index": 0, "semantics": {"kind": "rest", "option_index": 0}},
                           {"index": 1, "semantics": {"kind": "rest", "option_index": 1}}],
         "state_before": deepcopy(run), "state_after": deepcopy(run)},
        {"type": "encounter_started_v1", "game_step": 2, "act": 3, "floor": 50,
         "state": {"run": deepcopy(run), "player": {"hp": 64, "max_hp": 80},
                   "enemies": [{"name": "AWAKENED_ONE"}, {"name": "Cultist"}],
                   "combat_potion_inventory": {"potions": EMPTY.copy(), "potion_inventory_complete": True}}},
        {"type": "terminal_trace_v1", "complete": True, "outcome": "defeat", "error": None,
         "legal_actions_complete": True, "potion_inventory_snapshot_complete": True,
         "final_floor": 50, "final_act": 3, "final_hp": 0, **dict.fromkeys(SAFETY, 0)},
    ]


def test_current_facts_do_not_infer_capabilities_or_future_boss():
    original = trace()
    saved = deepcopy(original)
    profile = profile_trace(iter(original), META)
    decision = profile["decision_records"][0]
    assert decision["floor"] == 49  # pre-action floor, not the event's after-action floor
    assert decision["before"]["deck_size"] == 2
    assert decision["before"]["upgraded_card_count"] == 1
    assert decision["before"]["boss_identity_at_decision"] == UNKNOWN
    assert set(decision["before"]["deck_capabilities"].values()) == {UNKNOWN}
    assert profile["encounter_records"][0]["observed_boss_at_entry"] == ["AWAKENED_ONE"]
    assert profile["encounter_records"][0]["resources"]["hp_fraction"] == .8
    assert profile["encounter_records"][0]["resources"]["potion_count"] == 0
    assert original == saved
    profile["decision_records"][0]["before"]["deck"][0]["name"] = "changed"
    assert original == saved


def test_missing_live_inventory_never_uses_run_potions():
    facts = resource_facts(run_state(), player={"hp": 30, "max_hp": 80})
    assert facts["potion_count"] is None and facts["potions"] is None
    assert facts["potion_scope"] == "live_battle"
    assert resource_facts(run_state())["potion_count"] == 1


@pytest.mark.parametrize("deck", [None, [{"name": "Inflame"}], [{"name": "Defend", "upgrades": True}]])
def test_incomplete_deck_is_unknown_not_empty(deck):
    run = run_state()
    run["deck"] = deck
    assert resource_facts(run)["deck_size"] is None


@pytest.mark.parametrize("hp,max_hp", [(True, 80), (81, 80), (-1, 80), (0, 0)])
def test_invalid_hp_is_unknown(hp, max_hp):
    run = run_state()
    run.update(hp=hp, max_hp=max_hp)
    assert resource_facts(run)["hp_fraction"] is None


@pytest.mark.parametrize("stage", ["probe", "dev", "confirmation", "formal"])
def test_held_out_trace_rejected(stage):
    metadata = {**META, "stage": stage}
    events = trace()
    events[0]["run_metadata"] = metadata
    with pytest.raises(ValueError, match="Train"):
        profile_trace(events, metadata)


@pytest.mark.parametrize("mutation", ["metadata", "header", "terminal", "unsafe", "incomplete", "choice", "bool_index", "order", "trailing"])
def test_malformed_provenance_or_incomplete_safety_rejected(mutation):
    events = trace()
    if mutation == "metadata":
        events[0]["run_metadata"]["seed_id"] += 1
    elif mutation == "header":
        events.insert(1, deepcopy(events[0]))
    elif mutation == "terminal":
        events.pop()
    elif mutation == "unsafe":
        events[-1]["communication_error_count"] = 1
    elif mutation == "incomplete":
        events[1]["legal_choices_complete"] = False
    elif mutation == "choice":
        events[1]["selected_legal_action_index"] = -1
    elif mutation == "bool_index":
        events[1]["legal_choices"][0]["index"] = False
    elif mutation == "order":
        events[2]["game_step"] = 1
    else:
        events.append({"type": "combat_decision_trace_v1"})
    with pytest.raises(ValueError):
        profile_trace(events, META)


def test_canonical_empty_reward_skip_is_not_a_missing_choice():
    events = trace()
    events[1].update(screen_before="REWARDS", legal_choices=[], selected_legal_action_index=-1,
                     selected_choice="armg:reward:skip_empty")
    assert profile_trace(events, META)["decision_records"][0]["selected"] == {"choice": "skip_empty"}


def test_public_counters_no_raw_state_identifiers_or_invented_win_gain():
    profile = profile_trace(trace(), META)
    summary = public_summary([profile])
    assert summary["baseline_runs"] == 1 and summary["baseline_wins"] == 0
    assert summary["observed_boss_entries"] == 1
    assert summary["final_encounter_hp_bands"] == {"60_to_80_percent": 1}
    assert summary["unknown_decision_boss_count"] == 1
    assert summary["win_rate_improvement"] == UNKNOWN
    public = json.dumps(summary)
    assert all(text not in public for text in ("123456789", "Inflame", "FIRE_POTION", "AWAKENED_ONE", "sha256"))
