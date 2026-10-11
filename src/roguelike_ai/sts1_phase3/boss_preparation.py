"""Read-only Train diagnostics, not a readiness predictor or action selector."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Mapping

SCHEMA = "sts1-boss-preparation-diagnostic-v1"
SAFETY = ("illegal_action_count", "crash_count", "timeout_count", "communication_error_count")
UNKNOWN = "NOT_VERIFIED"
EMPTY_POTION = "EMPTY_POTION_SLOT"
BOSS_NAMES = {
    "Slime Boss", "SlimeBoss", "The Guardian", "TheGuardian", "Guardian", "Hexaghost",
    "The Champ", "Champ", "The Collector", "Collector", "TheCollector",
    "Bronze Automaton", "BronzeAutomaton", "Time Eater", "TimeEater",
    "Awakened One", "AwakenedOne", "Donu", "Deca",
}
BOSS_KEYS = {name.replace(" ", "").replace("_", "").upper() for name in BOSS_NAMES}


def _integer(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def resource_facts(run: Mapping[str, Any], *, player=None, live_potions=None) -> dict:
    """Use live player/potions in combat; never fall back to stale run inventory."""
    source = run if player is None else player
    hp, max_hp = source.get("hp"), source.get("max_hp")
    hp_valid = _integer(hp) and _integer(max_hp, 1) and hp <= max_hp
    deck = run.get("deck")
    deck_valid = isinstance(deck, list) and all(
        isinstance(card, dict) and isinstance(card.get("name"), str)
        and card["name"].strip() and _integer(card.get("upgrades")) for card in deck
    )
    inventory = run if live_potions is None and player is None else live_potions
    inventory = inventory if isinstance(inventory, Mapping) else {}
    slots = inventory.get("potions")
    potions_valid = inventory.get("potion_inventory_complete") is True and (
        isinstance(slots, list) and len(slots) == 5 and all(
            isinstance(slot, str) and slot.strip() and slot not in {"INVALID", "UNKNOWN"}
            for slot in slots
        )
    )
    return {
        "hp": hp if hp_valid else None,
        "max_hp": max_hp if hp_valid else None,
        "hp_fraction": hp / max_hp if hp_valid else None,
        "gold": run.get("gold") if _integer(run.get("gold")) else None,
        "deck": deepcopy(deck) if deck_valid else None,
        "deck_size": len(deck) if deck_valid else None,
        "upgraded_card_count": sum(card["upgrades"] > 0 for card in deck) if deck_valid else None,
        "potions": list(slots) if potions_valid else None,
        "potion_count": sum(slot != EMPTY_POTION for slot in slots) if potions_valid else None,
        "potion_scope": "run" if player is None else "live_battle",
        "relics": deepcopy(run.get("relics")) if isinstance(run.get("relics"), list) else None,
        "boss_identity_at_decision": UNKNOWN,
        "remaining_route_coverage": UNKNOWN,
        "deck_capabilities": {key: UNKNOWN for key in ("burst", "scaling", "draw", "block", "energy")},
        "readiness": UNKNOWN,
    }


def profile_trace(events: Iterable[dict], metadata: dict) -> dict:
    """Accept only one complete, current Train trace with matching provenance.

    The collector separately validates canonical actions against native evidence.
    This function retains decision-time facts without retroactive Boss labels.
    """
    if metadata.get("stage") != "train" or metadata.get("purpose") != "baseline_live_resource_diagnosis":
        raise ValueError("only fresh baseline Train diagnosis is accepted")
    records, encounters = [], []
    header, terminal, last_step = None, None, -1
    for event in events:
        if not isinstance(event, dict) or terminal is not None:
            raise ValueError("invalid record or data after terminal")
        kind = event.get("type")
        if header is None:
            if (kind != "diagnostic_trace_header_v1"
                    or event.get("trace_schema") != "sts1-diagnostic-trace-v1"
                    or event.get("run_metadata") != metadata
                    or event.get("live_combat_resources_required") is not True):
                raise ValueError("current Train provenance header required")
            header = event
            continue
        if kind == "diagnostic_trace_header_v1":
            raise ValueError("duplicate header")
        if kind in {"noncombat_decision_trace_v1", "encounter_started_v1"}:
            step = event.get("game_step")
            if not _integer(step) or step <= last_step:
                raise ValueError("preparation events must be ordered and unique")
            last_step = step
        if kind == "noncombat_decision_trace_v1":
            choices, selected = event.get("legal_choices"), event.get("selected_legal_action_index")
            empty_skip = (choices == [] and selected == -1 and event.get("screen_before") == "REWARDS"
                          and event.get("selected_choice") == "armg:reward:skip_empty")
            if (event.get("legal_choices_complete") is not True or not isinstance(choices, list)
                    or (not empty_skip and (not choices or not _integer(selected) or selected >= len(choices)))
                    or any(not isinstance(choice, dict) or not _integer(choice.get("index"))
                           or choice["index"] != index
                           or not isinstance(choice.get("semantics"), dict)
                           for index, choice in enumerate(choices))):
                raise ValueError("complete canonical noncombat choices required")
            before, after = event.get("state_before"), event.get("state_after")
            if not isinstance(before, dict) or not isinstance(after, dict):
                raise ValueError("decision-time snapshots required")
            records.append({
                "game_step": step, "act": before.get("act"), "floor": before.get("floor"),
                "screen": event.get("screen_before"),
                "before": resource_facts(before), "after": resource_facts(after),
                "selected_index": selected,
                "choices": [deepcopy(choice["semantics"]) for choice in choices],
                "selected": {"choice": "skip_empty"} if empty_skip else deepcopy(choices[selected]["semantics"]),
                "current_route": deepcopy(event.get("route")),
            })
        elif kind == "encounter_started_v1":
            state = event.get("state")
            if (not isinstance(state, dict) or not isinstance(state.get("run"), dict)
                    or not isinstance(state.get("player"), dict)
                    or not isinstance(state.get("enemies"), list)):
                raise ValueError("encounter snapshots required")
            names = [enemy.get("name") for enemy in state["enemies"] if isinstance(enemy, dict)]
            observed_boss = [name for name in names if isinstance(name, str)
                             and name.replace(" ", "").replace("_", "").upper() in BOSS_KEYS] or None
            encounters.append({
                "game_step": step, "act": event.get("act"), "floor": event.get("floor"),
                "observed_boss_at_entry": observed_boss,
                "resources": resource_facts(state["run"], player=state["player"],
                                            live_potions=state.get("combat_potion_inventory")),
            })
        elif kind == "terminal_trace_v1":
            if (event.get("complete") is not True or event.get("legal_actions_complete") is not True
                    or event.get("outcome") not in {"victory", "defeat"}
                    or event.get("error") is not None
                    or event.get("potion_inventory_snapshot_complete") is not True
                    or any(not _integer(event.get(key)) or event[key] != 0 for key in SAFETY)):
                raise ValueError("complete safe terminal required")
            terminal = {key: event.get(key) for key in ("outcome", "final_floor", "final_act", "final_hp")}
    if header is None or terminal is None or not encounters:
        raise ValueError("complete trace with encounters required")
    return {"schema_version": SCHEMA, "decision_records": records,
            "encounter_records": encounters, "terminal": terminal}


def public_summary(profiles: Iterable[dict]) -> dict:
    """Aggregate counters only; no seed IDs, private hashes or raw state."""
    profiles = list(profiles)
    if not profiles or any(profile.get("schema_version") != SCHEMA for profile in profiles):
        raise ValueError("validated preparation profiles required")
    decisions = [record for profile in profiles for record in profile["decision_records"]]
    encounters = [record for profile in profiles for record in profile["encounter_records"]]
    final_entries = [profile["encounter_records"][-1] for profile in profiles]
    hp_bands = Counter()
    for entry in final_entries:
        fraction = entry["resources"]["hp_fraction"]
        hp_bands["unknown" if fraction is None else "above_80_percent" if fraction > .8
                 else "below_60_percent" if fraction < .6 else "60_to_80_percent"] += 1
    return {
        "schema_version": SCHEMA, "policy_changed": False,
        "baseline_runs": len(profiles),
        "baseline_wins": sum(profile["terminal"]["outcome"] == "victory" for profile in profiles),
        "terminal_floors": dict(sorted(Counter(str(p["terminal"]["final_floor"]) for p in profiles).items())),
        "noncombat_decisions": len(decisions),
        "decision_screens": dict(sorted(Counter(str(d["screen"]) for d in decisions).items())),
        "observed_boss_entries": sum(e["observed_boss_at_entry"] is not None for e in encounters),
        "final_encounter_hp_bands": dict(sorted(hp_bands.items())),
        "unknown_decision_boss_count": len(decisions),
        "unknown_deck_capability_count": len(decisions),
        "unknown_live_potion_entry_count": sum(e["resources"]["potion_count"] is None for e in encounters),
        "candidate_selected": False, "win_rate_improvement": UNKNOWN,
    }
