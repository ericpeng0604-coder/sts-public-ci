"""H24 bounded draw-card coverage, not a complete deck-readiness measure."""
from .card_preparation import base_name, card_facts
from .clash_reward_skip import _prefix_view
from collections import Counter
import re

DRAW_CARDS = frozenset({"Pommel Strike", "Shrug It Off", "Battle Trance"})


def select_draw_reward(index, choices, deck, card_types, *, legal_choices_complete=True):
    selected, reason = index, "incomplete_input"
    if (legal_choices_complete is True and isinstance(choices, list)
            and type(index) is int and 0 <= index < len(choices)
            and isinstance(card_types, dict) and isinstance(deck, list) and deck
            and all(isinstance(c, dict) and base_name(c.get("name")) in card_types for c in deck)
            and all(isinstance(c, dict) and c.get("kind") == "card"
                    and (c == {"kind": "card", "choice": "skip"}
                         or c.get("choice") == "card" and base_name(c.get("card_name")) in card_types)
                    for c in choices)):
        recommended = choices[index]
        count = sum(base_name(c.get("name")) in DRAW_CARDS for c in deck)
        targets = [i for i, c in enumerate(choices) if c.get("choice") == "card"
                   and c.get("card_name") in DRAW_CARDS]
        reviewed = all(isinstance(card_facts(c, card_types)["effects"], dict)
                       for c in deck if base_name(c.get("name")) in DRAW_CARDS)
        if recommended.get("choice") != "card":
            reason = "preserve_skip"
        elif base_name(recommended.get("card_name")) in DRAW_CARDS:
            reason = "already_draw"
        elif not reviewed:
            reason = "unknown_draw_upgrade"
        elif count >= 2:
            reason = "two_target_draw_cards_present"
        elif len(targets) != 1:
            reason = "ambiguous_or_missing_draw_option"
        else:
            selected, reason = targets[0], "unique_draw_under_two"
    return selected, {"enabled": True, "overridden": selected != index,
                      "recommended_index": index, "selected_index": selected, "reason": reason}


def validate_override(event, card_types):
    choices = event.get("legal_choices")
    if not isinstance(choices, list) or event.get("screen_before") != "REWARDS":
        raise ValueError("H24 reward choices required")
    selected, detail = select_draw_reward(event.get("recommended_legal_action_index"),
        [c.get("semantics") if isinstance(c, dict) else None for c in choices],
        event.get("state_before", {}).get("deck"), card_types,
        legal_choices_complete=event.get("legal_choices_complete"))
    if (not detail["overridden"] or event.get("draw_reward_override") is not True
            or selected != event.get("selected_legal_action_index")
            or detail != event.get("draw_reward_intervention")
            or detail["reason"] != event.get("draw_reward_reason")):
        raise ValueError("H24 override differs from frozen selector")


def _accepted(before, after, card_name):
    if not isinstance(card_name, str):
        return False
    suffix = re.search(r"\+(\d*)$", card_name)
    upgrades = int(suffix[1] or "1") if suffix else 0
    def cards(deck):
        if any(not isinstance(c, dict) or not isinstance(c.get("name"), str)
               or type(c.get("upgrades")) is not int or c["upgrades"] < 0 for c in deck):
            raise ValueError("incomplete deck card identity")
        return Counter((c["name"], c["upgrades"]) for c in deck)
    try:
        return cards(after) == cards(before) + Counter({(card_name, upgrades): 1})
    except ValueError:
        return False


def effective_override(parent_events, candidate_events):
    parent = [e for e in parent_events if _prefix_view(e) is not None]
    candidate = [e for e in candidate_events if _prefix_view(e) is not None]
    for p, c in zip(parent, candidate):
        if c.get("draw_reward_override") is True:
            before, pa, ca = c.get("state_before"), p.get("state_after"), c.get("state_after")
            if not all(isinstance(x, dict) and isinstance(x.get("deck"), list) for x in (before, pa, ca)):
                return False
            choices = c.get("legal_choices")
            pi, ci = p.get("selected_legal_action_index"), c.get("selected_legal_action_index")
            if (not isinstance(choices, list) or any(type(i) is not int or not 0 <= i < len(choices) for i in (pi, ci))
                    or any(not isinstance(choices[i], dict) or not isinstance(choices[i].get("semantics"), dict) for i in (pi, ci))):
                return False
            return (p.get("type") == c.get("type") == "noncombat_decision_trace_v1"
                and p.get("game_step") == c.get("game_step")
                and p.get("screen_before") == c.get("screen_before") == "REWARDS"
                and p.get("state_before") == before
                and p.get("legal_choices") == c.get("legal_choices")
                and type(p.get("selected_legal_action_index")) is int
                and p["selected_legal_action_index"] == c.get("recommended_legal_action_index")
                and p.get("recommended_legal_action_index") == c.get("recommended_legal_action_index")
                and c.get("selected_legal_action_index") != c.get("recommended_legal_action_index")
                and pa["deck"] != ca["deck"]
                and _accepted(before["deck"], pa["deck"], choices[pi]["semantics"].get("card_name"))
                and _accepted(before["deck"], ca["deck"], choices[ci]["semantics"].get("card_name")))
        if _prefix_view(p) != _prefix_view(c):
            return False
    return False
