"""H23 exact legal reward Skip; deterministic, decision-time state only."""
from .card_preparation import shadow_clash_skip


def select_clash_reward_skip(index, choices, deck, card_types, *, legal_choices_complete=True):
    if legal_choices_complete is not True or not isinstance(choices, list):
        selected, reason = index, "incomplete_legal_choices"
    elif not isinstance(card_types, dict) or any(v not in {"ATTACK", "SKILL", "POWER", "CURSE", "STATUS"} for v in card_types.values()):
        selected, reason = index, "unknown_card_types"
    else:
        selected, reason = shadow_clash_skip({"screen": "REWARDS", "selected_index": index,
            "choices": choices, "before": {"deck": deck}}, card_types)
    return selected, {"enabled": True, "overridden": selected != index,
                      "recommended_index": index, "selected_index": selected, "reason": reason}


def validate_override(event, card_types):
    choices = event.get("legal_choices")
    if not isinstance(choices, list) or event.get("screen_before") != "REWARDS":
        raise ValueError("H23 reward legal choices required")
    semantics = [c.get("semantics") if isinstance(c, dict) else None for c in choices]
    expected, detail = select_clash_reward_skip(event.get("recommended_legal_action_index"),
        semantics, event.get("state_before", {}).get("deck"), card_types,
        legal_choices_complete=event.get("legal_choices_complete"))
    if (detail["overridden"] is not True or event.get("clash_reward_override") is not True
            or expected != event.get("selected_legal_action_index")
            or detail != event.get("clash_reward_intervention")
            or detail["reason"] != event.get("clash_reward_reason")):
        raise ValueError("H23 override differs from frozen selector")


def _prefix_view(event):
    kind = event.get("type")
    if kind == "noncombat_decision_trace_v1":
        return {k: event.get(k) for k in ("type", "game_step", "screen_before", "legal_choices",
            "recommended_legal_action_index", "selected_legal_action_index", "state_before", "state_after")}
    if kind == "combat_decision_trace_v1":
        return {k: event.get(k) for k in ("type", "game_step", "encounter_index", "battle_step",
            "state_before_signature_sha256", "mcts_recommended_action", "selected_action",
            "canonical_native_legal_actions", "selected_native_action_index")}
    if kind == "combat_action_applied_v1":
        return {k: event.get(k) for k in ("type", "game_step", "encounter_index", "battle_step",
            "state_after_signature_sha256", "selected_action")}
    if kind == "encounter_started_v1":
        return {k: event.get(k) for k in ("type", "game_step", "encounter_index", "state")}
    return None


def effective_override(parent_events, candidate_events):
    """One first changed reward with identical recorded initial state/action prefix."""
    parent = [e for e in parent_events if _prefix_view(e) is not None]
    candidate = [e for e in candidate_events if _prefix_view(e) is not None]
    for p, c in zip(parent, candidate):
        if c.get("clash_reward_override") is True:
            return (p.get("type") == c.get("type") == "noncombat_decision_trace_v1"
                and p.get("game_step") == c.get("game_step")
                and p.get("screen_before") == c.get("screen_before") == "REWARDS"
                and isinstance(c.get("state_before"), dict)
                and p.get("state_before") == c["state_before"]
                and p.get("legal_choices") == c.get("legal_choices")
                and type(p.get("selected_legal_action_index")) is int
                and p["selected_legal_action_index"] == c.get("recommended_legal_action_index")
                and p.get("recommended_legal_action_index") == c.get("recommended_legal_action_index")
                and c.get("selected_legal_action_index") != c.get("recommended_legal_action_index")
                and isinstance(c.get("state_after"), dict)
                and isinstance(p.get("state_after"), dict)
                and c["state_after"] != p["state_after"]
                and isinstance(c["state_before"].get("deck"), list)
                and isinstance(c["state_after"].get("deck"), list)
                and isinstance(p["state_after"].get("deck"), list)
                and len(c["state_after"]["deck"]) == len(c["state_before"]["deck"])
                and len(p["state_after"]["deck"]) == len(c["state_before"]["deck"]) + 1)
        if _prefix_view(p) != _prefix_view(c):
            return False
    return False
