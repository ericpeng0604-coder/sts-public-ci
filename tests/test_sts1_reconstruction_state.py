from roguelike_ai.sts1_teacher.contract import DecisionContext, PUBLIC_STATE_SCHEMA
from roguelike_ai.sts1_teacher.reconstruction import assess_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities


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
        "enemies": [{"index": 0, "name": "Cultist", "hp": 48, "max_hp": 48, "block": 0, "intent": "DARK_STRIKE", "powers": []}],
        "turn": 1,
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
        "legal_actions": [{"kind": "play_card", "hand_index": 1, "target_index": 0}, {"kind": "end_turn"}],
    }


def run_state() -> dict:
    return {
        "reconstruction": {
            "schema_version": "sts1-public-reconstruction-v1",
            "public_card_instance_state_complete": False,
            "public_relic_state_complete": True,
            "public_potion_state_complete": True,
            "public_enemy_state_complete": False,
        }
    }


def test_reconstruction_metadata_does_not_change_decision_signature() -> None:
    state = public_state()
    before = DecisionContext.from_public_state(state)
    after_state = attach_reconstruction_capabilities(state, run_state=run_state())
    after = DecisionContext.from_public_state(after_state)

    assert before.decision_signature == after.decision_signature
    assert after_state["reconstruction"]["public_card_instance_state_complete"] is True
    assert after_state["reconstruction"]["public_relic_state_complete"] is True
    assert after_state["reconstruction"]["public_potion_state_complete"] is True
    assert after_state["reconstruction"]["public_enemy_state_complete"] is False


def test_enemy_gap_still_blocks_full_reconstruction() -> None:
    state = attach_reconstruction_capabilities(public_state(), run_state=run_state())
    admission = assess_public_reconstruction(state)
    assert admission.allowed is False
    assert "capability_not_proven:public_enemy_state_complete" in admission.reasons
    assert "capability_not_proven:public_card_instance_state_complete" not in admission.reasons


def test_unsupported_card_keeps_card_capability_false() -> None:
    state = public_state()
    state["hand"][0]["id"] = "RAMPAGE"
    attached = attach_reconstruction_capabilities(state, run_state=run_state())
    marker = attached["reconstruction"]

    assert marker["public_card_instance_state_complete"] is False
    assert "card_unsupported_v1:RAMPAGE" in marker["card_admission_reasons"]
