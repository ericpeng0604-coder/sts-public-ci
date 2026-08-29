from roguelike_ai.sts1_teacher.contract import DecisionContext, PUBLIC_STATE_SCHEMA
from roguelike_ai.sts1_teacher.reconstruction import assess_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities


def jaw_worm_public_state() -> dict:
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "simulator",
        "hp": 70,
        "max_hp": 80,
        "block": 0,
        "energy": 3,
        "hand": [
            {"position": 1, "id": "STRIKE_RED", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0, "has_target": True},
            {"position": 2, "id": "DEFEND_RED", "name": "Defend", "type": "SKILL", "cost": 1, "upgrades": 0, "has_target": False},
        ],
        "draw_pile": [{"id": "BASH", "name": "Bash", "type": "ATTACK", "cost": 2, "upgrades": 0, "has_target": True}],
        "discard_pile": [],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 40, "max_hp": 40, "block": 0, "intent": "CHOMP", "intent_damage": 11, "intent_hits": 1, "is_gone": False, "powers": []}],
        "turn": 0,
        "combat_active": True,
        "relics": [{"id": "BURNING_BLOOD", "counter": -1}],
        "potions": [
            {"index": 0, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
            {"index": 1, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
            {"index": 2, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
        ],
        "gold": 99,
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
            {"kind": "play_card", "hand_index": 2},
            {"kind": "end_turn"},
        ],
    }


def run_state_capabilities() -> dict:
    return {
        "reconstruction": {
            "schema_version": "sts1-public-reconstruction-v1",
            "public_player_state_complete": False,
            "public_card_instance_state_complete": False,
            "public_relic_state_complete": True,
            "public_potion_state_complete": True,
            "public_enemy_state_complete": False,
        }
    }


def test_jaw_worm_basic_combat_reaches_full_reconstruction_admission() -> None:
    original = jaw_worm_public_state()
    original_signature = DecisionContext.from_public_state(original).decision_signature
    state = attach_reconstruction_capabilities(original, run_state=run_state_capabilities())
    marker = state["reconstruction"]
    admission = assess_public_reconstruction(state)
    assert marker["public_player_state_complete"] is True
    assert marker["public_card_instance_state_complete"] is True
    assert marker["public_relic_state_complete"] is True
    assert marker["public_potion_state_complete"] is True
    assert marker["public_enemy_state_complete"] is True
    assert admission.allowed is True
    assert admission.reasons == ()
    assert admission.decision_signature == original_signature
    assert DecisionContext.from_public_state(state).decision_signature == original_signature


def test_same_state_with_unsupported_enemy_fails_closed() -> None:
    state = jaw_worm_public_state()
    state["enemies"][0]["name"] = "Fungi Beast"
    attached = attach_reconstruction_capabilities(state, run_state=run_state_capabilities())
    admission = assess_public_reconstruction(attached)
    assert attached["reconstruction"]["public_enemy_state_complete"] is False
    assert admission.allowed is False
    assert "capability_not_proven:public_enemy_state_complete" in admission.reasons
