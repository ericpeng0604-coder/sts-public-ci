from __future__ import annotations

from copy import deepcopy
import io
import json

from scripts.sts1.phase1_draw_midturn_bridge_reference import CommunicationBridge


def menu_payload() -> dict:
    return {
        "available_commands": ["start", "state"],
        "ready_for_command": True,
        "in_game": False,
        "game_state": None,
    }


def pommel_payload() -> dict:
    hand = [
        {"uuid": "pommel-hidden", "id": "Pommel Strike", "name": "Pommel Strike", "type": "ATTACK", "cost": 1, "upgrades": 0, "is_playable": True, "has_target": True},
        {"uuid": "strike-1", "id": "Strike_R", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0, "is_playable": True, "has_target": True},
        {"uuid": "defend-1", "id": "Defend_R", "name": "Defend", "type": "SKILL", "cost": 1, "upgrades": 0, "is_playable": True, "has_target": False},
        {"uuid": "strike-2", "id": "Strike_R", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0, "is_playable": True, "has_target": True},
        {"uuid": "defend-2", "id": "Defend_R", "name": "Defend", "type": "SKILL", "cost": 1, "upgrades": 0, "is_playable": True, "has_target": False},
    ]
    return {
        "available_commands": ["play", "end", "state"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "room_phase": "COMBAT",
            "action_phase": "WAITING_ON_USER",
            "combat_state": {
                "turn": 1,
                "player": {"energy": 3},
                "hand": hand,
                "draw_pile": [
                    {"uuid": "draw-hidden", "id": "Strike_R", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0}
                ],
                "discard_pile": [],
                "exhaust_pile": [],
            },
        },
    }


def make_bridge(*payloads: dict) -> CommunicationBridge:
    stream = io.StringIO("".join(json.dumps(item) + "\n" for item in payloads))
    return CommunicationBridge(stream, io.StringIO())


def test_pommel_play_is_counted_even_when_hand_stays_five() -> None:
    before = pommel_payload()
    after = deepcopy(before)
    combat = after["game_state"]["combat_state"]
    pommel = combat["hand"].pop(0)
    combat["hand"].append(combat["draw_pile"].pop(0))
    combat["discard_pile"].append(pommel)
    combat["player"]["energy"] = 2

    bridge = make_bridge(menu_payload(), before, after)
    bridge.read_state()
    bridge.read_state()
    bridge.send_command("play 1 0")
    bridge.read_state()

    assert len(combat["hand"]) == 5
    assert 5 - len(combat["hand"]) == 0  # old heuristic is wrong here
    assert bridge.reconstruction_aux == {
        "schema_version": "sts1-public-reconstruction-aux-v1",
        "source": "communicationmod_command_trace_v1",
        "turn": 1,
        "complete": True,
        "cards_played_this_turn": 1,
        "attacks_played_this_turn": 1,
        "skills_played_this_turn": 0,
        "cards_discarded_this_turn": 0,
    }


def test_rejected_command_never_becomes_trace_history() -> None:
    before = pommel_payload()
    rejected = deepcopy(before)
    rejected["error"] = "invalid command"

    bridge = make_bridge(menu_payload(), before, rejected)
    bridge.read_state()
    bridge.read_state()
    bridge.send_command("play 1 0")
    bridge.read_state()

    assert bridge.reconstruction_aux["cards_played_this_turn"] == 0
    assert bridge.reconstruction_aux["attacks_played_this_turn"] == 0


def test_unsupported_card_fails_closed_for_reconstruction() -> None:
    before = pommel_payload()
    before["game_state"]["combat_state"]["hand"][0].update(
        {"id": "Havoc", "name": "Havoc", "type": "SKILL", "has_target": False}
    )
    after = deepcopy(before)
    after["game_state"]["combat_state"]["player"]["energy"] = 2

    bridge = make_bridge(menu_payload(), before, after)
    bridge.read_state()
    bridge.read_state()
    bridge.send_command("play 1")
    bridge.read_state()

    assert bridge.reconstruction_aux["cards_played_this_turn"] == 1
    assert bridge.reconstruction_aux["skills_played_this_turn"] == 1
    assert bridge.reconstruction_aux["complete"] is False


def test_attaching_directly_midturn_is_not_claimed_complete() -> None:
    bridge = make_bridge(pommel_payload())
    bridge.read_state()
    assert bridge.reconstruction_aux["complete"] is False


def test_new_turn_resets_to_known_zero() -> None:
    before = pommel_payload()
    next_turn = deepcopy(before)
    next_turn["game_state"]["combat_state"]["turn"] = 2

    bridge = make_bridge(menu_payload(), before, next_turn)
    bridge.read_state()
    bridge.read_state()
    bridge.send_command("end")
    bridge.read_state()

    assert bridge.reconstruction_aux["turn"] == 2
    assert bridge.reconstruction_aux["complete"] is True
    assert bridge.reconstruction_aux["cards_played_this_turn"] == 0
