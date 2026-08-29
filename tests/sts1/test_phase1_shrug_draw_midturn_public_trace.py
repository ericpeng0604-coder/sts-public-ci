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


def shrug_payload() -> dict:
    hand = [
        {"uuid": "shrug-hidden", "id": "Shrug It Off", "name": "Shrug It Off", "type": "SKILL", "cost": 1, "upgrades": 0, "is_playable": True, "has_target": False},
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
                "player": {"energy": 3, "block": 0},
                "hand": hand,
                "draw_pile": [
                    {"uuid": "draw-hidden", "id": "Strike_R", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0}
                ],
                "discard_pile": [],
                "exhaust_pile": [],
            },
        },
    }


def test_shrug_play_is_counted_as_skill_even_when_hand_stays_five() -> None:
    before = shrug_payload()
    after = deepcopy(before)
    combat = after["game_state"]["combat_state"]
    shrug = combat["hand"].pop(0)
    combat["hand"].append(combat["draw_pile"].pop(0))
    combat["discard_pile"].append(shrug)
    combat["player"]["energy"] = 2
    combat["player"]["block"] = 8

    stream = io.StringIO(
        json.dumps(menu_payload()) + "\n" + json.dumps(before) + "\n" + json.dumps(after) + "\n"
    )
    bridge = CommunicationBridge(stream, io.StringIO())
    bridge.read_state()
    bridge.read_state()
    bridge.send_command("play 1")
    bridge.read_state()

    assert len(combat["hand"]) == 5
    assert 5 - len(combat["hand"]) == 0
    assert bridge.reconstruction_aux == {
        "schema_version": "sts1-public-reconstruction-aux-v1",
        "source": "communicationmod_command_trace_v1",
        "turn": 1,
        "complete": True,
        "cards_played_this_turn": 1,
        "attacks_played_this_turn": 0,
        "skills_played_this_turn": 1,
        "cards_discarded_this_turn": 0,
    }
