from types import SimpleNamespace

from roguelike_ai.sts1_teacher.public_run_state import simulator_public_run_state
from roguelike_ai.sts1_teacher.reconstruction import (
    PUBLIC_RECONSTRUCTION_SCHEMA,
    assess_public_reconstruction,
)


def test_simulator_public_run_state_projects_visible_relics_and_potions_only() -> None:
    gc = SimpleNamespace(
        gold=99,
        floor_num=7,
        act=1,
        ascension=0,
        cc="CharacterClass.IRONCLAD",
        relics=[SimpleNamespace(id="RelicId.BURNING_BLOOD", data=-1)],
        potions=[
            {"index": 0, "id": "WEAK_POTION", "name": "Weak Potion", "empty": False},
            {"index": 1, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
        ],
        seed=123456789,
        potionRng=object(),
        cardRng=object(),
    )

    projected = simulator_public_run_state(gc)

    assert projected["gold"] == 99
    assert projected["floor"] == 7
    assert projected["act"] == 1
    assert projected["character"] == "IRONCLAD"
    assert projected["ascension_level"] == 0
    assert projected["relics"] == [{"id": "BURNING_BLOOD", "counter": -1}]
    assert projected["potions"] == [
        {"index": 0, "id": "WEAK_POTION", "name": "Weak Potion", "empty": False},
        {"index": 1, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
    ]
    assert "seed" not in projected
    assert all("rng" not in key.lower() for key in projected)


def test_run_state_marks_only_source_verified_surfaces_complete() -> None:
    gc = SimpleNamespace(
        gold=0,
        floor_num=1,
        act=1,
        ascension=0,
        cc="IRONCLAD",
        relics=[],
        potions=[],
    )
    marker = simulator_public_run_state(gc)["reconstruction"]

    assert marker == {
        "schema_version": PUBLIC_RECONSTRUCTION_SCHEMA,
        "public_player_state_complete": False,
        "public_card_instance_state_complete": False,
        "public_relic_state_complete": True,
        "public_potion_state_complete": True,
        "public_enemy_state_complete": False,
    }


def test_partial_run_state_can_never_unlock_reconstruction_by_itself() -> None:
    gc = SimpleNamespace(
        gold=0,
        floor_num=1,
        act=1,
        ascension=0,
        cc="IRONCLAD",
        relics=[],
        potions=[],
    )
    state = simulator_public_run_state(gc)
    state.update(
        {
            "schema_version": "sts1-public-state-v1",
            "source": "simulator",
            "hp": 70,
            "max_hp": 80,
            "block": 0,
            "energy": 3,
            "hand": [],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "powers": [],
            "enemies": [],
            "turn": 1,
            "combat_active": True,
            "legal_actions": [{"kind": "end_turn"}],
        }
    )

    admission = assess_public_reconstruction(state)
    assert admission.allowed is False
    assert "capability_not_proven:public_player_state_complete" in admission.reasons
    assert "capability_not_proven:public_card_instance_state_complete" in admission.reasons
    assert "capability_not_proven:public_enemy_state_complete" in admission.reasons
