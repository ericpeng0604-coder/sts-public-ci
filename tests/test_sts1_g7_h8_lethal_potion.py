from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

from roguelike_ai.sts1_phase3.simulator import _apply_lethal_potion_rescue

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sts1"))
import sts1_g7_h8_lethal_potion_eval as h8  # noqa: E402


class _Action:
    def __init__(self, action_type: str, source_idx: int = -1, target_idx: int = -1) -> None:
        self.action_type = action_type
        self.source_idx = source_idx
        self.target_idx = target_idx


def _setup(
    *,
    hp: int = 8,
    max_hp: int = 50,
    block: int = 0,
    damage: int = 10,
    hits: int = 1,
    potions: list[str] | None = None,
    relics: tuple[str, ...] = (),
    statuses: tuple[str, ...] = (),
    enemy_weak: int = 0,
    enemy_strength: int = 0,
    enemy_poison: int = 0,
    mcts_action: _Action | None = None,
    legal_potion_slots: tuple[int, ...] = (0,),
):
    player_status = SimpleNamespace(
        VULNERABLE="VULNERABLE",
        INTANGIBLE="INTANGIBLE",
        REGEN="REGEN",
        METALLICIZE="METALLICIZE",
        PLATED_ARMOR="PLATED_ARMOR",
    )
    active_statuses = set(statuses)
    player = SimpleNamespace(
        cur_hp=hp,
        max_hp=max_hp,
        block=block,
        has_status=lambda status: status in active_statuses,
    )
    enemy = SimpleNamespace(
        cur_hp=30,
        alive=True,
        strength=enemy_strength,
        weak=enemy_weak,
        poison=enemy_poison,
        intent_damage=lambda _battle: SimpleNamespace(damage=damage, attack_count=hits),
    )
    battle = SimpleNamespace(player=player, monsters=[enemy])
    inventory = potions or ["BLOCK_POTION", "EMPTY", "EMPTY", "EMPTY", "EMPTY"]
    game_context = SimpleNamespace(
        relics=[SimpleNamespace(id=name) for name in relics],
        potions=inventory,
    )
    sts = SimpleNamespace(PlayerStatus=player_status)
    recommended = mcts_action or _Action("END_TURN")
    legal_actions = [recommended] + [_Action("POTION", slot) for slot in legal_potion_slots]
    return recommended, legal_actions, battle, game_context, sts


def _apply(**kwargs):
    recommended, legal_actions, battle, game_context, sts = _setup(**kwargs)
    return _apply_lethal_potion_rescue(
        recommended,
        legal_actions,
        hand=(),
        battle=battle,
        game_context=game_context,
        sts=sts,
    )


def test_block_potion_overrides_end_turn_only_when_it_prevents_lethal_attack() -> None:
    action, changed, reason = _apply()

    assert action.action_type == "POTION"
    assert changed is True
    assert reason == "override"


def test_blood_potion_uses_pinned_battlecontext_heal_amount() -> None:
    action, changed, reason = _apply(
        hp=3,
        max_hp=40,
        damage=10,
        potions=["BLOOD_POTION", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
        relics=("SACRED_BARK",),
    )

    assert action.action_type == "POTION"
    assert changed is True
    assert reason == "override"


def test_magic_flower_and_mark_of_the_bloom_match_native_heal_rules() -> None:
    action, changed, _ = _apply(
        hp=5,
        max_hp=40,
        damage=10,
        potions=["BLOOD_POTION", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
        relics=("MARK_OF_THE_BLOOM",),
    )
    assert action.action_type == "END_TURN"
    assert changed is False

    action, changed, reason = _apply(
        hp=5,
        max_hp=40,
        damage=10,
        potions=["BLOOD_POTION", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
        relics=("MAGIC_FLOWER",),
    )
    assert action.action_type == "POTION"
    assert changed is True
    assert reason == "override"


def test_potion_must_be_legal_and_strictly_improve_projected_hp() -> None:
    action, changed, reason = _apply(legal_potion_slots=())
    assert action.action_type == "END_TURN"
    assert changed is False
    assert reason == "no_legal_potion_improves_projected_hp"

    action, changed, reason = _apply(hp=11, damage=10)
    assert action.action_type == "POTION"
    assert changed is True
    assert reason == "override"

    action, changed, reason = _apply(hp=30, block=10, damage=10)
    assert action.action_type == "END_TURN"
    assert changed is False
    assert reason == "no_legal_potion_improves_projected_hp"


def test_nonlethal_blood_potion_is_used_only_for_real_post_attack_hp_gain() -> None:
    action, changed, reason = _apply(
        hp=20,
        max_hp=50,
        damage=10,
        potions=["BLOOD_POTION", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
    )
    assert action.action_type == "POTION"
    assert changed is True
    assert reason == "override"

    action, changed, reason = _apply(
        hp=50,
        max_hp=50,
        damage=10,
        potions=["BLOOD_POTION", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
    )
    assert action.action_type == "END_TURN"
    assert changed is False
    assert reason == "no_legal_potion_improves_projected_hp"


def test_only_mcts_end_turn_and_one_living_enemy_are_supported() -> None:
    action, changed, reason = _apply(mcts_action=_Action("CARD"))
    assert action.action_type == "CARD"
    assert changed is False
    assert reason == "not_end_turn"

    recommended, legal, battle, game_context, sts = _setup()
    battle.monsters.append(SimpleNamespace(cur_hp=1, alive=True))
    action, changed, reason = _apply_lethal_potion_rescue(
        recommended,
        legal,
        hand=(),
        battle=battle,
        game_context=game_context,
        sts=sts,
    )
    assert action.action_type == "END_TURN"
    assert changed is False
    assert reason == "not_exactly_one_living_enemy"


def test_unsupported_damage_or_end_turn_relics_fail_closed() -> None:
    for relic in ("FOSSILIZED_HELIX", "LIZARD_TAIL", "ORICHALCUM", "TORII", "TUNGSTEN_ROD"):
        action, changed, reason = _apply(relics=(relic,))
        assert action.action_type == "END_TURN"
        assert changed is False
        assert reason == "unsupported_damage_or_end_turn_relic"


def test_attack_modifiers_and_block_are_projected_per_hit() -> None:
    action, changed, reason = _apply(
        hp=7,
        max_hp=40,
        damage=10,
        enemy_weak=1,
        statuses=("VULNERABLE",),
        relics=("PAPER_KRANE", "ODD_MUSHROOM"),
    )
    assert action.action_type == "POTION"
    assert changed is True
    assert reason == "override"

    action, changed, reason = _apply(hp=10, block=8, damage=10, hits=2)
    assert action.action_type == "POTION"
    assert changed is True
    assert reason == "override"


def test_incomplete_and_poisoned_enemy_states_fail_closed() -> None:
    action, changed, reason = _apply(enemy_poison=1)
    assert action.action_type == "END_TURN"
    assert changed is False
    assert reason == "enemy_may_die_before_attack"

    recommended, legal, battle, game_context, sts = _setup()
    battle.monsters[0].poison = None
    action, changed, reason = _apply_lethal_potion_rescue(
        recommended,
        legal,
        hand=(),
        battle=battle,
        game_context=game_context,
        sts=sts,
    )
    assert action.action_type == "END_TURN"
    assert changed is False
    assert reason == "incomplete_enemy_modifiers"


def test_checkpoint_materialization_reads_only_pinned_g7_member(tmp_path, monkeypatch) -> None:
    checkpoint_bytes = b"test G7 checkpoint"
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    monkeypatch.setattr(h8, "G7_SHA256", checkpoint_hash)
    archive_path = tmp_path / "actions.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("offline-champion.pt", checkpoint_bytes)
        archive.writestr("candidate-A.pt", b"do not load")

    destination = tmp_path / "private" / "pinned-g7" / "offline-champion.pt"
    materialized = h8._materialize_g7_checkpoint(archive_path, destination)

    assert materialized == destination.resolve()
    assert materialized.read_bytes() == checkpoint_bytes
    assert [path.name for path in materialized.parent.iterdir()] == ["offline-champion.pt"]
    assert h8._materialize_g7_checkpoint(archive_path, destination) == materialized


def test_checkpoint_materialization_rejects_adapter_sidecar(tmp_path, monkeypatch) -> None:
    checkpoint_bytes = b"test G7 checkpoint"
    monkeypatch.setattr(h8, "G7_SHA256", hashlib.sha256(checkpoint_bytes).hexdigest())
    archive_path = tmp_path / "actions.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("offline-champion.pt", checkpoint_bytes)

    destination = tmp_path / "private" / "pinned-g7" / "offline-champion.pt"
    destination.parent.mkdir(parents=True)
    Path(str(destination) + ".adapter.pt").write_bytes(b"unexpected sidecar")

    with pytest.raises(h8.EvaluationIntegrityError, match="adapter sidecar"):
        h8._materialize_g7_checkpoint(archive_path, destination)


def test_trace_coverage_uses_h7_validated_counter_names_and_fails_closed() -> None:
    totals = h8._empty_trace_coverage()
    h8._accumulate_trace_coverage(totals, {
        "combat_decision_count": 3,
        "encounter_count": 2,
        "noncombat_decision_count": 5,
        "route_decision_count": 4,
        "potion_snapshot_count": 21,
    })
    h8._accumulate_trace_coverage(totals, {
        "combat_decision_count": 1,
        "encounter_count": 1,
        "noncombat_decision_count": 2,
        "route_decision_count": 3,
        "potion_snapshot_count": 9,
    })
    assert totals == {
        "combat_decision_count": 4,
        "encounter_count": 3,
        "noncombat_decision_count": 7,
        "route_decision_count": 7,
        "potion_snapshot_count": 30,
    }

    with pytest.raises(h8.EvaluationIntegrityError, match="counter is invalid"):
        h8._accumulate_trace_coverage(totals, {"combat_decisions": 1})
