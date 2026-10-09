from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sts1"))

import sts1_g7_h16_lethal_defend_eval as h16_runner
from roguelike_ai.sts1_phase3.simulator import _apply_lethal_intent_defend_rescue


class _Card:
    def __init__(self, card_id: str, upgraded: bool | None = False) -> None:
        self.id = card_id
        self.upgraded = upgraded


class _Action:
    action_type = "CARD"

    def __init__(self, source_idx: int) -> None:
        self.source_idx = source_idx
        self.target_idx = -1


class _Enemy:
    cur_hp = 20
    alive = True

    def __init__(self, damage: int | None, hits: int = 1) -> None:
        self.damage = damage
        self.hits = hits

    def intent_damage(self, _battle: object) -> object:
        if self.damage is None:
            raise RuntimeError("unknown intent")
        return SimpleNamespace(damage=self.damage, attack_count=self.hits)


def _battle(*, hp: int = 8, block: int = 2, damage: int | None = 16) -> object:
    return SimpleNamespace(
        player=SimpleNamespace(cur_hp=hp, block=block),
        monsters=[_Enemy(damage)],
    )


def test_h16_selects_strongest_legal_defend_that_closes_visible_deficit() -> None:
    hand = [_Card("DEFEND_R"), _Card("DEFEND_G", upgraded=True), _Card("BASH")]
    recommended = _Action(2)
    selected, overridden, reason, detail = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1), _Action(2)],
        hand=hand,
        battle=_battle(hp=8, block=2, damage=16),
    )

    assert overridden is True
    assert reason == "legal_defend_closes_visible_lethal_deficit"
    assert selected is not recommended
    assert selected.source_idx == 1
    assert detail["projected_deficit"] == 6
    assert detail["selected_defend_base_block"] == 8
    assert detail["selected_defend_hand_index"] == 2


def test_h16_tie_breaks_by_lowest_hand_index_not_legal_action_order() -> None:
    hand = [_Card("DEFEND_R", upgraded=True), _Card("DEFEND_G", upgraded=True), _Card("BASH")]
    recommended = _Action(2)
    selected, overridden, _, detail = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(1), _Action(0), _Action(2)],
        hand=hand,
        battle=_battle(),
    )

    assert overridden is True
    assert selected.source_idx == 0
    assert detail["selected_defend_hand_index"] == 1


def test_h16_keeps_g7_action_when_intent_is_not_lethal() -> None:
    hand = [_Card("DEFEND_R"), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(hp=8, block=2, damage=9),
    )

    assert selected is recommended
    assert overridden is False
    assert reason == "visible_intent_not_lethal"


def test_h16_keeps_g7_defend_recommendation() -> None:
    hand = [_Card("DEFEND_R"), _Card("BASH")]
    recommended = _Action(0)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(),
    )

    assert selected is recommended
    assert overridden is False
    assert reason == "g7_already_selected_defend"


def test_h16_fails_closed_when_defend_cannot_close_deficit_or_identity_is_unknown() -> None:
    hand = [_Card("DEFEND_R"), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(hp=8, block=0, damage=20),
    )
    assert selected is recommended
    assert overridden is False
    assert reason == "defend_block_does_not_close_deficit"

    unknown_upgrade = [_Card("DEFEND_R", upgraded=None), _Card("BASH")]
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=unknown_upgrade,
        battle=_battle(),
    )
    assert selected is recommended
    assert overridden is False
    assert reason == "defend_upgrade_status_unknown"


def test_h16_fails_closed_when_visible_intent_is_unavailable() -> None:
    hand = [_Card("DEFEND_R"), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(damage=None),
    )

    assert selected is recommended
    assert overridden is False
    assert reason == "unknown_attack_intent"


def test_h17_uses_registered_hypothesis_three_pool_and_manifest() -> None:
    assert h16_runner.TRIAL_POOL_ROLE["h16"]["train"] == "train_hypothesis_2"
    assert h16_runner.TRIAL_POOL_ROLE["h17"]["train"] == "train_hypothesis_3"
    assert h16_runner.EXPECTED_TRIAL_POOL_MANIFESTS["h17"]["train"] == (
        "fe8f0bc33ba50bac941ddb40278775cb0f36934df027b9e2c1df595b7b3d4b7d"
    )
    assert h16_runner.EXPECTED_TRIAL_POOL_MANIFESTS["h17"]["probe"] == (
        "56de6fce082c49227c24652ab152ba331aa0730fc087366d068ba5c33902e5d0"
    )
    assert h16_runner.EXPECTED_TRIAL_POOL_MANIFESTS["h17"]["dev"] == (
        "8636a6121e8123e2f8746dfeb57c6422ee15cf425c403ab4f276ca10aaf29d14"
    )


def test_h17_private_paths_are_separate_from_h16_and_repo(tmp_path) -> None:
    pools_dir = tmp_path / "round-008-seeds" / "pools"
    h17_output = tmp_path / "round-008-h17-train-20261009"
    h17_usage = pools_dir.parent / "h17-usage-private.jsonl"

    h16_runner._validate_private_paths(pools_dir, "h17", "train", h17_output, h17_usage)

    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_private_paths(
            pools_dir,
            "h17",
            "train",
            tmp_path / "round-008-h16-train-20261009",
            h17_usage,
        )


def test_h17_train_transition_requires_exact_pool_allocation_history() -> None:
    allocation = deepcopy(h16_runner.EXPECTED_H17_ALLOCATION)
    candidate_commit = "a" * 40
    identities = {key: f"{key}-sha256" for key in (
        "simulator_policy_source_sha256",
        "candidate_evaluator_sha256",
        "simulator_binding_sha256",
        "armg_source_sha256",
        "armg_vocab_sha256",
        "g7_checkpoint_sha256",
    )}

    h16_runner._validate_transition(
        trial_id="h17",
        stage="train",
        events=[allocation],
        candidate_commit=candidate_commit,
        identities=identities,
    )

    altered = deepcopy(allocation)
    altered["prior_h16"]["probe"]["status"] = "COMPLETE"
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_transition(
            trial_id="h17",
            stage="train",
            events=[altered],
            candidate_commit=candidate_commit,
            identities=identities,
        )


def test_h17_probe_requires_its_own_eligible_train_summary() -> None:
    candidate_commit = "b" * 40
    identities = {key: f"{key}-sha256" for key in (
        "simulator_policy_source_sha256",
        "candidate_evaluator_sha256",
        "simulator_binding_sha256",
        "armg_source_sha256",
        "armg_vocab_sha256",
        "g7_checkpoint_sha256",
    )}
    summary = {
        "record_type": "h17_stage_summary",
        "trial_id": "h17",
        "stage": "train",
        "status": "COMPLETE",
        "candidate_commit": candidate_commit,
        "advance_eligible": True,
        **identities,
    }

    h16_runner._validate_transition(
        trial_id="h17",
        stage="probe",
        events=[deepcopy(h16_runner.EXPECTED_H17_ALLOCATION), summary],
        candidate_commit=candidate_commit,
        identities=identities,
    )

    wrong_trial = {**summary, "trial_id": "h16"}
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_transition(
            trial_id="h17",
            stage="probe",
            events=[deepcopy(h16_runner.EXPECTED_H17_ALLOCATION), wrong_trial],
            candidate_commit=candidate_commit,
            identities=identities,
        )


def test_h16_legacy_stage_summary_remains_usable_without_trial_id() -> None:
    candidate_commit = "c" * 40
    identities = {key: f"{key}-sha256" for key in (
        "simulator_policy_source_sha256",
        "candidate_evaluator_sha256",
        "simulator_binding_sha256",
        "armg_source_sha256",
        "armg_vocab_sha256",
        "g7_checkpoint_sha256",
    )}
    legacy_summary = {
        "record_type": "h16_stage_summary",
        "stage": "train",
        "status": "COMPLETE",
        "candidate_commit": candidate_commit,
        "advance_eligible": True,
        **identities,
    }

    assert h16_runner._stage_summary([legacy_summary], "train", "h16") == legacy_summary
    h16_runner._validate_transition(
        trial_id="h16",
        stage="probe",
        events=[legacy_summary],
        candidate_commit=candidate_commit,
        identities=identities,
    )
