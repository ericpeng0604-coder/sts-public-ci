from __future__ import annotations

import json

import pytest

from roguelike_ai.sts1_phase3.simulator import _apply_low_hp_emergency_potion
from scripts.sts1.sts1_g7_h3_emergency_potion_eval import (
    EvaluationIntegrityError,
    _check_trace,
)
from scripts.sts1.sts1_g7_h3_stage_eval import (
    STAGES,
    StageEvaluationError,
    _paired_summary,
    _seed_contract_kwargs,
    _stage_spec,
)


class _Action:
    def __init__(
        self,
        action_type: str,
        source_idx: int = -1,
        target_idx: int = -1,
    ) -> None:
        self.action_type = action_type
        self.source_idx = source_idx
        self.target_idx = target_idx


def _apply(
    recommended: _Action,
    legal: list[_Action],
    *,
    hp: int | None = 20,
    max_hp: int | None = 50,
    threshold: float = 0.5,
) -> tuple[object, bool]:
    return _apply_low_hp_emergency_potion(
        recommended,
        legal,
        hand=[],
        player_hp=hp,
        player_max_hp=max_hp,
        hp_ratio_threshold=threshold,
    )


def test_uses_lowest_legal_potion_slot_at_inclusive_threshold() -> None:
    card = _Action("CARD", source_idx=0)
    slot_one = _Action("POTION", source_idx=1)
    slot_zero = _Action("POTION", source_idx=0)

    selected, overridden = _apply(card, [slot_one, slot_zero, card], hp=25)

    assert selected is slot_zero
    assert overridden is True


@pytest.mark.parametrize(
    ("hp", "max_hp", "legal", "recommended"),
    [
        (26, 50, [_Action("CARD", source_idx=0), _Action("POTION", source_idx=0)], 0),
        (20, 0, [_Action("CARD", source_idx=0), _Action("POTION", source_idx=0)], 0),
        (None, 50, [_Action("CARD", source_idx=0), _Action("POTION", source_idx=0)], 0),
        (20, 50, [_Action("CARD", source_idx=0)], 0),
        (20, 50, [_Action("CARD", source_idx=0), _Action("POTION", source_idx=0)], 1),
    ],
)
def test_fails_closed_or_keeps_recommended_action(
    hp: int | None,
    max_hp: int | None,
    legal: list[_Action],
    recommended: int,
) -> None:
    selected, overridden = _apply(legal[recommended], legal, hp=hp, max_hp=max_hp)

    assert selected is legal[recommended]
    assert overridden is False


def _trace_event(*, hp: int = 20, selected_index: int = 0, override: bool = True) -> dict:
    legal = [
        {"kind": "use_potion", "potion_index": 0},
        {"kind": "play_card", "hand_index": 0},
    ]
    return {
        "type": "combat_decision_trace_v1",
        "mcts_sims": 2000,
        "legal_actions_complete": True,
        "canonical_native_legal_actions": legal,
        "public_state": {"hp": hp, "max_hp": 50},
        "mcts_recommended_action": legal[1],
        "selected_native_action_index": selected_index,
        "selected_action": legal[selected_index],
        "emergency_potion_override": override,
    }


def _write_trace(path, event: dict) -> None:
    records = [
        event,
        {
            "type": "terminal_trace_v1",
            "complete": True,
            "legal_actions_complete": True,
            "illegal_action_count": 0,
            "timeout_count": 0,
            "crash_count": 0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")


def test_trace_validator_accepts_only_the_frozen_low_hp_legal_override(tmp_path) -> None:
    trace = tmp_path / "candidate.trace.ndjson"
    _write_trace(trace, _trace_event())

    result = _check_trace(trace, candidate=True, expected_overrides=1)

    assert result["overrides"] == 1
    assert result["low_hp_potion_opportunities"] == 1


def test_trace_validator_rejects_override_above_threshold(tmp_path) -> None:
    trace = tmp_path / "candidate.trace.ndjson"
    _write_trace(trace, _trace_event(hp=26))

    with pytest.raises(EvaluationIntegrityError):
        _check_trace(trace, candidate=True, expected_overrides=1)


@pytest.mark.parametrize("stage", ["probe", "dev"])
def test_heldout_stage_uses_only_the_heldout_seed_contract(stage: str) -> None:
    contract = _seed_contract_kwargs(stage, (1, 2))

    assert contract == {"heldout_seeds": (1, 2), "training_seeds": None}


def test_heldout_stage_specs_are_exactly_the_preregistered_pools() -> None:
    assert _stage_spec("probe") == STAGES["probe"]
    assert STAGES["probe"]["seed_count"] == 10
    assert STAGES["dev"]["seed_count"] == 30


def test_unregistered_stage_cannot_use_a_seed_contract() -> None:
    with pytest.raises(StageEvaluationError):
        _seed_contract_kwargs("train_hypothesis_3", (1, 2))


def test_paired_summary_supports_the_registered_dev30_size() -> None:
    parent = ["victory", "defeat"] * 15
    candidate = ["victory", "defeat"] * 14 + ["victory", "victory"]

    result = _paired_summary(parent, candidate)

    assert result["parent_wins"] == 15
    assert result["candidate_wins"] == 16
    assert result["candidate_only_wins"] == 1
    assert result["parent_only_wins"] == 0
    assert result["net_wins"] == 1
    assert result["discordant_pairs"] == 1
    assert result["exact_one_sided_sign_p_candidate_positive"] == 0.5


def test_paired_summary_reports_p_one_when_no_pairs_are_discordant() -> None:
    result = _paired_summary(["defeat"] * 30, ["defeat"] * 30)

    assert result["discordant_pairs"] == 0
    assert result["exact_one_sided_sign_p_candidate_positive"] == 1.0


def test_paired_summary_rejects_unknown_or_unpaired_outcomes() -> None:
    with pytest.raises(StageEvaluationError):
        _paired_summary(["victory"], ["unknown"])
    with pytest.raises(StageEvaluationError):
        _paired_summary(["victory"], [])
