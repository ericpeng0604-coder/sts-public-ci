from __future__ import annotations

import importlib.util
import math
from pathlib import Path

HELPER = Path(__file__).parents[1] / "src/roguelike_ai/sts1_phase3/residual_scoring.py"
SPEC = importlib.util.spec_from_file_location("residual_scoring_v311", HELPER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
apply_residual_scores = MODULE.apply_residual_scores
adapter_kind_enabled = MODULE.adapter_kind_enabled
choice_relative_margin_loss = MODULE.choice_relative_margin_loss
choice_relative_margin_shortfall = MODULE.choice_relative_margin_shortfall
top1_index = MODULE.top1_index


def test_each_alternative_uses_its_own_parent_gap():
    parent = [3.0, 1.0, -1.0]
    residual = [0.0, -0.2, 1.5]
    shortfall_1, parent_top = choice_relative_margin_shortfall(parent, residual, 1, 0.25)
    shortfall_2, _ = choice_relative_margin_shortfall(parent, residual, 2, 0.25)
    assert parent_top == 0
    assert shortfall_1 == 2.45
    assert shortfall_2 == 2.75


def test_common_residual_offset_does_not_change_relative_delta_or_top1():
    parent = [1.0, 0.0, -1.0]
    residual = [0.1, 1.4, 0.3]
    base, _ = choice_relative_margin_shortfall(parent, residual, 1, 0.2)
    shifted = [x + 100.0 for x in residual]
    shifted_shortfall, _ = choice_relative_margin_shortfall(parent, shifted, 1, 0.2)
    assert math.isclose(shifted_shortfall, base, rel_tol=0.0, abs_tol=1e-12)
    assert top1_index(apply_residual_scores(parent, residual, True)) == top1_index(
        apply_residual_scores(parent, shifted, True)
    )


def test_relative_delta_changes_choice_and_ties_keep_lowest_legal_index():
    parent = [2.0, 0.0, -3.0]
    residual = [0.0, 2.5, 0.0]
    assert top1_index(apply_residual_scores(parent, residual, True)) == 1
    assert top1_index([2.0, 2.0, -3.0]) == 0
    assert top1_index(apply_residual_scores(parent, residual, False)) == 0


def test_parent_top_teacher_is_not_counted_as_a_flip_and_indices_are_checked():
    parent = [4.0, 1.0, 0.0]
    residual = [0.0, 10.0, 0.0]
    assert choice_relative_margin_loss(parent, residual, 0, 0.5, lambda x: max(0.0, x)) is None
    try:
        choice_relative_margin_shortfall(parent, residual, 3, 0.5)
    except ValueError as exc:
        assert "legal choice" in str(exc)
    else:
        raise AssertionError("out-of-range Teacher index must fail closed")


def test_inference_applies_unscaled_residual_to_matching_choice_once():
    parent = [0.5, 1.0, 0.0]
    stored_residual = [1.0, -0.25, 0.4]
    effective = apply_residual_scores(parent, stored_residual, True)
    assert effective == [1.5, 0.75, 0.4]
    assert top1_index(effective) == 0
    assert apply_residual_scores(parent, stored_residual, False) is parent


def test_adapter_gate_is_not_checked_for_disabled_decision_kind():
    enabled = {"card", "map"}
    assert adapter_kind_enabled("card", enabled)
    assert not adapter_kind_enabled("shop", enabled)
