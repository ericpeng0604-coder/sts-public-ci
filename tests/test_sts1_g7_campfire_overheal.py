from types import SimpleNamespace

import pytest

from roguelike_ai.sts1_phase3.simulator import _apply_campfire_overheal_override
from scripts.sts1.sts1_g7_h3_emergency_potion_eval import (
    EvaluationIntegrityError as RunnerEvaluationIntegrityError,
    _paired_summary_for_stage,
    _variant_override_count_key,
)


class _Policy:
    @staticmethod
    def describe_choice(kind, descriptor):
        assert kind == "rest"
        return {"kind": "rest", "option_index": descriptor["option_index"]}


class _GameContext:
    def __init__(self, hp, max_hp, *, pillow=False):
        self.cur_hp = hp
        self.max_hp = max_hp
        self._pillow = pillow

    def hasRelic(self, relic_id):
        assert relic_id == "REGAL_PILLOW"
        return self._pillow


class _Sts:
    RelicId = SimpleNamespace(REGAL_PILLOW="REGAL_PILLOW")


@pytest.mark.parametrize(
    ("hp", "max_hp", "pillow", "recommended", "expected_index", "expected_override"),
    [
        (75, 100, False, 0, 1, True),
        (70, 100, False, 0, 0, False),
        (60, 100, True, 0, 1, True),
        (75, 100, False, 1, 1, False),
    ],
)
def test_campfire_overheal_rule_uses_exact_legal_smith_choice(
    hp, max_hp, pillow, recommended, expected_index, expected_override
):
    selected, detail = _apply_campfire_overheal_override(
        recommended,
        "rest",
        [{"option_index": 0}, {"option_index": 1}],
        _GameContext(hp, max_hp, pillow=pillow),
        _Sts,
        _Policy(),
    )

    assert selected == expected_index
    assert detail["overridden"] is expected_override
    if expected_override:
        assert detail["selected_index"] == 1
        assert detail["reason"] == "rest_heal_would_exceed_max_hp"


def test_campfire_overheal_rule_fails_closed_without_legal_smith():
    selected, detail = _apply_campfire_overheal_override(
        0,
        "rest",
        [{"option_index": 0}, {"option_index": 2}],
        _GameContext(90, 100),
        _Sts,
        _Policy(),
    )

    assert selected == 0
    assert detail["overridden"] is False
    assert detail["reason"] == "rest_or_smith_not_uniquely_legal"


def test_campfire_overheal_rule_fails_closed_without_relic_state():
    context = SimpleNamespace(cur_hp=90, max_hp=100)
    selected, detail = _apply_campfire_overheal_override(
        0,
        "rest",
        [{"option_index": 0}, {"option_index": 1}],
        context,
        _Sts,
        _Policy(),
    )

    assert selected == 0
    assert detail["overridden"] is False
    assert detail["reason"] == "relic_state_unavailable"


@pytest.mark.parametrize(
    (
        "hp",
        "max_hp",
        "pillow",
        "expected_index",
        "expected_overridden",
        "expected_overheal",
        "expected_effective_heal",
        "expected_reason",
    ),
    [
        (90, 100, False, 1, True, 20, 10, "overheal_exceeds_effective_rest_heal"),
        (85, 100, False, 0, False, 15, 15, "overheal_not_greater_than_effective_rest_heal"),
        (75, 100, False, 0, False, 5, 25, "overheal_not_greater_than_effective_rest_heal"),
        (80, 100, True, 1, True, 25, 20, "overheal_exceeds_effective_rest_heal"),
        (60, 100, True, 0, False, 5, 40, "overheal_not_greater_than_effective_rest_heal"),
        (50, 100, True, 0, False, 0, 45, "rest_heal_does_not_overflow"),
    ],
)
def test_h5_overheal_must_exceed_effective_capped_rest_heal(
    hp,
    max_hp,
    pillow,
    expected_index,
    expected_overridden,
    expected_overheal,
    expected_effective_heal,
    expected_reason,
):
    selected, detail = _apply_campfire_overheal_override(
        0,
        "rest",
        [{"option_index": 0}, {"option_index": 1}],
        _GameContext(hp, max_hp, pillow=pillow),
        _Sts,
        _Policy(),
        require_overheal_exceeds_effective_rest_heal=True,
    )

    assert selected == expected_index
    assert detail["kind"] == "prefer_smith_when_overheal_exceeds_effective_rest_heal"
    assert detail["overridden"] is expected_overridden
    assert detail["overheal_amount"] == expected_overheal
    assert detail["effective_rest_heal_amount"] == expected_effective_heal
    assert detail["reason"] == expected_reason


def test_paired_stage_summary_supports_dev30_and_exact_sign_test():
    parent = ["defeat"] * 29 + ["victory"]
    candidate = ["victory", "victory"] + ["defeat"] * 28

    summary = _paired_summary_for_stage(parent, candidate)

    assert summary == {
        "parent_wins": 1,
        "candidate_wins": 2,
        "candidate_only_wins": 2,
        "parent_only_wins": 1,
        "net_wins": 1,
        "discordant_pairs": 3,
        "exact_one_sided_sign_p_candidate_positive": 0.5,
    }


@pytest.mark.parametrize(
    ("parent", "candidate"),
    [
        (["victory"], []),
        (["victory"], ["timeout"]),
    ],
)
def test_paired_stage_summary_fails_closed_on_incomplete_or_unknown_outcomes(parent, candidate):
    with pytest.raises(RunnerEvaluationIntegrityError):
        _paired_summary_for_stage(parent, candidate)


@pytest.mark.parametrize(
    ("candidate_policy", "expected_key"),
    [
        ("h3-emergency-potion", "emergency_potion_override_count"),
        ("h4-campfire-overheal", "campfire_overheal_override_count"),
        ("h5-effective-rest-heal", "campfire_overheal_override_count"),
    ],
)
def test_candidate_trace_uses_the_matching_variant_override_counter(candidate_policy, expected_key):
    assert _variant_override_count_key(candidate_policy) == expected_key
