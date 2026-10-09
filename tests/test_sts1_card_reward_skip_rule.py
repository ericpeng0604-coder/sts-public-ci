from __future__ import annotations

from types import SimpleNamespace

import pytest

from roguelike_ai.sts1_phase3.simulator import (
    SimulatorRunError,
    _apply_card_reward_skip_deck_threshold,
    run_simulator_game,
)


class _Policy:
    def describe_choice(self, kind: str, descriptor: str) -> dict[str, str]:
        assert kind == "card"
        return {"kind": "card", "choice": descriptor}


class _DuplicatePolicy:
    def describe_choice(self, kind: str, descriptor: str) -> dict[str, str]:
        assert kind == "card"
        if descriptor == "skip":
            return {"kind": "card", "choice": "skip"}
        return {"kind": "card", "choice": "card", "card_name": descriptor}

    def deck_snapshot(self, gc: SimpleNamespace) -> list[dict[str, str]]:
        return [{"name": name} for name in gc.deck]


@pytest.mark.parametrize(
    ("deck_size", "recommended", "expected", "reason", "overridden"),
    [
        (30, 0, 1, "large_deck_skip_override", True),
        (29, 0, 0, "deck_below_threshold", False),
        (30, 1, 1, "parent_already_selected_skip", False),
    ],
)
def test_large_deck_rule_uses_only_the_unique_legal_skip(
    deck_size: int,
    recommended: int,
    expected: int,
    reason: str,
    overridden: bool,
) -> None:
    selected, detail = _apply_card_reward_skip_deck_threshold(
        recommended,
        "card",
        ["take-card", "skip"],
        SimpleNamespace(deck=[object() for _ in range(deck_size)]),
        _Policy(),
        deck_size_threshold=30,
    )

    assert selected == expected
    assert detail["reason"] == reason
    assert detail["eligible"] is (deck_size >= 30)
    assert detail["overridden"] is overridden


def test_rule_fails_closed_when_deck_or_skip_choice_is_incomplete() -> None:
    no_deck, no_deck_detail = _apply_card_reward_skip_deck_threshold(
        0,
        "card",
        ["take-card", "skip"],
        SimpleNamespace(),
        _Policy(),
        deck_size_threshold=30,
    )
    no_skip, no_skip_detail = _apply_card_reward_skip_deck_threshold(
        0,
        "card",
        ["take-card"],
        SimpleNamespace(deck=[object() for _ in range(30)]),
        _Policy(),
        deck_size_threshold=30,
    )

    assert no_deck == no_skip == 0
    assert no_deck_detail["reason"] == "incomplete_deck"
    assert no_skip_detail["reason"] == "skip_choice_missing_or_ambiguous"
    assert not no_deck_detail["overridden"]
    assert not no_skip_detail["overridden"]


def test_duplicate_only_rule_skips_only_a_recommended_card_already_in_deck() -> None:
    selected, detail = _apply_card_reward_skip_deck_threshold(
        0,
        "card",
        ["Iron Wave", "skip"],
        SimpleNamespace(deck=["Iron Wave"] * 30),
        _DuplicatePolicy(),
        deck_size_threshold=30,
        require_recommended_duplicate=True,
    )

    assert selected == 1
    assert detail["eligible"] is True
    assert detail["recommended_card_is_duplicate"] is True
    assert detail["reason"] == "large_deck_duplicate_skip_override"


def test_duplicate_only_rule_preserves_a_novel_recommendation() -> None:
    selected, detail = _apply_card_reward_skip_deck_threshold(
        0,
        "card",
        ["Iron Wave", "skip"],
        SimpleNamespace(deck=["Strike"] * 30),
        _DuplicatePolicy(),
        deck_size_threshold=30,
        require_recommended_duplicate=True,
    )

    assert selected == 0
    assert detail["eligible"] is False
    assert detail["recommended_card_is_duplicate"] is False
    assert detail["reason"] == "recommended_card_not_duplicate"


def test_h14_preserves_duplicate_recommendation_when_any_legal_reward_is_novel() -> None:
    selected, detail = _apply_card_reward_skip_deck_threshold(
        0,
        "card",
        ["Iron Wave", "Bash", "skip"],
        SimpleNamespace(deck=["Strike"] * 30 + ["Iron Wave"]),
        _DuplicatePolicy(),
        deck_size_threshold=30,
        require_recommended_duplicate=True,
        require_all_options_duplicate=True,
    )

    assert selected == 0
    assert detail["eligible"] is False
    assert detail["overridden"] is False
    assert detail["recommended_card_is_duplicate"] is True
    assert detail["all_reward_options_are_duplicates"] is False
    assert detail["legal_reward_card_count"] == 2
    assert detail["duplicate_reward_card_count"] == 1
    assert detail["novel_reward_card_count"] == 1
    assert detail["reason"] == "legal_novel_reward_option"


def test_h14_skips_only_when_every_legal_reward_card_is_already_in_deck() -> None:
    selected, detail = _apply_card_reward_skip_deck_threshold(
        0,
        "card",
        ["Iron Wave", "Bash", "skip"],
        SimpleNamespace(deck=["Strike"] * 30 + ["Iron Wave", "Bash"]),
        _DuplicatePolicy(),
        deck_size_threshold=30,
        require_recommended_duplicate=True,
        require_all_options_duplicate=True,
    )

    assert selected == 2
    assert detail["eligible"] is True
    assert detail["overridden"] is True
    assert detail["all_reward_options_are_duplicates"] is True
    assert detail["legal_reward_card_count"] == 2
    assert detail["duplicate_reward_card_count"] == 2
    assert detail["novel_reward_card_count"] == 0
    assert detail["reason"] == "large_deck_duplicate_skip_override"


def test_h14_fails_closed_when_a_nonrecommended_reward_identity_is_missing() -> None:
    class _PartialIdentityPolicy(_DuplicatePolicy):
        def describe_choice(self, kind: str, descriptor: str) -> dict[str, str]:
            if descriptor == "skip":
                return {"kind": "card", "choice": "skip"}
            if descriptor == "Bash":
                return {"kind": "card", "choice": "card"}
            return super().describe_choice(kind, descriptor)

    selected, detail = _apply_card_reward_skip_deck_threshold(
        0,
        "card",
        ["Iron Wave", "Bash", "skip"],
        SimpleNamespace(deck=["Strike"] * 30 + ["Iron Wave"]),
        _PartialIdentityPolicy(),
        deck_size_threshold=30,
        require_recommended_duplicate=True,
        require_all_options_duplicate=True,
    )

    assert selected == 0
    assert detail["eligible"] is False
    assert detail["overridden"] is False
    assert detail["reason"] == "incomplete_reward_card_identity"


def test_h14_mode_requires_duplicate_skip_rule_and_boolean_flag() -> None:
    with pytest.raises(SimulatorRunError, match="requires the duplicate-card Skip intervention"):
        run_simulator_game(
            student=None,
            sts=object(),
            seed=1,
            armg_policy=object(),
            require_all_card_reward_options_are_duplicates=True,
        )
    with pytest.raises(SimulatorRunError, match="must be a boolean"):
        run_simulator_game(
            student=None,
            sts=object(),
            seed=1,
            armg_policy=object(),
            skip_duplicate_card_reward_when_deck_size_at_least=30,
            require_all_card_reward_options_are_duplicates=1,  # type: ignore[arg-type]
        )


def test_duplicate_only_rule_fails_closed_when_recommended_card_identity_is_missing() -> None:
    class _MissingIdentityPolicy(_DuplicatePolicy):
        def describe_choice(self, kind: str, descriptor: str) -> dict[str, str]:
            if descriptor == "skip":
                return {"kind": "card", "choice": "skip"}
            return {"kind": "card", "choice": "card"}

    selected, detail = _apply_card_reward_skip_deck_threshold(
        0,
        "card",
        ["unknown", "skip"],
        SimpleNamespace(deck=["Unknown"] * 30),
        _MissingIdentityPolicy(),
        deck_size_threshold=30,
        require_recommended_duplicate=True,
    )

    assert selected == 0
    assert detail["eligible"] is False
    assert detail["reason"] == "incomplete_recommended_card_identity"


def test_rule_does_not_intervene_on_other_screens_or_when_disabled() -> None:
    other_screen, other_detail = _apply_card_reward_skip_deck_threshold(
        0,
        "map",
        ["take-card", "skip"],
        SimpleNamespace(deck=[object() for _ in range(30)]),
        _Policy(),
        deck_size_threshold=30,
    )
    disabled, disabled_detail = _apply_card_reward_skip_deck_threshold(
        0,
        "card",
        ["take-card", "skip"],
        SimpleNamespace(deck=[object() for _ in range(30)]),
        _Policy(),
        deck_size_threshold=None,
    )

    assert other_screen == disabled == 0
    assert other_detail["reason"] == "not_card_reward"
    assert disabled_detail["reason"] == "disabled"
    assert not other_detail["overridden"]
    assert not disabled_detail["overridden"]


def test_run_rejects_skip_rule_without_armg_policy() -> None:
    with pytest.raises(SimulatorRunError, match="requires the ArmG policy"):
        run_simulator_game(
            student=None,
            sts=object(),
            seed=1,
            skip_card_reward_when_deck_size_at_least=30,
        )


def test_run_rejects_duplicate_skip_rule_without_armg_policy() -> None:
    with pytest.raises(SimulatorRunError, match="requires the ArmG policy"):
        run_simulator_game(
            student=None,
            sts=object(),
            seed=1,
            skip_duplicate_card_reward_when_deck_size_at_least=30,
        )


@pytest.mark.parametrize("threshold", [0, -1, True, 30.0])
def test_run_rejects_invalid_skip_rule_threshold(threshold: object) -> None:
    with pytest.raises(SimulatorRunError, match="positive integer"):
        run_simulator_game(
            student=None,
            sts=object(),
            seed=1,
            armg_policy=object(),
            skip_card_reward_when_deck_size_at_least=threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("threshold", [0, -1, True, 30.0])
def test_run_rejects_invalid_duplicate_skip_rule_threshold(threshold: object) -> None:
    with pytest.raises(SimulatorRunError, match="positive integer"):
        run_simulator_game(
            student=None,
            sts=object(),
            seed=1,
            armg_policy=object(),
            skip_duplicate_card_reward_when_deck_size_at_least=threshold,  # type: ignore[arg-type]
        )
