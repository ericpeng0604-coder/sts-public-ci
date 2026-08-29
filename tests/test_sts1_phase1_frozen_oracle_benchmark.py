from __future__ import annotations

import pytest

from roguelike_ai.sts1_teacher.benchmark import (
    BenchmarkContractError,
    BenchmarkDecision,
    conservative_tie_agreement,
    exact_native_action_map,
    load_heldout_seeds,
    oracle_ties,
    phase1_baseline_gate,
    semantic_score_consistent,
    summarize_rows,
)
from roguelike_ai.sts1_teacher.contract import DecisionContext, PUBLIC_STATE_SCHEMA
from roguelike_ai.sts1_teacher.jialeiv_armb import ACTION_TYPE_INDEX, encode_candidate


def _context(actions: list[dict], *, hand: list[dict] | None = None) -> DecisionContext:
    return DecisionContext.from_public_state(
        {
            "schema_version": PUBLIC_STATE_SCHEMA,
            "source": "simulator",
            "combat_active": True,
            "hand": hand or [],
            "legal_actions": actions,
        }
    )


def test_load_heldout_seeds_allows_full_line_comments(tmp_path) -> None:
    path = tmp_path / "seeds.txt"
    path.write_text(
        "# held-out benchmark seeds\n\n" + "\n".join(str(seed) for seed in range(50)) + "\n",
        encoding="utf-8",
    )
    assert load_heldout_seeds(path) == tuple(range(50))


def test_load_heldout_seeds_rejects_non_comment_garbage(tmp_path) -> None:
    path = tmp_path / "seeds.txt"
    path.write_text(
        "not-a-seed\n" + "\n".join(str(seed) for seed in range(50)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkContractError, match="heldout_seed_not_integer"):
        load_heldout_seeds(path)


def test_oracle_ties_are_tolerance_aware_and_sorted() -> None:
    ties = oracle_ties({"b": 10.0, "a": 10.0 + 5e-10, "c": 9.0})
    assert ties == ("a", "b")


def test_armb_tie_agreement_is_conservative() -> None:
    assert conservative_tie_agreement(("a",), ("a", "b"))
    assert conservative_tie_agreement(("a", "b"), ("a", "b"))
    assert not conservative_tie_agreement(("a", "c"), ("a", "b"))
    assert not conservative_tie_agreement((), ("a", "b"))


def test_exact_native_action_map_is_bijective_without_fallback() -> None:
    public_actions = [{"kind": "end_turn"}, {"kind": "choose", "choice_index": 0, "selection_type": "SINGLE_CARD_SELECT"}]
    context = _context(public_actions)
    native = [object(), object()]
    mapped = exact_native_action_map(context, public_actions, native)
    assert set(mapped) == {action.action_id for action in context.legal_actions}
    assert set(mapped.values()) == set(native)


def test_public_choose_preserves_armb_action_type() -> None:
    context = _context([{"kind": "choose", "choice_index": 0, "selection_type": "SINGLE_CARD_SELECT"}])
    action = context.legal_actions[0]
    vector = encode_candidate(context, action, {})
    assert vector[ACTION_TYPE_INDEX["SINGLE_CARD_SELECT"]] == 1.0
    assert sum(vector[: len(ACTION_TYPE_INDEX)]) == 1.0


def test_semantic_duplicate_score_drift_fails_closed() -> None:
    hand = [
        {"position": 1, "name": "Strike", "type": "ATTACK", "cost": 1},
        {"position": 2, "name": "Strike", "type": "ATTACK", "cost": 1},
    ]
    context = _context(
        [{"kind": "play_card", "hand_index": 1}, {"kind": "play_card", "hand_index": 2}],
        hand=hand,
    )
    ids = [action.action_id for action in context.legal_actions]
    assert len({action.semantic_key for action in context.legal_actions}) == 1
    assert semantic_score_consistent(context, {ids[0]: 3.0, ids[1]: 3.0})
    assert not semantic_score_consistent(context, {ids[0]: 3.0, ids[1]: 3.1})


def test_phase1_baseline_gate_requires_strict_armb_wins_and_zero_errors() -> None:
    oracle = {"a": 2.0, "b": 1.0}
    good_rows = []
    for index in range(10):
        good_rows.append(
            BenchmarkDecision(
                seed=index,
                floor=1,
                combat_index=1,
                decision_index=index + 1,
                decision_signature=f"sig-{index}",
                legal_action_ids=("a", "b"),
                oracle_scores=oracle,
                oracle_tie_ids=("a",),
                armb_tie_ids=("a",),
                random_action_id="a" if index < 2 else "b",
                simple_action_id="a" if index < 3 else "b",
                trajectory_action_id="a",
            ).to_dict()
        )
    summary = summarize_rows(good_rows, seeds=list(range(50)))
    passed, reasons = phase1_baseline_gate(summary, deterministic=True)
    assert passed
    assert reasons == []

    tied = dict(summary)
    tied["agreement_rates"] = {"armb": 0.3, "random": 0.3, "simple": 0.2}
    passed, reasons = phase1_baseline_gate(tied, deterministic=True)
    assert not passed
    assert "armb_not_strictly_better_than_random" in reasons

    bad = dict(summary)
    bad["unresolved"] = 1
    passed, reasons = phase1_baseline_gate(bad, deterministic=True)
    assert not passed
    assert "unresolved_must_be_zero" in reasons
