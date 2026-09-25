from __future__ import annotations

from roguelike_ai.sts1_phase3.simulator import _hybrid_vote_choice_bits


def test_hybrid_student_breaks_mcts_tie() -> None:
    chosen, used_student = _hybrid_vote_choice_bits(
        [(1000, 11), (2000, 22)],
        student_bits=11,
    )
    assert chosen == 11
    assert used_student is True


def test_hybrid_mcts_consensus_beats_student() -> None:
    chosen, used_student = _hybrid_vote_choice_bits(
        [(1000, 22), (2000, 22)],
        student_bits=11,
    )
    assert chosen == 22
    assert used_student is False


def test_hybrid_student_outside_tie_falls_back_to_high_budget() -> None:
    chosen, used_student = _hybrid_vote_choice_bits(
        [(1000, 11), (2000, 22)],
        student_bits=33,
    )
    assert chosen == 22
    assert used_student is False
