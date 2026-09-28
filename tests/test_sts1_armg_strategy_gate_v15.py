from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sts1" / "sts1_armg_strategy_gate_v15.py"
SPEC = importlib.util.spec_from_file_location("strategy_gate_v15", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def _row(seed: int, *, win: bool, floor: int, policy: str = "mcts_2000"):
    return {
        "seed": seed,
        "result": "PASS_SIMULATOR_COMPLETE_RUN",
        "outcome": "victory" if win else "defeat",
        "final_floor": floor,
        "combat_policy": policy,
        "student_action_count": 0,
        "hybrid_student_vote_count": 0,
        "hybrid_student_tiebreak_count": 0,
        "illegal_action_count": 0,
        "timeout_count": 0,
        "crash_count": 0,
        "remote_error_count": 0,
    }


def test_pure_mcts_is_hard_requirement():
    parent = [_row(i, win=False, floor=20) for i in range(30)]
    candidate = [_row(i, win=False, floor=30) for i in range(30)]
    candidate[0]["combat_policy"] = "hybrid_mcts_1000_2000_student_tiebreak"
    summary = m._paired_summary(parent, candidate, expected_policy="mcts_2000")
    assert summary["combat_locked_pure_mcts"] is False
    assert m._gate_dev(summary)["status"] == "HOLD"


def test_dev_accepts_material_floor_gain_without_win_regression():
    parent = [_row(i, win=False, floor=20) for i in range(30)]
    candidate = [_row(i, win=False, floor=21) for i in range(30)]
    summary = m._paired_summary(parent, candidate, expected_policy="mcts_2000")
    assert m._gate_dev(summary)["status"] == "PASS"


def test_hidden_requires_material_improvement():
    parent = [_row(i, win=False, floor=20) for i in range(50)]
    candidate = [_row(i, win=False, floor=20) for i in range(50)]
    summary = m._paired_summary(parent, candidate, expected_policy="mcts_2000")
    assert m._gate_hidden(summary)["status"] == "HOLD"


def test_fresh_requires_win_gain_and_paired_signal():
    parent = [_row(i, win=(i < 10), floor=60 if i < 10 else 20) for i in range(100)]
    candidate = [_row(i, win=(i < 14), floor=60 if i < 14 else 22) for i in range(100)]
    summary = m._paired_summary(parent, candidate, expected_policy="mcts_2000")
    assert summary["win_delta"] == 4
    assert summary["candidate_better"] > summary["candidate_worse"]
    assert m._gate_fresh(summary)["status"] == "PASS"


def test_fresh_rejects_one_extra_win_with_weak_paired_signal():
    parent = [_row(i, win=(i < 10), floor=60 if i < 10 else 20) for i in range(100)]
    candidate = [_row(i, win=(i < 11), floor=60 if i < 11 else 20) for i in range(100)]
    summary = m._paired_summary(parent, candidate, expected_policy="mcts_2000")
    assert summary["win_delta"] == 1
    assert m._gate_fresh(summary)["status"] == "HOLD"


def test_fresh_seed_generator_never_reuses_forbidden():
    forbidden = {1, 2, 3, 4, 5}
    seeds = m._fresh_seeds(count=100, rng_seed=1234, forbidden=forbidden)
    assert len(seeds) == 100
    assert len(set(seeds)) == 100
    assert not (set(seeds) & forbidden)
