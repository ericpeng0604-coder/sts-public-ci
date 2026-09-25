from __future__ import annotations

import pytest

from roguelike_ai.sts1_phase3.simulator import SimulatorRunError, run_simulator_game


class _SeedOnlySTS:
    @staticmethod
    def get_seed_long(seed: str) -> int:
        if seed != "347001":
            raise ValueError(seed)
        return 163868251

    @staticmethod
    def get_seed_str(seed: int) -> str:
        if seed != 163868251:
            raise ValueError(seed)
        return "347001"


class _V1LikePolicy:
    def select_action(self, public_state, *, require_command=True):
        raise AssertionError("should not reach policy selection")

    def sample_action(self, public_state, *, deterministic=False, require_command=True):
        raise AssertionError("should not reach policy sampling")


def test_teacher_collection_requires_mcts() -> None:
    with pytest.raises(SimulatorRunError, match="requires an MCTS"):
        run_simulator_game(
            student=_V1LikePolicy(),
            sts=_SeedOnlySTS,
            seed="347001",
            collect_teacher=True,
        )


def test_teacher_and_ppo_collection_are_mutually_exclusive() -> None:
    with pytest.raises(SimulatorRunError, match="mutually exclusive"):
        run_simulator_game(
            student=_V1LikePolicy(),
            sts=_SeedOnlySTS,
            seed="347001",
            collect_teacher=True,
            collect_ppo=True,
            combat_mcts_sims=2000,
        )
