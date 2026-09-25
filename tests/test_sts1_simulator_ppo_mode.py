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


class _V0LikePolicy:
    def select_action(self, public_state, *, require_command=True):
        raise AssertionError("should not reach policy selection")


class _V1LikePolicy(_V0LikePolicy):
    def sample_action(self, public_state, *, deterministic=False, require_command=True):
        raise AssertionError("should not reach policy sampling")


def test_collect_ppo_requires_sample_action_policy() -> None:
    with pytest.raises(SimulatorRunError, match="requires a Student v1"):
        run_simulator_game(
            student=_V0LikePolicy(),
            sts=_SeedOnlySTS,
            seed="347001",
            collect_ppo=True,
        )


def test_collect_ppo_cannot_mix_with_mcts() -> None:
    with pytest.raises(SimulatorRunError, match="cannot run with MCTS"):
        run_simulator_game(
            student=_V1LikePolicy(),
            sts=_SeedOnlySTS,
            seed="347001",
            collect_ppo=True,
            combat_mcts_sims=2000,
        )
