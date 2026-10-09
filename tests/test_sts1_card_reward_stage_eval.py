from __future__ import annotations

import pytest

from scripts.sts1.sts1_g7_h12_card_reward_eval import (
    EvaluationIntegrityError,
    _pool_key_for_stage,
    _seed_contract_kwargs,
)


@pytest.mark.parametrize("stage", ["probe", "dev"])
def test_h13_heldout_stages_use_heldout_seed_allowlist(stage: str) -> None:
    seeds = (101, 202)

    assert _pool_key_for_stage("h13", stage) == stage
    assert _seed_contract_kwargs(stage, seeds) == {
        "heldout_seeds": seeds,
        "training_seeds": None,
    }


def test_train_stage_uses_only_training_seed_allowlist() -> None:
    seeds = (101, 202)

    assert _pool_key_for_stage("h13", "train") == "train_hypothesis_2"
    assert _seed_contract_kwargs("train", seeds) == {
        "heldout_seeds": None,
        "training_seeds": seeds,
    }


@pytest.mark.parametrize(
    ("hypothesis", "stage"),
    [("h12", "probe"), ("h12", "dev"), ("h13", "train_hypothesis_2")],
)
def test_only_registered_stage_names_are_accepted(hypothesis: str, stage: str) -> None:
    with pytest.raises(EvaluationIntegrityError):
        _pool_key_for_stage(hypothesis, stage)


@pytest.mark.parametrize("stage", ["probe", "dev"])
def test_heldout_seed_contract_rejects_empty_pool(stage: str) -> None:
    with pytest.raises(EvaluationIntegrityError):
        _seed_contract_kwargs(stage, ())
