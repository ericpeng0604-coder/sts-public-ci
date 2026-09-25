from __future__ import annotations

import pytest

from roguelike_ai.sts1_phase3.reference_policy import (
    ARMG_MCTS_2000_REFERENCE_ID,
    ARMG_MCTS_2000_REFERENCE_V1,
    REFERENCE_POLICY_SCHEMA_VERSION,
    ReferencePolicyError,
    get_reference_policy,
)


def test_armg_mcts2000_reference_policy_identity_is_frozen() -> None:
    spec = get_reference_policy(ARMG_MCTS_2000_REFERENCE_ID)
    assert spec is ARMG_MCTS_2000_REFERENCE_V1
    assert spec.role == "teacher_reference"
    assert spec.character == "IRONCLAD"
    assert spec.ascension == 0
    assert spec.combat_policy == "mcts"
    assert spec.noncombat_policy == "armg"
    assert spec.mcts_sims == 2000
    assert spec.armg_upstream_commit == "b20eb2cac2f52b22fbb6c79900c309b51ea0a1db"
    assert spec.armg_weight_git_blob_sha == "6666fb690fcceb592a1aa8dc74fa27bfee4575f0"


def test_reference_policy_manifest_is_reference_only() -> None:
    manifest = ARMG_MCTS_2000_REFERENCE_V1.manifest()
    assert manifest["schema_version"] == REFERENCE_POLICY_SCHEMA_VERSION
    assert manifest["policy_id"] == ARMG_MCTS_2000_REFERENCE_ID
    assert manifest["promotion_semantics"] == "REFERENCE_ONLY_NOT_CHAMPION"


def test_unknown_reference_policy_fails_closed() -> None:
    with pytest.raises(ReferencePolicyError):
        get_reference_policy("does-not-exist")
