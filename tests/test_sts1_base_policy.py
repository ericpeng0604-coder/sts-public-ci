from __future__ import annotations

from roguelike_ai.sts1_phase3.base_policy import (
    BASE_POLICY_SCHEMA_VERSION,
    STS1_BASE_POLICY,
    STS1_BASE_POLICY_ID,
    get_base_policy,
)


def test_project_base_is_armg_plus_mcts_2000() -> None:
    base = get_base_policy()
    assert base is STS1_BASE_POLICY
    assert base.policy_id == STS1_BASE_POLICY_ID
    assert base.role == "project_base_policy"
    assert base.combat_policy == "mcts"
    assert base.mcts_sims == 2000
    assert base.noncombat_policy == "armg"
    assert base.character == "IRONCLAD"
    assert base.ascension == 0


def test_base_manifest_preserves_reference_identity() -> None:
    manifest = STS1_BASE_POLICY.manifest()
    assert manifest["schema_version"] == BASE_POLICY_SCHEMA_VERSION
    assert manifest["reference_policy"]["policy_id"] == "armg+mcts2000-reference-v1"
    assert manifest["reference_policy"]["mcts_sims"] == 2000
    assert "PRIMARY_PROJECT_BASE" in manifest["foundation_semantics"]
