from __future__ import annotations

from roguelike_ai.sts1_phase3.hybrid_model import (
    DEFAULT_HYBRID_MCTS_BUDGETS,
    HYBRID_FUSION_RULE,
    HYBRID_RUNTIME_ID,
    build_hybrid_model_spec,
    write_hybrid_model_manifest,
)


def test_hybrid_model_bundle_always_contains_mcts(tmp_path) -> None:
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"checkpoint")
    spec = build_hybrid_model_spec(checkpoint, loop_round=7)
    assert spec.runtime_id == HYBRID_RUNTIME_ID
    assert spec.mcts_budgets == DEFAULT_HYBRID_MCTS_BUDGETS == (1000, 2000)
    assert spec.fusion_rule == HYBRID_FUSION_RULE
    assert spec.noncombat_policy == "armg"
    assert spec.loop_round == 7


def test_hybrid_manifest_roundtrip_content(tmp_path) -> None:
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "model-manifest.json"
    payload = write_hybrid_model_manifest(
        checkpoint,
        manifest,
        loop_round=2,
    )
    assert manifest.is_file()
    assert payload["model_semantics"] == "STUDENT_CHECKPOINT_PLUS_MCTS_PLUS_ARMG"
    assert payload["mcts_budgets"] == [1000, 2000]
    assert payload["student_checkpoint_sha256"]
