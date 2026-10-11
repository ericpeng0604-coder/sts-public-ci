import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/sts1/sts1_g7_live_train_collect.py"
spec = importlib.util.spec_from_file_location("live_collect", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def fixture(monkeypatch, tmp_path, *, invariant=True, live=True):
    from types import SimpleNamespace
    args = SimpleNamespace(module_dir=tmp_path, armg_root=tmp_path, g7_checkpoint=tmp_path)
    lock = {"code_commit": "a" * 40, "expected_sha256": {}}
    monkeypatch.setattr(collector, "validate_preflight", lambda *a: (lock, tuple(range(10)), {"manifest_sha256": "b" * 64}))
    monkeypatch.setattr(collector.evaluation, "_load_sts", lambda *a: object())
    monkeypatch.setattr(collector.evaluation, "ArmGNoncombatPolicy", lambda **kw: object())
    calls = []

    def run(**kw):
        calls.append(kw)
        kw["evidence_path"].write_text("{}\n")
        return {"outcome": "defeat", "error": None, "potion_inventory_snapshot_complete": True,
                **{key: 0 for key in collector.evaluation.SAFETY_COUNTERS}}

    monkeypatch.setattr(collector.evaluation, "run_simulator_game", run)
    monkeypatch.setattr(collector.evaluation, "_check_pair_integrity", lambda *a: None)
    monkeypatch.setattr(collector.evaluation, "_evidence_action_signature", lambda *a: ("c" * 64, 10))
    monkeypatch.setattr(collector.evaluation.h7, "_validate_trace", lambda *a: {})
    monkeypatch.setattr(collector.evaluation.h7, "_read_jsonl", lambda *a: iter([{"live_combat_resources_required": live}]))
    monkeypatch.setattr(collector.evaluation, "_validate_action_trace_completeness", lambda *a: None)
    monkeypatch.setattr(collector.evaluation, "_trace_replay_matches", lambda *a: invariant)
    return args, calls


def test_exact_budget_replay_and_unchanged_policy(monkeypatch, tmp_path):
    args, calls = fixture(monkeypatch, tmp_path)
    result = collector.collect(args, tmp_path)
    assert result["status"] == "COMPLETE"
    assert result["baseline_completed"] == 10
    assert result["actual_episodes"] == 11
    assert result["trace_invariance"] == "PASS"
    assert [call["seed"] for call in calls] == [0, 0, *range(1, 10)]
    assert all(call["combat_mcts_sims"] == 2000 for call in calls)
    assert all(call["collect_teacher"] is False and call["collect_ppo"] is False for call in calls)
    assert calls[1]["diagnostic_trace_path"] is None
    assert not any("lethal_potion_rescue" in call or "inflame_end_turn" in call for call in calls)
    assert json.loads((tmp_path / "status-public.json").read_text())["actual_episodes"] == 11


def test_trace_difference_blocks_further_seeds(monkeypatch, tmp_path):
    args, calls = fixture(monkeypatch, tmp_path, invariant=False)
    with pytest.raises(ValueError, match="invariance"):
        collector.collect(args, tmp_path)
    assert len(calls) == 2


def test_legacy_header_is_not_current_live_evidence(monkeypatch, tmp_path):
    args, calls = fixture(monkeypatch, tmp_path, live=False)
    with pytest.raises(ValueError, match="live resource"):
        collector.collect(args, tmp_path)
    assert len(calls) == 1


def test_consumed_run_cannot_restart(monkeypatch, tmp_path):
    args, calls = fixture(monkeypatch, tmp_path)
    (tmp_path / "usage-private.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="already started"):
        collector.collect(args, tmp_path)
    assert calls == []


def test_private_paths_reject_repository():
    with pytest.raises(ValueError, match="outside"):
        collector.private_path(collector.REPO / "evidence/sts1/private")
