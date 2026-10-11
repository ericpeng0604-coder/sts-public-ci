import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/sts1/sts1_g7_boss_prepare_collect.py"
spec = importlib.util.spec_from_file_location("boss_collect", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
base = collector.base


def prepared(monkeypatch, tmp_path):
    prior = tmp_path / "prior"
    seeds = prior / "round-016-seeds"
    (seeds / "pools").mkdir(parents=True)
    inventory = {"schema_version": base.ledger.INVENTORY_SCHEMA_VERSION,
                 "complete": True, "inventory_id": "fixture-exclusions",
                 "source_manifests": [
                     {"category": category, "source_ref": f"fixture:{category}",
                      "sha256": "a" * 64, "seed_ids": [index + 1]}
                     for index, category in enumerate(base.ledger.REQUIRED_SOURCE_CATEGORIES)]}
    base.write_new(seeds / "exclusion-inventory.json", inventory)
    generated = base.ledger.generate_exploration_round(
        inventory, inventory_sha256=base.evaluation._sha256(seeds / "exclusion-inventory.json"),
        round_id="fixture-prior", generation_key="private-fixture-key")
    for name, pool in generated["pools"].items():
        base.write_new(seeds / "pools" / f"{name}.json", pool)
    runtime = {}
    for key in ("armg_source_sha256", "armg_vocab_sha256", "g7_checkpoint_sha256",
                "simulator_binding_sha256", "boss_preparation_profiler"):
        runtime[key] = tmp_path / key
        runtime[key].write_text(key)
    base.write_new(seeds / "h22-identity-private.json",
                   {"expected_sha256": {key: base.evaluation._sha256(path) for key, path in runtime.items()}})
    args = collector.configure(SimpleNamespace(prior_root=prior, g7_checkpoint=runtime["g7_checkpoint_sha256"]))
    monkeypatch.setattr(base, "runtime_files", lambda args: runtime)
    monkeypatch.setattr(base, "head", lambda: "b" * 40)
    monkeypatch.setattr(base.evaluation, "_validate_pinned_gameplay_manifest", lambda path: None)
    root = tmp_path / "new-round"
    collector.prepare(args, root)
    return args, root, runtime, generated


def test_new_round_excludes_all_prior_reserved_pools_and_freezes_profiler(monkeypatch, tmp_path):
    args, root, runtime, prior = prepared(monkeypatch, tmp_path)
    lock, train, pool = base.validate_preflight(args, root, collector.ROUND, collector.SEEDS_NAME)
    inventory = json.loads((root / collector.SEEDS_NAME / "exclusion-inventory.json").read_text())
    excluded, _ = base.ledger.validate_inventory(inventory)
    all_old = {seed for p in prior["pools"].values() for seed in p["seed_ids"]}
    assert all_old <= excluded and not all_old.intersection(train)
    assert len(train) == 10 and pool["purpose"] == "train"
    assert lock["round_id"] == collector.ROUND
    assert "boss_preparation_profiler" in lock["expected_sha256"]
    assert lock["expected_episodes"] == 11
    assert len(inventory["source_manifests"]) == len(base.ledger.REQUIRED_SOURCE_CATEGORIES) + 5


@pytest.mark.parametrize("change", ["runtime", "pool", "adapter", "commit"])
def test_mutated_frozen_inputs_block_collection(monkeypatch, tmp_path, change):
    args, root, runtime, prior = prepared(monkeypatch, tmp_path)
    if change == "runtime":
        runtime["boss_preparation_profiler"].write_text("changed")
    elif change == "pool":
        path = root / collector.SEEDS_NAME / "pools/train_hypothesis_1.json"
        path.write_text("{}")
    elif change == "adapter":
        Path(str(args.g7_checkpoint) + ".adapter.pt").write_text("forbidden")
    else:
        monkeypatch.setattr(base, "head", lambda: "c" * 40)
    with pytest.raises(ValueError):
        base.validate_preflight(args, root, collector.ROUND, collector.SEEDS_NAME)


def test_changed_native_binding_rejected_before_allocation(monkeypatch, tmp_path):
    args, root, runtime, prior = prepared(monkeypatch, tmp_path)
    runtime["simulator_binding_sha256"].write_text("changed")
    other_root = tmp_path / "another-round"
    with pytest.raises(ValueError, match="native binding"):
        collector.prepare(args, other_root)
    assert not other_root.exists()


def test_profiler_failure_stops_collection_without_retry(monkeypatch, tmp_path):
    args = SimpleNamespace()
    def run(args, root, **kwargs):
        kwargs["trace_observer"]([], {"pair_index": 1})
        pytest.fail("must stop at failed profiler")
    monkeypatch.setattr(base, "collect", run)
    monkeypatch.setattr(collector, "profile_trace", lambda *args: (_ for _ in ()).throw(ValueError("bad trace")))
    with pytest.raises(ValueError, match="bad trace"):
        collector.collect(args, tmp_path)


def test_configured_code_identity_includes_collector_and_profiler():
    args = collector.configure(SimpleNamespace(module_dir=Path("native"), armg_root=Path("armg"), g7_checkpoint=Path("G7")))
    files = base.runtime_files(args)
    assert files["boss_preparation_collector"] == SCRIPT
    assert files["boss_preparation_profiler"].is_file()
