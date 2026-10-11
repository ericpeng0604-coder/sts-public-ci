import importlib.util
import json
from pathlib import Path

import pytest

from roguelike_ai.sts1_phase3.boss_preparation import public_summary

path = Path(__file__).parents[1] / "scripts/sts1/sts1_card_preparation_audit.py"
spec = importlib.util.spec_from_file_location("card_preparation_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def collection(tmp_path):
    safety = dict.fromkeys(audit.SAFETY, 0)
    status = {"status": "COMPLETE", "baseline_completed": 10, "baseline_wins": 0,
              "actual_episodes": 11, "trace_invariance": "PASS", "safety": safety}
    write(tmp_path / "status-public.json", status)
    usage = [{"status": "RUNNING", "code_commit": audit.FROZEN_COLLECTOR, "round_id": audit.ROUND}]
    for i, mode in [(1, "on"), (1, "off")] + [(i, "on") for i in range(2, 11)]:
        usage.extend([{"status": "EPISODE_STARTED", "index": i, "trace_mode": mode},
                      {"status": "EPISODE_COMPLETE", "index": i, "trace_mode": mode,
                       "result": {**safety, "seed_contract": "training_internal", "outcome": "defeat", "error": None}}])
    usage.append({"status": "COMPLETE", "summary": status})
    (tmp_path / "usage-private.jsonl").write_text("\n".join(json.dumps(x) for x in usage))
    (tmp_path / "episodes").mkdir()
    profiles = []
    for i in range(1, 11):
        h = {"type": "diagnostic_trace_header_v1", "run_metadata": {
            "stage": "train", "purpose": "baseline_live_resource_diagnosis", "round_id": audit.ROUND,
            "code_commit": audit.FROZEN_COLLECTOR, "mcts_sims": 2000, "pair_index": i}}
        (tmp_path / "episodes" / f"pair-{i:02d}-parent.trace.ndjson").write_text(json.dumps(h) + "\n")
        p = {"schema_version": "sts1-boss-preparation-diagnostic-v1", "decision_records": [],
             "encounter_records": [{"observed_boss_at_entry": None, "resources": {"hp_fraction": .9, "potion_count": 1}}],
             "terminal": {"outcome": "defeat", "final_floor": 50}}
        profiles.append(p)
        write(tmp_path / f"preparation-pair-{i:02d}-private.json", p)
    write(tmp_path / "boss-preparation-public.json", public_summary(profiles))
    return tmp_path


def test_completed_bounded_profiles_are_reused(collection):
    profiles, hashes = audit.load_round17(collection)
    assert len(profiles) == 10 and len(hashes) == 11


@pytest.mark.parametrize("mutation", ["heldout", "unsafe", "bool_safety", "order", "extra_profile", "aggregate", "incomplete_header"])
def test_rejects_drift_and_incomplete_collection(collection, mutation):
    if mutation in {"heldout", "incomplete_header"}:
        path = collection / "episodes/pair-01-parent.trace.ndjson"
        if mutation == "incomplete_header":
            path.write_text('{}')
        else:
            h = json.loads(path.read_text())
            h["run_metadata"]["stage"] = "dev"
            path.write_text(json.dumps(h) + "\n")
    elif mutation in {"unsafe", "bool_safety", "order"}:
        path = collection / "usage-private.jsonl"
        records = [json.loads(l) for l in path.read_text().splitlines()]
        if mutation == "order":
            records[1], records[2] = records[2], records[1]
        else:
            records[2]["result"]["crash_count"] = 1 if mutation == "unsafe" else False
        path.write_text("\n".join(json.dumps(x) for x in records))
    elif mutation == "extra_profile":
        write(collection / "preparation-pair-11-private.json", {})
    else:
        path = collection / "boss-preparation-public.json"
        data = json.loads(path.read_text())
        data["baseline_wins"] = 1
        write(path, data)
    with pytest.raises(ValueError):
        audit.load_round17(collection)
