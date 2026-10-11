"""Bounded offline analysis of the completed Round017 Train10 profiles only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from roguelike_ai.sts1_phase3.boss_preparation import public_summary
from roguelike_ai.sts1_phase3.card_preparation import analyze_profiles, load_native_types

FROZEN_COLLECTOR = "0a2e5d275b2df0bbf046014fe0eb7930784e8611"
ROUND = "round-017-20261011"
SAFETY = ("illegal_action_count", "crash_count", "timeout_count", "communication_error_count")


def safe_counts(record):
    return isinstance(record, dict) and all(type(record.get(k)) is int and record[k] == 0 for k in SAFETY)


def validate_header(header, pair_index):
    metadata = header.get("run_metadata", {})
    if (header.get("type") != "diagnostic_trace_header_v1"
            or metadata.get("stage") != "train"
            or metadata.get("purpose") != "baseline_live_resource_diagnosis"
            or metadata.get("round_id") != ROUND
            or metadata.get("code_commit") != FROZEN_COLLECTOR
            or metadata.get("mcts_sims") != 2000
            or type(metadata.get("pair_index")) is not int
            or metadata["pair_index"] != pair_index):
        raise ValueError("only frozen Round017 Train allowed")


def load_round17(root):
    root = Path(root)
    status = json.loads((root / "status-public.json").read_text(encoding="utf-8"))
    if (status.get("status") != "COMPLETE" or status.get("baseline_completed") != 10
            or status.get("actual_episodes") != 11 or status.get("trace_invariance") != "PASS"
            or not safe_counts(status.get("safety"))):
        raise ValueError("complete safe Round017 required")
    usage_path = root / "usage-private.jsonl"
    usage = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
    if not usage or usage[0].get("code_commit") != FROZEN_COLLECTOR or usage[0].get("round_id") != ROUND:
        raise ValueError("frozen collection provenance required")
    starts = [e for e in usage if e.get("status") == "EPISODE_STARTED"]
    completes = [e for e in usage if e.get("status") == "EPISODE_COMPLETE"]
    expected = [(1, "on"), (1, "off")] + [(i, "on") for i in range(2, 11)]
    if ([(e.get("index"), e.get("trace_mode")) for e in starts] != expected
            or [(e.get("index"), e.get("trace_mode")) for e in completes] != expected
            or any(type(e.get("index")) is not int for e in starts + completes)):
        raise ValueError("fixed episode sequence required")
    expected_statuses = ["RUNNING"] + [s for _ in expected for s in ("EPISODE_STARTED", "EPISODE_COMPLETE")] + ["COMPLETE"]
    if ([e.get("status") for e in usage] != expected_statuses
            or usage[-1].get("summary") != status):
        raise ValueError("complete ordered usage record required")
    for e in completes:
        result = e.get("result", {})
        if (not safe_counts(result) or result.get("seed_contract") != "training_internal"
                or result.get("outcome") not in {"victory", "defeat"} or result.get("error") is not None):
            raise ValueError("complete safe Train result required")
    profiles, input_hashes = [], {}
    expected_files = {f"preparation-pair-{i:02d}-private.json" for i in range(1, 11)}
    if {p.name for p in root.glob("preparation-pair-*-private.json")} != expected_files:
        raise ValueError("exactly ten preparation profiles required")
    for i in range(1, 11):
        with (root / "episodes" / f"pair-{i:02d}-parent.trace.ndjson").open(encoding="utf-8") as stream:
            line = stream.readline(1024 * 1024)
            if not line.endswith("\n"):
                raise ValueError("bounded complete header required")
            validate_header(json.loads(line), i)
        path = root / f"preparation-pair-{i:02d}-private.json"
        data = path.read_bytes()
        input_hashes[path.name] = sha256(data).hexdigest()
        profiles.append(json.loads(data))
    stored = json.loads((root / "boss-preparation-public.json").read_text(encoding="utf-8"))
    if public_summary(profiles) != stored or stored["baseline_wins"] != status.get("baseline_wins"):
        raise ValueError("derived preparation aggregates changed")
    input_hashes[usage_path.name] = sha256(usage_path.read_bytes()).hexdigest()
    return profiles, input_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-record", type=Path, required=True)
    args = parser.parse_args()
    types = load_native_types(args.native_source)
    profiles, hashes = load_round17(args.private_root)
    report = analyze_profiles(profiles, types)
    # Hashes record this analysis snapshot, not retrospective raw-trace authentication.
    with args.private_record.open("x", encoding="utf-8") as stream:
        json.dump({"round": ROUND, "input_hashes": hashes, "raw_traces_revalidated": False}, stream, indent=2)
    with args.public_output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "COMPLETE", "profiles": len(profiles), "episodes_started": 0,
                      "shadow_opportunities": report["shadow_clash_skip_opportunities"],
                      "shadow_runs": report["shadow_clash_runs"], "policy_changed": False}))


if __name__ == "__main__":
    main()
