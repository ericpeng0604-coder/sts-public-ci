"""Read-only, bounded Train-parent shadow audit; emits aggregate counters only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from roguelike_ai.sts1_phase3.inflame_end_turn import select_inflame_before_end_turn


class AuditError(RuntimeError):
    pass


def audit(directory: Path) -> dict:
    files = sorted(path for path in directory.glob("*.ndjson")
                   if "parent" in path.name and "trace" in path.name)
    if len(files) != 10 or sum(path.stat().st_size for path in files) > 64 * 1024 * 1024:
        raise AuditError("requires exactly ten bounded parent Train traces")
    seeds, identities = set(), None
    decisions = changed = covered = 0
    reasons: Counter[str] = Counter()
    for path in files:
        terminal, per_trace = None, 0
        with path.open(encoding="utf-8") as handle:
            header = json.loads(next(handle))
            meta = header.get("run_metadata", {})
            if (header.get("type") != "diagnostic_trace_header_v1"
                    or meta.get("arm") != "parent" or meta.get("stage") != "train"
                    or meta.get("round_id") != "round-011-20261010"
                    or meta.get("trial_id") != "h19" or meta.get("mcts_sims") != 2000
                    or meta.get("seed_disjointness_verified") is not True
                    or meta.get("simulator_gameplay_commit") != "7476a81954020087da31d41d16fddf475746ec2d"):
                raise AuditError("source is not the registered parent Train evidence")
            seed = meta.get("seed_id")
            if seed is None or seed in seeds:
                raise AuditError("missing or duplicate source seed identity")
            seeds.add(seed)
            current = {key: meta.get(key) for key in (
                "g7_checkpoint_sha256", "simulator_binding_sha256", "armg_source_sha256",
                "armg_vocab_sha256", "simulator_policy_source_sha256", "candidate_evaluator_sha256",
            )}
            if not all(isinstance(value, str) and len(value) == 64 for value in current.values()):
                raise AuditError("incomplete source identities")
            if identities is None:
                identities = current
            elif current != identities:
                raise AuditError("source identities disagree across traces")
            for line in handle:
                row = json.loads(line)
                if row.get("type") == "combat_decision_trace_v1":
                    if (row.get("legal_actions_complete") is not True
                            or row.get("mcts_sims") != 2000
                            or row.get("mcts_recommended_action") != row.get("selected_action")):
                        raise AuditError("incomplete legal actions or parent recommendation mismatch")
                    choice, reason = select_inflame_before_end_turn(
                        row.get("public_state"), row.get("canonical_native_legal_actions"),
                        row.get("mcts_recommended_action"), legal_actions_complete=True,
                    )
                    reasons[reason] += 1
                    decisions += 1
                    if choice is not None:
                        if choice.action != row["canonical_native_legal_actions"][choice.native_action_index]:
                            raise AuditError("shadow selector returned an illegal action")
                        changed += 1
                        per_trace += 1
                if "final_floor" in row and "complete" in row:
                    terminal = row
        if terminal is None or terminal.get("complete") is not True or any(
            type(terminal.get(key)) is not int or terminal[key] != 0 for key in (
                "illegal_action_count", "crash_count", "timeout_count", "communication_error_count"
            )
        ):
            raise AuditError("source terminal integrity is not verified")
        covered += per_trace > 0
    source = Path(sys.modules[select_inflame_before_end_turn.__module__].__file__)
    return {
        "status": "SHADOW_ONLY", "scope": "Train-only Sample(10), Round011 parent",
        "source_trace_count": len(files), "combat_decisions": decisions,
        "shadow_action_changes": changed, "distinct_trace_coverage": covered,
        "reason_counts": dict(sorted(reasons.items())),
        "selector_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "new_episodes": 0, "win_rate_improvement": "NOT_VERIFIED",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.train_trace_dir), sort_keys=True))
    except (AuditError, OSError, ValueError, TypeError, StopIteration):
        print(json.dumps({"status": "NOT_VERIFIED", "error": "Train shadow audit failed closed"}))
        raise SystemExit(2)
