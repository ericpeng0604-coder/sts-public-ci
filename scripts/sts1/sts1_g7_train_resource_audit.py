"""Read-only, bounded Round013 Train resource audit; emit aggregates only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


class AuditError(RuntimeError):
    pass


def inventory(root: Path) -> dict[str, int]:
    files = sorted(root.glob("pair-*-parent.trace.ndjson"))
    sizes = [path.stat().st_size for path in files]
    if len(files) != 10 or sum(sizes) > 64 * 1024 * 1024:
        raise AuditError("expected ten bounded parent traces")
    return {"selected_train_parent_traces": len(files), "total_bytes": sum(sizes)}


def consumption_diagnostic(decisions: list[dict]) -> dict[str, int]:
    """A suspicious stale slot is evidence quality, not a missed-use policy label."""
    eligible = stale = 0
    for event, following in zip(decisions, decisions[1:]):
        action = event.get("selected_action", {})
        if action.get("kind") != "use_potion" or event.get("encounter_index") != following.get("encounter_index"):
            continue
        slot = action.get("potion_index")
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < 5:
            raise AuditError("invalid canonical potion ordinal")
        before = event.get("public_state", {}).get("potions")
        after = following.get("public_state", {}).get("potions")
        legal = following.get("canonical_native_legal_actions")
        if not isinstance(before, list) or len(before) != 5 or not isinstance(after, list) or len(after) != 5 or not isinstance(legal, list) or any(not isinstance(item, dict) for item in legal):
            raise AuditError("incomplete resource diagnostic inputs")
        eligible += 1
        if before[slot] not in {"EMPTY_POTION_SLOT", "INVALID"} and after[slot] == before[slot] and not any(
            item.get("kind") == "use_potion" and item.get("potion_index") == slot for item in legal
        ):
            stale += 1
    return {"same_encounter_successors": eligible, "retained_slot_without_legal_use": stale}


def audit(root: Path) -> dict:
    bounds = inventory(root)
    private_seeds = set()
    totals = Counter()
    identity = None
    for path in sorted(root.glob("pair-*-parent.trace.ndjson")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if any(not isinstance(row, dict) for row in rows):
            raise AuditError("invalid record shape")
        headers = [e for e in rows if e.get("type") == "diagnostic_trace_header_v1"]
        terminals = [e for e in rows if e.get("type") == "terminal_trace_v1"]
        if len(headers) != 1 or len(terminals) != 1:
            raise AuditError("missing unique provenance or terminal")
        meta = headers[0].get("run_metadata", {})
        if (meta.get("stage") != "train" or meta.get("arm") != "parent"
            or meta.get("round_id") != "round-013-20261010" or meta.get("trial_id") != "h20"
            or meta.get("mcts_sims") != 2000 or meta.get("seed_disjointness_verified") is not True):
            raise AuditError("only registered Round013 Train parent evidence is accepted")
        seed = meta.get("seed_id")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed in private_seeds:
            raise AuditError("invalid or duplicate private source identity")
        private_seeds.add(seed)
        current_identity = tuple(meta.get(key) for key in (
            "candidate_commit", "g7_checkpoint_sha256", "simulator_binding_sha256",
            "simulator_policy_source_sha256", "candidate_evaluator_sha256",
        ))
        if any(not isinstance(item, str) for item in current_identity):
            raise AuditError("missing source identity")
        if identity is not None and identity != current_identity:
            raise AuditError("mixed source identities")
        identity = current_identity
        terminal = terminals[0]
        if terminal.get("outcome") not in {"victory", "defeat"} or any(
            type(terminal.get(key)) is not int or terminal[key] != 0 for key in (
                "illegal_action_count", "crash_count", "timeout_count", "communication_error_count",
            )
        ):
            raise AuditError("incomplete or unsafe terminal")
        decisions = [e for e in rows if e.get("type") == "combat_decision_trace_v1"]
        if not decisions or any(e.get("legal_actions_complete") is not True or e.get("mcts_sims") != 2000 for e in decisions):
            raise AuditError("incomplete canonical action evidence")
        diag = consumption_diagnostic(decisions)
        totals.update(diag)
        totals["affected_train_traces"] += int(diag["retained_slot_without_legal_use"] > 0)
        totals["combat_decisions"] += len(decisions)
    return {"status": "DIAGNOSTIC_ONLY", "scope": "Round013 Train parent Sample10",
            **bounds, **dict(totals), "new_episodes": 0,
            "missed_potion_use_causal_label": "NOT_VERIFIED",
            "win_rate_improvement": "NOT_VERIFIED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-trace-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("Inventory", "Sample"), default="Inventory")
    args = parser.parse_args()
    try:
        print(json.dumps(inventory(args.private_trace_dir) if args.mode == "Inventory" else audit(args.private_trace_dir), sort_keys=True))
    except (AuditError, OSError, ValueError, TypeError, KeyError):
        print('{"status":"AUDIT_FAIL"}')
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
