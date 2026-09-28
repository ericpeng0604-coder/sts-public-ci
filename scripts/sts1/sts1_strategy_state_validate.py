#!/usr/bin/env python3
"""Fail-closed validator for durable STS1 Strategy loop state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SCHEMA = "sts1-armg-strategy-loop-state-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_state_dir(state_dir: Path) -> dict[str, object]:
    state_path = state_dir / "strategy-state.json"
    weight_path = state_dir / "current-strategy.pt"

    if not state_path.is_file():
        raise RuntimeError("missing strategy-state.json")
    if not weight_path.is_file():
        raise RuntimeError("missing current-strategy.pt")

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError(
            f"state schema mismatch: {payload.get('schema_version')!r}"
        )

    expected_sha = payload.get("current_strategy_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise RuntimeError("state missing valid current_strategy_sha256")

    actual_sha = sha256(weight_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"current-strategy SHA mismatch: expected={expected_sha} actual={actual_sha}"
        )

    combat = payload.get("combat_mcts_sims")
    if isinstance(combat, bool) or not isinstance(combat, int) or combat < 1:
        raise RuntimeError("state missing positive combat_mcts_sims")

    for key in (
        "generation",
        "accepted_rounds",
        "rejected_rounds",
        "stagnation_count",
    ):
        value = payload.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid non-negative integer state field: {key}")

    for key in ("used_training_seeds", "used_evaluation_seeds"):
        values = payload.get(key, [])
        if not isinstance(values, list):
            raise RuntimeError(f"{key} must be a list")
        normalized = []
        for raw in values:
            if isinstance(raw, bool):
                raise RuntimeError(f"{key} contains boolean seed")
            normalized.append(int(raw))
        if len(normalized) != len(set(normalized)):
            raise RuntimeError(f"{key} contains duplicate seeds")

    replay = state_dir / "strategy-replay.jsonl"
    replay_lines = 0
    if replay.is_file():
        for line_no, raw in enumerate(
            replay.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("schema_version") != "sts1-armg-strategy-branch-dataset-v1":
                raise RuntimeError(
                    f"replay schema mismatch at line {line_no}"
                )
            replay_lines += 1

    return {
        "result": "PASS_STRATEGY_STATE",
        "strategy_sha256": actual_sha,
        "combat_mcts_sims": combat,
        "generation": int(payload.get("generation", 0)),
        "accepted_rounds": int(payload.get("accepted_rounds", 0)),
        "rejected_rounds": int(payload.get("rejected_rounds", 0)),
        "stagnation_count": int(payload.get("stagnation_count", 0)),
        "replay_examples": replay_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate_state_dir(args.state_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
