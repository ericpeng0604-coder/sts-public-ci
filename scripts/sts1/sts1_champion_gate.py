#!/usr/bin/env python3
"""Evaluate STS1 Champion/Candidate evidence with the frozen promotion rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.champion_gate import (
    FAST_GATE_POLICY,
    FORMAL_GATE_POLICY,
    ChampionGateError,
    evaluate_fixed_seed_gate,
    evaluate_real_game_gate,
)


def _load_many(paths: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            rows = payload["runs"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ChampionGateError(f"{path} does not contain a runs list")
        for row in rows:
            if not isinstance(row, dict):
                raise ChampionGateError(f"{path} contains a non-object run")
            runs.append(row)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("30", "50", "real"), required=True)
    parser.add_argument("--champion", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        champion = _load_many(args.champion)
        candidate = _load_many(args.candidate)
        if args.stage == "30":
            result = evaluate_fixed_seed_gate(
                champion, candidate, policy=FAST_GATE_POLICY
            )
        elif args.stage == "50":
            result = evaluate_fixed_seed_gate(
                champion, candidate, policy=FORMAL_GATE_POLICY
            )
        else:
            result = evaluate_real_game_gate(champion, candidate)
    except (OSError, json.JSONDecodeError, ChampionGateError, ValueError) as exc:
        result = {
            "schema_version": "sts1-champion-gate-v1",
            "gate": args.stage,
            "status": "HOLD",
            "error": str(exc),
        }

    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result.get("status") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
