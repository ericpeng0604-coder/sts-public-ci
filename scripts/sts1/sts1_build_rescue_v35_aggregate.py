#!/usr/bin/env python3
"""Aggregate v3.5 positive Full-Win teachers and verified negative alternatives."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sts1_build_rescue_v34_aggregate as v34


NEG_SCHEMA = "sts1-armg-negative-branch-v1"
NEG_TYPE = "v35_failed_rescue_alternative"
NEG_SOURCE = "sts1-build-rescue-new-win-miner-v35"


def _finite(xs: list[Any]) -> bool:
    try:
        return all(
            not isinstance(x, bool)
            and isinstance(x, (int, float))
            and math.isfinite(float(x))
            for x in xs
        )
    except Exception:
        return False


def _fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "kind": str(row["kind"]),
        "obs": row["obs"],
        "descs": row["descs"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _negative_error(row: Any, seed: int) -> str | None:
    if not isinstance(row, dict):
        return "negative row is not an object"
    if (
        row.get("schema_version") != NEG_SCHEMA
        or row.get("type") != NEG_TYPE
        or row.get("source") != NEG_SOURCE
    ):
        return "negative schema/source/type mismatch"
    try:
        if int(row.get("seed", -1)) != seed:
            return "negative seed mismatch"
        obs = row["obs"]
        descs = row["descs"]
        current = int(row["current_armg_index"])
        rejected = int(row["rejected_index"])
        target = int(row["target_boss_floor"])
    except (KeyError, TypeError, ValueError):
        return "negative fields malformed"
    if not isinstance(obs, list) or len(obs) != 412 or not _finite(obs):
        return "negative obs malformed"
    if (
        not isinstance(descs, list)
        or len(descs) < 2
        or any(not isinstance(d, list) or len(d) != 368 or not _finite(d) for d in descs)
    ):
        return "negative desc malformed"
    if not (0 <= current < len(descs) and 0 <= rejected < len(descs)) or current == rejected:
        return "negative action indices malformed"
    if row.get("combat_policy") != "mcts_2000":
        return "negative combat policy mismatch"

    stage = str(row.get("failure_stage", ""))
    b10 = row.get("boss_10k")
    b50 = row.get("boss_50k")
    if not v34._safe_sim(b10, seed):
        return "negative boss10 unsafe"
    if stage == "boss10_failed":
        if v34._passes_target(b10, target):
            return "boss10_failed row actually passed target"
        if b50 is not None:
            return "boss10_failed row unexpectedly has boss50"
    elif stage == "boss50_failed":
        if not v34._passes_target(b10, target):
            return "boss50_failed row did not pass boss10"
        if not v34._safe_sim(b50, seed):
            return "negative boss50 unsafe"
        if v34._passes_target(b50, target):
            return "boss50_failed row actually passed target"
    else:
        return "unknown negative failure stage"
    return None


def aggregate(results_dir: Path, screen_dir: Path, output_dir: Path) -> dict[str, Any]:
    summary = v34.aggregate(results_dir, screen_dir, output_dir)
    training = {int(x) for x in (screen_dir / "train300.txt").read_text().split()}
    positive_rows = [
        json.loads(x)
        for x in (output_dir / "new-win-teachers.jsonl").read_text().splitlines()
        if x.strip()
    ]
    positive_states = {_fingerprint(row) for row in positive_rows}

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    malformed: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("result-*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            seed = int(result.get("seed", -1))
        except Exception as exc:
            malformed.append({"path": str(path), "error": str(exc)})
            continue
        if seed not in training:
            continue
        for row in result.get("negative_examples") or []:
            err = _negative_error(row, seed)
            if err:
                malformed.append({"seed": seed, "path": str(path), "error": err})
                continue
            if _fingerprint(row) in positive_states:
                continue
            by_seed[seed].append(row)

    # Cap each seed so NO_RESCUE seeds cannot dominate the positive Full-Win pool.
    selected: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    for seed in sorted(by_seed):
        uniq: dict[tuple[str, int, int], dict[str, Any]] = {}
        for row in by_seed[seed]:
            key = (
                _fingerprint(row),
                int(row["current_armg_index"]),
                int(row["rejected_index"]),
            )
            uniq[key] = row
        rows = list(uniq.values())
        rows.sort(
            key=lambda r: (
                0 if r["failure_stage"] == "boss50_failed" else 1,
                -int(r.get("floor", 0)),
                str(r["kind"]),
                int(r["rejected_index"]),
            )
        )
        for row in rows[:6]:
            selected.append(row)
            stage_counts[str(row["failure_stage"])] += 1
            kind_counts[str(row["kind"])] += 1

    if malformed:
        # Malformed negative evidence is never silently admitted.
        (output_dir / "negative-validation-errors.json").write_text(
            json.dumps(malformed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    (output_dir / "negative-examples.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in selected),
        encoding="utf-8",
    )
    (output_dir / "negative-seeds.txt").write_text(
        "".join(f"{x}\n" for x in sorted({int(r["seed"]) for r in selected})),
        encoding="utf-8",
    )
    summary.update(
        {
            "schema_version": "sts1-v35-generalization-dataset-v1",
            "negative_examples": len(selected),
            "negative_seeds": len({int(r["seed"]) for r in selected}),
            "negative_stage_counts": dict(sorted(stage_counts.items())),
            "negative_kind_counts": dict(sorted(kind_counts.items())),
            "negative_validation_errors": len(malformed),
            "positive_negative_state_overlap": 0,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--screen-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    report = aggregate(a.results_dir, a.screen_dir, a.output_dir)
    print("V35_AGGREGATE", json.dumps(report, sort_keys=True), flush=True)
    print(f"teacher_count={report['teacher_examples']}")
    print(f"teacher_seed_count={report['teacher_seeds']}")
    print(f"negative_count={report['negative_examples']}")
    print(f"negative_seed_count={report['negative_seeds']}")
    print(f"failed_seed_count={report['failed_or_missing_seed_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
