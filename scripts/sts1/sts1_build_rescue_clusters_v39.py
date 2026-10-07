#!/usr/bin/env python3
"""Weight repeated multi-seed rescue patterns while retaining all teacher rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{line_no} is not an object")
        obs = row.get("obs")
        if not isinstance(obs, list) or len(obs) != 412:
            raise ValueError(f"{path.name}:{line_no} has an invalid observation")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in obs):
            raise ValueError(f"{path.name}:{line_no} has non-finite observations")
        try:
            int(row["seed"])
            int(row.get("floor", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}:{line_no} has invalid seed/floor") from exc
        rows.append(row)
    if not rows:
        raise ValueError("cluster input is empty")
    return rows


def _semantic(value: Any) -> str:
    if isinstance(value, dict):
        ignored = {"index", "selected_index", "candidate_index", "description", "repr", "position"}
        value = {str(k): v for k, v in value.items() if str(k) not in ignored}
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _optional_context(row: dict[str, Any], key: str, width: float) -> int | None:
    value = row.get(key)
    if key == "hp_ratio" and value is None and row.get("max_hp"):
        hp = row.get("hp", row.get("cur_hp"))
        if hp is not None:
            value = float(hp) / float(row["max_hp"])
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(math.floor(number / width))


def _act_bucket(row: dict[str, Any]) -> int:
    act = row.get("act")
    if act is not None:
        try:
            return int(act)
        except (TypeError, ValueError):
            pass
    floor = max(1, int(row.get("floor", 1)))
    return min(3, ((floor - 1) // 17) + 1)


def _base_signature(row: dict[str, Any]) -> tuple[str, ...]:
    floor = max(0, int(row.get("floor", 0)))
    deck = row.get("deck") or row.get("deck_before")
    if isinstance(deck, list):
        deck = sorted(str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in deck)
    relics = row.get("relics") or row.get("relics_before")
    if isinstance(relics, list):
        relics = sorted(str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in relics)
    return (
        str(row.get("kind", "unknown")),
        str(_act_bucket(row)),
        str(floor // 5),
        _semantic(row.get("teacher_choice")),
        _semantic(row.get("original_choice")),
        str(_optional_context(row, "hp_ratio", .2)),
        str(_optional_context(row, "gold", 50.0)),
        _semantic(deck),
        _semantic(relics),
    )


def _observation_z(rows: list[dict[str, Any]]) -> list[list[float]]:
    dim = len(rows[0]["obs"])
    means = [sum(float(row["obs"][j]) for row in rows) / len(rows) for j in range(dim)]
    variances = [sum((float(row["obs"][j]) - means[j]) ** 2 for row in rows) / len(rows) for j in range(dim)]
    std = [max(math.sqrt(v), 1e-3) for v in variances]
    return [[(float(value) - means[j]) / std[j] for j, value in enumerate(row["obs"])] for row in rows]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def cluster_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    z = _observation_z(rows)
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(_base_signature(row), []).append(index)

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    radii: dict[str, float] = {}
    for signature, indices in groups.items():
        nearest: list[float] = []
        pair_distances: dict[tuple[int, int], float] = {}
        for i in indices:
            peers = [j for j in indices if int(rows[j]["seed"]) != int(rows[i]["seed"])]
            for j in peers:
                if j <= i:
                    continue
                d = _distance(z[i], z[j])
                pair_distances[(i, j)] = d
            if peers:
                nearest.append(min(_distance(z[i], z[j]) for j in peers))
        if not nearest:
            continue
        radius = _median(nearest)
        key = hashlib.sha256("|".join(signature).encode("utf-8")).hexdigest()[:12]
        radii[key] = radius
        for (i, j), d in pair_distances.items():
            if d <= radius:
                union(i, j)

    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)
    cluster_ids: dict[int, str] = {}
    cluster_seed_counts: dict[int, int] = {}
    cluster_weights: dict[int, float] = {}
    repeated_clusters = 0
    for root, indices in components.items():
        seeds = sorted({int(rows[i]["seed"]) for i in indices})
        if len(seeds) >= 2:
            repeated_clusters += 1
            weight = min(2.0, 1.0 + .25 * (len(seeds) - 1))
        else:
            weight = .25
        identity = json.dumps({"signature": _base_signature(rows[indices[0]]), "seeds": seeds},
                              sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        cluster_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        for index in indices:
            cluster_ids[index] = cluster_id
            cluster_seed_counts[index] = len(seeds)
            cluster_weights[index] = weight

    annotated = []
    for index, row in enumerate(rows):
        annotated.append({
            **row,
            "cluster_id": cluster_ids[index],
            "cluster_seed_count": cluster_seed_counts[index],
            "cluster_weight": cluster_weights[index],
        })
    report = {
        "schema_version": "sts1-v39-cross-seed-rescue-clusters",
        "teacher_examples": len(rows),
        "teacher_seeds": len({int(row["seed"]) for row in rows}),
        "clusters": len(components),
        "repeated_multi_seed_clusters": repeated_clusters,
        "repeated_examples": sum(x["cluster_seed_count"] >= 2 for x in annotated),
        "single_seed_examples": sum(x["cluster_seed_count"] < 2 for x in annotated),
        "single_seed_weight": .25,
        "repeated_cluster_weight": "min(2.0, 1 + 0.25 * (distinct_seed_count - 1))",
        "context_keys": ["kind", "act_bucket", "floor_band_5", "choice_semantics", "hp_ratio", "gold", "deck", "relics", "normalized_observation_distance"],
        "cluster_radii_by_signature": dict(sorted(radii.items())),
        "weights_by_kind": {
            kind: {
                "examples": sum(row["kind"] == kind for row in annotated),
                "repeated": sum(row["kind"] == kind and row["cluster_seed_count"] >= 2 for row in annotated),
            }
            for kind in sorted({str(row["kind"]) for row in rows})
        },
    }
    return annotated, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = _load_jsonl(args.input)
    annotated, report = cluster_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in annotated), encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("V39_CLUSTER_REPORT", json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

