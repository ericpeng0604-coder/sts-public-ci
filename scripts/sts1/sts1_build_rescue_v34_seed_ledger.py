#!/usr/bin/env python3
"""Frozen, disjoint STS1 v3.4 seed ledger. Never tune on the validation sets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

SCHEMA = "sts1-v34-disjoint-seed-ledger-v1"
COUNTS = {"train160": 160, "gate50": 50, "fresh100": 100, "fresh500": 500}
OLD_RANDOM_STREAMS = (2026100131, 2026100132, 20261001322,
                      2026100132 + 25, 2026100132 + 50,
                      2026100132 + 75, 2026100132 + 90)


def protected_seeds(root: Path) -> set[int]:
    protected: set[int] = set()
    for path in sorted((root / "control").rglob("*seed*.txt")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            t = raw.strip()
            if t.isdecimal():
                protected.add(int(t))
    # Previous experiments generated some seed sets on the fly.
    # Reserve all of their deterministic streams, including untested sets.
    for stream in OLD_RANDOM_STREAMS:
        rng = random.Random(stream)
        for _ in range(3000):
            protected.add(rng.randrange(1, 2_000_000_000))
    return protected


def make(root: Path, out: Path) -> dict:
    protected = protected_seeds(root)
    taken = set(protected)
    lists: dict[str, list[int]] = {}
    for label, length in COUNTS.items():
        items: list[int] = []
        counter = 0
        while len(items) < length:
            digest = hashlib.sha256(
                f"sts1-v34-independent-ledger-20261001:{label}:{counter}".encode()
            ).digest()
            seed = 1 + int.from_bytes(digest[:8], "big") % 1_999_999_999
            counter += 1
            if seed in taken:
                continue
            taken.add(seed)
            items.append(seed)
        lists[label] = items
    assert sum(len(x) for x in lists.values()) == len(
        set(x for rows in lists.values() for x in rows)
    ), "train/gate overlap"
    assert not (set(x for rows in lists.values() for x in rows) & protected), (
        "protected seed leaked into v3.4 ledger"
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, vals in lists.items():
        (out / f"{name}.txt").write_text(
            "\n".join(map(str, vals)) + "\n", encoding="utf-8"
        )
    report = {
        "schema": SCHEMA,
        "version": 1,
        "protected_seed_count": len(protected),
        "sets": {
            k: {"size": len(v), "sha256": hashlib.sha256(
                ("\n".join(map(str, v))+"\n").encode()).hexdigest()}
            for k,v in lists.items()
        },
        "pairwise_overlap": 0,
        "overlap_with_protected": 0,
        "policy": "heldout sets are used for one-time advancement gates only; never as teacher labels or training data",
    }
    (out / "seed-ledger.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    result = make(a.root, a.output_dir)
    print("V34_SEED_CONTRACT_PASS", json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
