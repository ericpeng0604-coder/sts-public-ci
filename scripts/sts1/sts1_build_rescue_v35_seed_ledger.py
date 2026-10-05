#!/usr/bin/env python3
"""Frozen, disjoint STS1 v3.5 seed ledger.

v3.5 never trains on or reuses any v3.4 formal evaluation seed.  The new
training pool and all advancement gates are mutually disjoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

SCHEMA = "sts1-v35-disjoint-seed-ledger-v1"
COUNTS = {"train300": 300, "gate50": 50, "fresh100": 100, "fresh500": 500}
V34_COUNTS = {"train160": 160, "gate50": 50, "fresh100": 100, "fresh500": 500}
OLD_RANDOM_STREAMS = (
    2026100131, 2026100132, 20261001322,
    2026100132 + 25, 2026100132 + 50,
    2026100132 + 75, 2026100132 + 90,
)


def _hashed_ledger(tag: str, counts: dict[str, int], blocked: set[int]) -> dict[str, list[int]]:
    taken = set(blocked)
    out: dict[str, list[int]] = {}
    for label, length in counts.items():
        items: list[int] = []
        counter = 0
        while len(items) < length:
            digest = hashlib.sha256(f"{tag}:{label}:{counter}".encode()).digest()
            seed = 1 + int.from_bytes(digest[:8], "big") % 1_999_999_999
            counter += 1
            if seed in taken:
                continue
            taken.add(seed)
            items.append(seed)
        out[label] = items
    return out


def protected_seeds(root: Path) -> set[int]:
    protected: set[int] = set()
    for path in sorted((root / "control").rglob("*seed*.txt")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            t = raw.strip()
            if t.isdecimal():
                protected.add(int(t))
    for stream in OLD_RANDOM_STREAMS:
        rng = random.Random(stream)
        for _ in range(3000):
            protected.add(rng.randrange(1, 2_000_000_000))

    # Reconstruct and permanently reserve every v3.4 ledger seed, including
    # train160 and all three unseen advancement sets.
    v34 = _hashed_ledger(
        "sts1-v34-independent-ledger-20261001",
        V34_COUNTS,
        protected,
    )
    protected.update(x for rows in v34.values() for x in rows)
    return protected


def make(root: Path, out: Path) -> dict:
    protected = protected_seeds(root)
    lists = _hashed_ledger(
        "sts1-v35-generalization-expansion-20261006",
        COUNTS,
        protected,
    )
    flat = [x for rows in lists.values() for x in rows]
    assert len(flat) == len(set(flat)), "v3.5 train/gate overlap"
    assert not (set(flat) & protected), "protected seed leaked into v3.5 ledger"

    out.mkdir(parents=True, exist_ok=True)
    for name, vals in lists.items():
        (out / f"{name}.txt").write_text(
            "\n".join(map(str, vals)) + "\n", encoding="utf-8"
        )
    report = {
        "schema": SCHEMA,
        "version": 1,
        "protected_seed_count": len(protected),
        "v34_ledger_reserved": sum(V34_COUNTS.values()),
        "sets": {
            k: {
                "size": len(v),
                "sha256": hashlib.sha256(
                    ("\n".join(map(str, v)) + "\n").encode()
                ).hexdigest(),
            }
            for k, v in lists.items()
        },
        "pairwise_overlap": 0,
        "overlap_with_protected": 0,
        "policy": (
            "v3.5 train, gate50, fresh100, and fresh500 are mutually disjoint; "
            "all v3.4 ledger seeds and historical control seeds are protected"
        ),
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
    print("V35_SEED_CONTRACT_PASS", json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
