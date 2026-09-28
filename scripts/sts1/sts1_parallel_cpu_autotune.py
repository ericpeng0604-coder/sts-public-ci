#!/usr/bin/env python3
"""Recommend safe Strategy Teacher worker count from one completed round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def recommend_collection_workers(
    parallel: Mapping[str, Any],
    *,
    configured_workers: int,
    max_workers: int = 4,
) -> dict[str, Any]:
    if configured_workers < 1 or max_workers < 1:
        raise ValueError("worker counts must be positive")

    available = max(1, int(parallel.get("available_cpu", 1) or 1))
    ceiling = min(max_workers, available)
    current = min(configured_workers, ceiling)

    fallback = parallel.get("fallback_reason")
    parity = str(parallel.get("parity_status", "unknown"))
    mode = str(parallel.get("mode", "unknown"))
    observed = float(parallel.get("observed_parallelism", 1.0) or 1.0)
    cpu_cores = float(parallel.get("observed_cpu_cores", observed) or observed)
    probe_speedup = float(parallel.get("parity_probe_speedup", 0.0) or 0.0)

    if fallback not in (None, "", "none"):
        return {
            "workers": max(1, current - 1),
            "decision": "DECREASE",
            "reason": "parallel_fallback",
        }

    if current > 1 and parity != "PASS":
        return {
            "workers": max(1, current - 1),
            "decision": "DECREASE",
            "reason": "parity_not_passed",
        }

    if current >= ceiling:
        return {
            "workers": current,
            "decision": "KEEP",
            "reason": "cpu_ceiling",
        }

    if current == 1:
        return {
            "workers": min(2, ceiling),
            "decision": "INCREASE" if ceiling >= 2 else "KEEP",
            "reason": "probe_parallel_capacity" if ceiling >= 2 else "cpu_ceiling",
        }

    # Require real overlap, not just more processes. Either wall-time overlap or
    # the independent parity probe must show enough headroom to justify one more
    # worker. Thresholds are deliberately conservative.
    overlap_threshold = max(1.20, current * 0.55)
    probe_threshold = 1.25 if current == 2 else 1.50
    cpu_threshold = max(1.10, current * 0.45)
    healthy_mode = mode.startswith("fork_")
    efficient = (
        healthy_mode
        and parity == "PASS"
        and (
            observed >= overlap_threshold
            or probe_speedup >= probe_threshold
            or cpu_cores >= cpu_threshold
        )
    )

    if efficient:
        return {
            "workers": min(current + 1, ceiling),
            "decision": "INCREASE",
            "reason": "verified_cpu_headroom",
        }

    return {
        "workers": current,
        "decision": "KEEP",
        "reason": "insufficient_verified_headroom",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-report", type=Path, required=True)
    parser.add_argument("--configured-workers", type=int, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    report = json.loads(args.round_report.read_text(encoding="utf-8"))
    parallel = (report.get("dataset") or {}).get("parallel_teacher") or {}
    result = recommend_collection_workers(
        parallel,
        configured_workers=args.configured_workers,
        max_workers=args.max_workers,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
