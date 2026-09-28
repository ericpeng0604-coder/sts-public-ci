from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sts1"
    / "sts1_parallel_cpu_autotune.py"
)
SPEC = importlib.util.spec_from_file_location("parallel_cpu_autotune", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def _parallel(**overrides):
    base = {
        "available_cpu": 4,
        "fallback_reason": None,
        "parity_status": "PASS",
        "mode": "fork_rolling",
        "observed_parallelism": 1.4,
        "observed_cpu_cores": 1.3,
        "parity_probe_speedup": 1.4,
    }
    base.update(overrides)
    return base


def test_two_workers_scale_to_three_when_overlap_is_verified() -> None:
    result = m.recommend_collection_workers(
        _parallel(),
        configured_workers=2,
    )
    assert result["workers"] == 3
    assert result["decision"] == "INCREASE"


def test_three_workers_scale_to_four_only_with_stronger_evidence() -> None:
    result = m.recommend_collection_workers(
        _parallel(
            observed_parallelism=1.8,
            observed_cpu_cores=1.7,
            parity_probe_speedup=1.7,
        ),
        configured_workers=3,
    )
    assert result["workers"] == 4
    assert result["decision"] == "INCREASE"


def test_three_workers_stay_three_when_headroom_is_weak() -> None:
    result = m.recommend_collection_workers(
        _parallel(
            observed_parallelism=1.2,
            observed_cpu_cores=1.1,
            parity_probe_speedup=1.2,
        ),
        configured_workers=3,
    )
    assert result["workers"] == 3
    assert result["decision"] == "KEEP"


def test_parallel_fallback_reduces_worker_count() -> None:
    result = m.recommend_collection_workers(
        _parallel(fallback_reason="RuntimeError:test"),
        configured_workers=3,
    )
    assert result["workers"] == 2
    assert result["decision"] == "DECREASE"


def test_parity_failure_reduces_worker_count() -> None:
    result = m.recommend_collection_workers(
        _parallel(parity_status="FAIL"),
        configured_workers=4,
    )
    assert result["workers"] == 3
    assert result["decision"] == "DECREASE"


def test_never_exceeds_available_cpu() -> None:
    result = m.recommend_collection_workers(
        _parallel(available_cpu=2),
        configured_workers=4,
    )
    assert result["workers"] == 2
    assert result["reason"] == "cpu_ceiling"
