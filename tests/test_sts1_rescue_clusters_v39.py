from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/sts1/sts1_build_rescue_clusters_v39.py"
SPEC = importlib.util.spec_from_file_location("sts1_build_rescue_clusters_v39", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(seed: int, obs: list[float], *, choice: str = "take-card") -> dict:
    return {
        "seed": seed,
        "kind": "card",
        "floor": 12,
        "act": 1,
        "hp": 40,
        "max_hp": 60,
        "gold": 100,
        "teacher_choice": {"name": choice, "index": 0},
        "original_choice": {"name": "skip"},
        "obs": obs,
    }


def test_repeated_cross_seed_pattern_gets_higher_weight():
    rows = [row(101, [0.0] * 412), row(202, [0.0] * 412)]
    annotated, report = MODULE.cluster_rows(rows)
    assert annotated[0]["cluster_id"] == annotated[1]["cluster_id"]
    assert [r["cluster_seed_count"] for r in annotated] == [2, 2]
    assert [r["cluster_weight"] for r in annotated] == [1.25, 1.25]
    assert report["repeated_multi_seed_clusters"] == 1
    assert report["single_seed_examples"] == 0


def test_same_seed_rows_do_not_count_as_repeated_pattern():
    rows = [row(101, [0.0] * 412), row(101, [0.0] * 412)]
    annotated, report = MODULE.cluster_rows(rows)
    assert all(r["cluster_seed_count"] == 1 for r in annotated)
    assert all(r["cluster_weight"] == 0.25 for r in annotated)
    assert report["repeated_multi_seed_clusters"] == 0


def test_different_choice_semantics_are_not_clustered():
    rows = [row(101, [0.0] * 412), row(202, [0.0] * 412, choice="other-card")]
    annotated, report = MODULE.cluster_rows(rows)
    assert annotated[0]["cluster_id"] != annotated[1]["cluster_id"]
    assert report["repeated_examples"] == 0

