from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sts1"
    / "sts1_armg_strategy_loop.py"
)
SPEC = importlib.util.spec_from_file_location("strategy_loop_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class Scores:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


def _candidate(kind: str, identity: str, priority: float, bucket: str, floor: int = 10):
    return {
        "kind": kind,
        "identity": identity,
        "prelabel_priority": priority,
        "state_bucket": bucket,
        "floor": floor,
        "uncertainty": {"uncertainty": min(1.0, priority / 3.0)},
    }


def test_uncertainty_is_higher_for_close_scores() -> None:
    close = m._choice_uncertainty(Scores([0.0, 0.05, -0.05]))
    clear = m._choice_uncertainty(Scores([4.0, 0.0, -1.0]))
    assert close["uncertainty"] > clear["uncertainty"]
    assert close["probability_margin"] < clear["probability_margin"]


def test_prelable_priority_values_late_dangerous_uncertain_states() -> None:
    safe = {
        "floor": 5,
        "hp": 70,
        "max_hp": 80,
        "kind": "map",
        "act": 1,
    }
    hard = {
        "floor": 45,
        "hp": 10,
        "max_hp": 80,
        "kind": "map",
        "act": 3,
    }
    low_uncertainty = {"uncertainty": 0.1}
    high_uncertainty = {"uncertainty": 0.8}
    assert m._prelabel_priority(hard, high_uncertainty, choices=4) > m._prelabel_priority(
        safe, low_uncertainty, choices=2
    )


def test_selector_guarantees_kind_coverage_before_global_fill() -> None:
    rows = [
        _candidate("map", "m1", 3.0, "map-a"),
        _candidate("map", "m2", 2.9, "map-b"),
        _candidate("map", "m3", 2.8, "map-c"),
        _candidate("card", "c1", 1.0, "card-a"),
        _candidate("event", "e1", 0.9, "event-a"),
        _candidate("shop", "s1", 0.8, "shop-a"),
        _candidate("rest", "r1", 0.7, "rest-a"),
    ]
    selected = m._select_teacher_candidates(rows, budget=7, min_per_kind=1)
    kinds = {row["kind"] for row in selected}
    assert kinds == {"map", "card", "event", "shop", "rest"}


def test_selector_prefers_diverse_buckets_inside_kind() -> None:
    rows = [
        _candidate("card", "c1", 3.0, "same"),
        _candidate("card", "c2", 2.9, "same"),
        _candidate("card", "c3", 2.0, "different"),
        _candidate("map", "m1", 1.0, "map-a"),
    ]
    selected = m._select_teacher_candidates(rows, budget=3, min_per_kind=2)
    card_buckets = {
        row["state_bucket"] for row in selected if row["kind"] == "card"
    }
    assert "same" in card_buckets
    assert "different" in card_buckets


def test_selector_deduplicates_identical_state_identity() -> None:
    rows = [
        _candidate("map", "same-id", 1.0, "bucket-a"),
        _candidate("map", "same-id", 3.0, "bucket-b"),
        _candidate("card", "card-id", 2.0, "card-a"),
    ]
    selected = m._select_teacher_candidates(rows, budget=5, min_per_kind=1)
    identities = [row["identity"] for row in selected]
    assert identities.count("same-id") == 1
    kept = next(row for row in selected if row["identity"] == "same-id")
    assert kept["prelabel_priority"] == 3.0


def test_state_bucket_separates_act_floor_hp_and_choice_complexity() -> None:
    base = {"kind": "event", "act": 1, "floor": 9, "hp": 60, "max_hp": 80}
    late = {"kind": "event", "act": 3, "floor": 39, "hp": 12, "max_hp": 80}
    assert m._state_bucket(base, choices=2) != m._state_bucket(late, choices=4)


def _replay_row(kind: str, index: int, *, margin: float = 5.0) -> dict:
    return {
        "schema_version": m.STRATEGY_DATASET_SCHEMA_VERSION,
        "combat_policy": "mcts_2000",
        "kind": kind,
        "obs": [float(index)],
        "descs": [[0.0], [1.0]],
        "priority": 2.0,
        "teacher_margin": margin,
        "current_armg_index": 0,
        "teacher_best_index": 1,
        "target_probs": [0.1, 0.9],
    }


def test_teacher_confidence_downweights_ambiguous_labels() -> None:
    assert m._teacher_confidence_weight(0.0) == 0.10
    assert m._teacher_confidence_weight(0.5) == 0.25
    assert m._teacher_confidence_weight(2.0) == 0.50
    assert m._teacher_confidence_weight(5.0) == 0.75
    assert m._teacher_confidence_weight(20.0) == 1.00


def test_replay_caps_shop_fraction_and_keeps_high_confidence(tmp_path: Path) -> None:
    replay = tmp_path / "replay.jsonl"
    new_data = tmp_path / "new.jsonl"
    rows = [
        *[_replay_row("shop", i, margin=0.0 if i < 15 else 10.0) for i in range(30)],
        *[_replay_row("map", 100 + i, margin=10.0) for i in range(30)],
    ]
    new_data.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = m._merge_replay(
        replay,
        new_data,
        max_examples=100,
        shop_max_fraction=0.25,
    )
    kept = m._read_examples(replay)
    shops = [row for row in kept if row["kind"] == "shop"]
    assert report["dropped_shop_examples"] == 20
    assert len(shops) == 10
    assert len(shops) / len(kept) <= 0.25
    assert all(float(row["confidence_weight"]) >= 0.75 for row in shops)


def test_parent_protected_checkpoint_scales_the_update(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pt"
    trained = tmp_path / "trained.pt"
    candidate = tmp_path / "candidate.pt"
    torch.save({"weight": torch.tensor([0.0, 2.0])}, parent)
    torch.save({"weight": torch.tensor([4.0, 6.0])}, trained)
    m._interpolate_checkpoint(
        source_weight=parent,
        trained_weight=trained,
        output_weight=candidate,
        alpha=0.25,
    )
    payload = torch.load(candidate, weights_only=True, map_location="cpu")
    assert torch.allclose(payload["weight"], torch.tensor([1.0, 3.0]))


def test_candidate_pool_prefers_a_gate_pass() -> None:
    rollback = {
        "name": "full",
        "dev_gate": {
            "status": "ROLLBACK",
            "win_delta": 3,
            "floor_delta": 3.0,
            "paired": {
                "candidate_better": 20,
                "candidate_worse": 10,
                "one_sided_sign_p": 0.01,
            },
        },
    }
    passed = {
        "name": "half",
        "dev_gate": {
            "status": "PASS",
            "win_delta": 0,
            "floor_delta": 0.5,
            "paired": {
                "candidate_better": 12,
                "candidate_worse": 10,
                "one_sided_sign_p": 0.25,
            },
        },
    }
    assert m._select_dev_candidate([rollback, passed])["name"] == "half"
