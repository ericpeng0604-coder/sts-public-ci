from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_strategy_loop.py"
SPEC = importlib.util.spec_from_file_location("sts1_strategy_v7_human_verified", SCRIPT)
assert SPEC and SPEC.loader
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class FakeArmG:
    def describe_choice(self, kind, desc):
        assert kind == "map"
        return {"room": str(desc).upper()}


def test_human_path_probe_only_prioritizes_disagreement() -> None:
    prior = {
        "global_log_probs": {"MONSTER": -2.0, "REST": -0.1},
        "act_log_probs": {"1": {"MONSTER": -2.0, "REST": -0.1}},
        "context_log_probs": {
            "1|critical|medium": {"MONSTER": -2.0, "REST": -0.1},
        },
    }
    snapshot = {
        "kind": "map",
        "floor": 8,
        "hp": 20,
        "max_hp": 80,
        "gold": 100,
    }
    probe = m._human_path_probe(
        snapshot=snapshot,
        descs=["monster", "rest"],
        raw_scores=[1.0, 0.0],
        armg=FakeArmG(),
        prior=prior,
        disagreement_bonus=1.5,
    )
    assert probe["active"] is True
    assert probe["disagreement"] is True
    assert probe["armg_index"] == 0
    assert probe["human_index"] == 1
    assert 0.0 < probe["priority_bonus"] <= 1.5


def test_human_path_probe_never_changes_scores_or_labels() -> None:
    prior = {
        "global_log_probs": {"MONSTER": -2.0, "REST": -0.1},
        "act_log_probs": {"1": {"MONSTER": -2.0, "REST": -0.1}},
        "context_log_probs": {
            "1|critical|medium": {"MONSTER": -2.0, "REST": -0.1},
        },
    }
    raw_scores = [1.0, 0.0]
    snapshot = {
        "kind": "map",
        "floor": 8,
        "hp": 20,
        "max_hp": 80,
        "gold": 100,
    }
    before = list(raw_scores)
    probe = m._human_path_probe(
        snapshot=snapshot,
        descs=["monster", "rest"],
        raw_scores=raw_scores,
        armg=FakeArmG(),
        prior=prior,
        disagreement_bonus=1.5,
    )
    assert raw_scores == before
    assert set(probe) >= {"priority_bonus", "human_index", "armg_index"}
    assert "teacher_best_index" not in probe


def test_non_map_state_gets_no_human_priority() -> None:
    probe = m._human_path_probe(
        snapshot={"kind": "shop", "floor": 8},
        descs=["a", "b"],
        raw_scores=[0.0, 1.0],
        armg=FakeArmG(),
        prior={"global_log_probs": {}},
        disagreement_bonus=1.5,
    )
    assert probe["active"] is False
    assert probe["priority_bonus"] == 0.0
