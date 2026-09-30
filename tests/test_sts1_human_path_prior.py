from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from roguelike_ai.sts1_phase3.human_expert import HumanExpertPolicy
from roguelike_ai.sts1_phase3.human_path_prior import (
    SCHEMA_VERSION,
    build_path_prior,
    gold_bucket,
    hp_bucket,
    normalize_room,
    path_prior_score,
)


def _write_run(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "play_id": "path-expert",
        "character_chosen": "IRONCLAD",
        "ascension_level": 20,
        "victory": True,
        "is_daily": False,
        "is_trial": False,
        "is_endless": False,
        "path_taken": ["M", "E", "R", "$", "?", "T"],
        "current_hp_per_floor": [80, 70, 20, 20, 50, 60],
        "max_hp_per_floor": [80] * 6,
        "gold_per_floor": [20, 80, 120, 220, 300, 50],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_room_and_context_buckets() -> None:
    assert normalize_room("M") == "MONSTER"
    assert normalize_room("E") == "ELITE"
    assert normalize_room("$") == "SHOP"
    assert normalize_room("?") == "EVENT"
    assert normalize_room("T") == "TREASURE"
    assert normalize_room("unknown") is None
    assert hp_bucket(20, 80) == "critical"
    assert hp_bucket(70, 80) == "high"
    assert gold_bucket(50) == "low"
    assert gold_bucket(260) == "rich"


def test_build_path_prior_from_expert_runs(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_run(root / "runs" / "panacea-ironclad-sample" / "one.run")
    out = tmp_path / "path-prior.json"

    report = build_path_prior(root, out, min_ascension=15)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["accepted_runs"] == 1
    assert report["example_count"] == 6
    assert report["ignored_nodes"] == 0
    assert "1|critical|medium" in report["context_log_probs"]
    assert out.is_file()


def test_path_score_uses_context() -> None:
    prior = {
        "global_log_probs": {"ELITE": -1.0, "REST": -1.0},
        "act_log_probs": {"1": {"ELITE": -0.8, "REST": -1.2}},
        "context_log_probs": {
            "1|critical|medium": {"ELITE": -2.0, "REST": -0.1},
        },
    }
    rest = path_prior_score(prior, "REST", floor=8, hp=20, max_hp=80, gold=100)
    elite = path_prior_score(prior, "ELITE", floor=8, hp=20, max_hp=80, gold=100)
    assert rest > elite


class _Scores:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


def test_human_expert_policy_reranks_map_with_path_prior() -> None:
    policy = object.__new__(HumanExpertPolicy)
    policy.expert_strength = 0.0
    policy.expert_prior = {}
    policy.path_strength = 1.0
    policy.path_prior = {
        "global_log_probs": {"ELITE": -1.0, "REST": -1.0},
        "act_log_probs": {"1": {"ELITE": -1.0, "REST": -1.0}},
        "context_log_probs": {
            "1|critical|medium": {"ELITE": -2.0, "REST": -0.1},
        },
    }
    policy.choices = lambda gc: ("map", ["elite", "rest"], [lambda g: None, lambda g: None])
    policy.score_choices = lambda gc: ("map", ["elite", "rest"], _Scores([0.0, 0.0]))
    policy.describe_choice = lambda kind, desc: {"room": str(desc).upper()}

    gc = SimpleNamespace(floor_num=8, cur_hp=20, max_hp=80, gold=100)
    result = HumanExpertPolicy.decide(policy, gc, SimpleNamespace())

    assert result[0] == "map"
    assert result[1] == 1
    assert result[4][1] > result[4][0]


def test_human_expert_path_gate_respects_armg_confidence() -> None:
    policy = object.__new__(HumanExpertPolicy)
    policy.expert_strength = 0.0
    policy.expert_prior = {}
    policy.path_strength = 8.0
    policy.path_min_prior_spread = 0.0
    policy.path_max_armg_margin = 1.0
    policy.path_prior = {
        "global_log_probs": {"ELITE": -3.0, "REST": -0.1},
        "act_log_probs": {"1": {"ELITE": -3.0, "REST": -0.1}},
        "context_log_probs": {
            "1|critical|medium": {"ELITE": -3.0, "REST": -0.1},
        },
    }
    policy.choices = lambda gc: ("map", ["elite", "rest"], [lambda g: None, lambda g: None])
    policy.score_choices = lambda gc: ("map", ["elite", "rest"], _Scores([4.0, 0.0]))
    policy.describe_choice = lambda kind, desc: {"room": str(desc).upper()}

    gc = SimpleNamespace(floor_num=8, cur_hp=20, max_hp=80, gold=100)
    result = HumanExpertPolicy.decide(policy, gc, SimpleNamespace())

    assert result[1] == 0
    diag = policy.path_diagnostics_snapshot()
    assert diag["eligible_decisions"] == 0
    assert diag["blocked_armg_confident"] == 1
    assert diag["map_flips"] == 0


def test_human_expert_path_gate_requires_prior_separation() -> None:
    policy = object.__new__(HumanExpertPolicy)
    policy.expert_strength = 0.0
    policy.expert_prior = {}
    policy.path_strength = 8.0
    policy.path_min_prior_spread = 0.5
    policy.path_max_armg_margin = 10.0
    policy.path_prior = {
        "global_log_probs": {"ELITE": -1.0, "REST": -0.9},
        "act_log_probs": {"1": {"ELITE": -1.0, "REST": -0.9}},
        "context_log_probs": {
            "1|critical|medium": {"ELITE": -1.0, "REST": -0.9},
        },
    }
    policy.choices = lambda gc: ("map", ["elite", "rest"], [lambda g: None, lambda g: None])
    policy.score_choices = lambda gc: ("map", ["elite", "rest"], _Scores([0.2, 0.0]))
    policy.describe_choice = lambda kind, desc: {"room": str(desc).upper()}

    gc = SimpleNamespace(floor_num=8, cur_hp=20, max_hp=80, gold=100)
    result = HumanExpertPolicy.decide(policy, gc, SimpleNamespace())

    assert result[1] == 0
    diag = policy.path_diagnostics_snapshot()
    assert diag["eligible_decisions"] == 0
    assert diag["blocked_low_prior_confidence"] == 1
    assert diag["map_flips"] == 0
