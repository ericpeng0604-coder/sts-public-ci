from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "sts1" / "sts1_armg_ppo_v13_adapt.py"
SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v13_adapt", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def state(stagnation: int, *, decision: str = "HOLD", win_delta: int = 0):
    return {
        "schema_version": "sts1-armg-ppo-v13-loop-state-v1",
        "round_index": 7,
        "stagnation_count": stagnation,
        "last_decision": decision,
        "last_gate_30": {"win_delta": win_delta},
    }


def test_profile_ladder_is_bounded():
    expected = {
        0: "stable",
        1: "stable",
        2: "diversify",
        3: "broaden",
        4: "escape",
    }
    for stagnation, name in expected.items():
        profile = mod.choose_profile(state(stagnation))
        assert profile.name == name
        mod.validate_profile(profile)


def test_long_plateau_uses_wide_exploration_when_not_close():
    profile = mod.choose_profile(state(5, win_delta=0))
    assert profile.name == "wide_explore"
    assert profile.games_per_worker == 100
    assert profile.temperature > 1.0
    mod.validate_profile(profile)


def test_long_plateau_refines_a_near_miss():
    profile = mod.choose_profile(state(8, win_delta=3))
    assert profile.name == "near_miss_refine"
    assert profile.temperature < 1.0
    assert profile.learning_rate < mod.PROFILES["stable"].learning_rate
    mod.validate_profile(profile)


def test_promotion_resets_to_stable():
    profile = mod.choose_profile(state(9, decision="PROMOTE_OFFLINE", win_delta=5))
    assert profile.name == "stable"


def test_adapt_changes_training_only_not_gate():
    order = [
        "attempt",
        "retry_count",
        "games_per_worker",
        "temperature",
        "epochs",
        "learning_rate",
        "clip",
        "target_kl",
        "mcts_sims",
        "max_retries",
        "max_rounds",
        "report_issue",
        "reason",
    ]
    control = {
        "attempt": "8",
        "retry_count": "0",
        "games_per_worker": "50",
        "temperature": "1.0",
        "epochs": "4",
        "learning_rate": "3e-5",
        "clip": "0.20",
        "target_kl": "0.02",
        "mcts_sims": "2000",
        "max_retries": "3",
        "max_rounds": "0",
        "report_issue": "15",
        "reason": "continue_after_gate",
    }
    updated, report = mod.adapt(
        state=state(3, win_delta=1),
        control_order=order,
        control=control,
    )
    assert updated["strategy_profile"] == "broaden"
    assert updated["games_per_worker"] == "75"
    assert updated["temperature"] == "1.1"
    assert updated["epochs"] == "6"
    assert updated["mcts_sims"] == "2000"
    assert updated["report_issue"] == "15"
    assert report["gate_policy_changed"] is False
    assert report["production_champion_changed"] is False


def test_all_profiles_pass_safety_bounds():
    for profile in mod.PROFILES.values():
        mod.validate_profile(profile)
