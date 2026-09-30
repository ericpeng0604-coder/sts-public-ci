from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v15_tournament.py"

SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v15_tournament", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


TRAIN_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_train_v14_sharded.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_train_v14_sharded_v15", TRAIN_SCRIPT)
assert TRAIN_SPEC and TRAIN_SPEC.loader
train_mod = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train_mod
TRAIN_SPEC.loader.exec_module(train_mod)


def row(name: str, *, passed: bool, wins: int, floor: float, safe: bool = True, idx: int = 0):
    return {
        "name": name,
        "rank_key": [
            int(safe),
            int(passed),
            wins,
            floor,
            0,
            -idx,
        ],
    }


def test_tournament_prefers_existing_dev_gate_pass():
    winner = mod.choose_winner(
        [
            row("A", passed=False, wins=5, floor=42.0, idx=0),
            row("B", passed=True, wins=4, floor=41.0, idx=1),
        ]
    )
    assert winner["name"] == "B"


def test_tournament_prefers_more_wins_then_floor():
    winner = mod.choose_winner(
        [
            row("A", passed=True, wins=4, floor=41.0, idx=0),
            row("B", passed=True, wins=5, floor=39.0, idx=1),
            row("C", passed=True, wins=5, floor=40.0, idx=2),
        ]
    )
    assert winner["name"] == "C"


def test_tournament_never_prefers_unsafe_candidate():
    winner = mod.choose_winner(
        [
            row("A", passed=True, wins=8, floor=50.0, safe=False, idx=0),
            row("B", passed=False, wins=3, floor=38.0, safe=True, idx=1),
        ]
    )
    assert winner["name"] == "B"



def _strategy_row(kind: str, *, current: int, teacher: int, priority: float = 2.0):
    return {
        "schema_version": "sts1-armg-strategy-branch-dataset-v1",
        "combat_policy": "mcts_2000",
        "kind": kind,
        "floor": 22,
        "obs": [0.0, 1.0],
        "descs": [[0.0], [1.0]],
        "current_armg_index": current,
        "teacher_best_index": teacher,
        "target_probs": [0.25, 0.75],
        "priority": priority,
        "teacher_margin": 8.0,
    }


def test_strategy_teacher_import_keeps_only_disagreements_and_balances_kinds(tmp_path):
    import json

    path = tmp_path / "strategy-replay.jsonl"
    rows = [
        _strategy_row("map", current=0, teacher=1, priority=3.0),
        _strategy_row("map", current=0, teacher=1, priority=2.0),
        _strategy_row("shop", current=0, teacher=1, priority=2.5),
        _strategy_row("rest", current=1, teacher=1, priority=3.0),
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = train_mod.load_strategy_teacher_examples(
        path,
        max_examples=3,
        combat_policy="mcts_2000",
    )

    assert len(selected) == 3
    assert all(row["current_armg_index"] != row["teacher_best_index"] for row in selected)
    assert {row["kind"] for row in selected[:2]} == {"map", "shop"}
    assert all(abs(sum(row["target_probs"]) - 1.0) < 1e-6 for row in selected)


def test_strategy_teacher_import_fails_closed_on_mcts_policy_drift(tmp_path):
    import json
    import pytest

    path = tmp_path / "strategy-replay.jsonl"
    row = _strategy_row("map", current=0, teacher=1)
    row["combat_policy"] = "mcts_50000"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="combat policy mismatch"):
        train_mod.load_strategy_teacher_examples(
            path,
            max_examples=32,
            combat_policy="mcts_2000",
        )



ADAPT_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v15_adapt.py"
ADAPT_SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v15_adapt_test", ADAPT_SCRIPT)
assert ADAPT_SPEC and ADAPT_SPEC.loader
adapt_mod = importlib.util.module_from_spec(ADAPT_SPEC)
sys.modules[ADAPT_SPEC.name] = adapt_mod
ADAPT_SPEC.loader.exec_module(adapt_mod)


def test_v15_adapt_heals_missing_teacher_parallel_control_keys():
    state = {
        "schema_version": "sts1-armg-ppo-v14-loop-state-v1",
        "round_index": 7,
        "stagnation_count": 6,
        "last_decision": "HOLD_PARENT",
        "last_dev_gate": {
            "win_delta": -1,
            "mean_paired_floor_delta": 0.1,
        },
    }
    order = [
        "attempt",
        "retry_count",
        "games_per_worker",
        "mcts_sims",
        "max_rounds",
        "ppo_version",
    ]
    control = {
        "attempt": "4",
        "retry_count": "0",
        "games_per_worker": "80",
        "mcts_sims": "2000",
        "max_rounds": "0",
        "ppo_version": "1.5",
    }

    updated, report = adapt_mod.adapt(
        state=state,
        control_order=order,
        control=control,
    )

    assert updated["strategy_teacher_coef"] == "0.01"
    assert updated["strategy_teacher_max_examples"] == "512"
    assert updated["strategy_teacher_required"] == "1"
    assert updated["candidate_threads"] == "4"
    assert updated["mcts_sims"] == "2000"
    assert updated["max_rounds"] == "0"
    assert all(
        key in order
        for key in (
            "strategy_teacher_coef",
            "strategy_teacher_max_examples",
            "strategy_teacher_required",
            "candidate_threads",
        )
    )
    assert report["production_champion_changed"] is False



EVAL_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v15_eval.py"
EVAL_SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v15_eval_test", EVAL_SCRIPT)
assert EVAL_SPEC and EVAL_SPEC.loader
eval_mod = importlib.util.module_from_spec(EVAL_SPEC)
sys.modules[EVAL_SPEC.name] = eval_mod
EVAL_SPEC.loader.exec_module(eval_mod)


def _complete_eval_run(seed: int):
    return {
        "seed": seed,
        "result": "PASS_SIMULATOR_COMPLETE_RUN",
        "outcome": "defeat",
        "final_floor": 20,
        "illegal_action_count": 0,
        "crash_count": 0,
        "timeout_count": 0,
        "remote_error_count": 0,
    }


def test_parent_evidence_validation_is_identity_strict():
    seeds = list(range(1, 51))
    payload = {
        "schema_version": eval_mod.PARENT_SCHEMA,
        "parent_weight_sha256": "abc",
        "simulator_id": "sim",
        "mcts_sims": 2000,
        "all_dev_seeds": seeds,
        "evaluated_seeds": seeds[:30],
        "runs": [_complete_eval_run(seed) for seed in seeds[:30]],
    }
    ordered = eval_mod._validate_parent_evidence(
        payload,
        dev_seeds=seeds,
        mcts_sims=2000,
        simulator_id="sim",
    )
    assert [row["seed"] for row in ordered] == seeds[:30]

    import pytest
    with pytest.raises(RuntimeError, match="MCTS mismatch"):
        eval_mod._validate_parent_evidence(
            payload,
            dev_seeds=seeds,
            mcts_sims=50000,
            simulator_id="sim",
        )
    with pytest.raises(RuntimeError, match="simulator mismatch"):
        eval_mod._validate_parent_evidence(
            payload,
            dev_seeds=seeds,
            mcts_sims=2000,
            simulator_id="other",
        )


def test_tournament_evidence_seed_map_fails_closed():
    import pytest
    seeds = list(range(1, 31))
    rows = [_complete_eval_run(seed) for seed in seeds]
    mapped = mod._runs_by_seed(rows, seeds)
    assert len(mapped) == 30

    duplicate = rows + [dict(rows[0])]
    with pytest.raises(RuntimeError, match="invalid/duplicate"):
        mod._runs_by_seed(duplicate, seeds)



PARENT_CACHE_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v15_parent_cache.py"
PARENT_CACHE_SPEC = importlib.util.spec_from_file_location(
    "sts1_armg_ppo_v15_parent_cache_test",
    PARENT_CACHE_SCRIPT,
)
assert PARENT_CACHE_SPEC and PARENT_CACHE_SPEC.loader
parent_cache_mod = importlib.util.module_from_spec(PARENT_CACHE_SPEC)
sys.modules[PARENT_CACHE_SPEC.name] = parent_cache_mod
PARENT_CACHE_SPEC.loader.exec_module(parent_cache_mod)


def test_parent_fast_cache_materializes_without_external_hydration(tmp_path):
    import json

    weight = tmp_path / "parent.pt"
    weight.write_bytes(b"parent-weight")
    seeds = list(range(1, 51))
    runs = [_complete_eval_run(seed) for seed in seeds[:30]]
    sim_head = "7476a81954020087da31d41d16fddf475746ec2d"
    cache = {
        "schema_version": parent_cache_mod.CACHE_SCHEMA,
        "weight_sha256": parent_cache_mod.sha256(weight),
        "seeds": seeds,
        "mcts_sims": 2000,
        "simulator_id": parent_cache_mod.simulator_id(sim_head),
        "runs": runs,
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"v14_eval_caches": {"dev_parent": cache}}) + "\n",
        encoding="utf-8",
    )
    seed_file = tmp_path / "dev-seeds.txt"
    seed_file.write_text("\n".join(str(seed) for seed in seeds) + "\n", encoding="utf-8")
    out = tmp_path / "out"

    assert parent_cache_mod.materialize_cached_parent(
        state_path=state_path,
        parent_weight=weight,
        dev_seed_file=seed_file,
        output_dir=out,
        mcts_sims=2000,
        sim_head=sim_head,
    )
    payload = json.loads((out / "parent-eval.json").read_text(encoding="utf-8"))
    assert payload["fresh_runs"] == 0
    assert payload["cache_hits_before"] == 30
    assert payload["all_dev_seeds"] == seeds
    assert len(payload["runs"]) == 30


def test_parent_fast_cache_fails_closed_on_identity_or_safety_drift(tmp_path):
    import json

    weight = tmp_path / "parent.pt"
    weight.write_bytes(b"parent-weight")
    seeds = list(range(1, 51))
    runs = [_complete_eval_run(seed) for seed in seeds[:30]]
    sim_head = "7476a81954020087da31d41d16fddf475746ec2d"
    cache = {
        "schema_version": parent_cache_mod.CACHE_SCHEMA,
        "weight_sha256": parent_cache_mod.sha256(weight),
        "seeds": seeds,
        "mcts_sims": 2000,
        "simulator_id": parent_cache_mod.simulator_id(sim_head),
        "runs": runs,
    }
    state_path = tmp_path / "state.json"
    seed_file = tmp_path / "dev-seeds.txt"
    seed_file.write_text("\n".join(str(seed) for seed in seeds) + "\n", encoding="utf-8")

    cache["mcts_sims"] = 50000
    state_path.write_text(json.dumps({"v14_eval_caches": {"dev_parent": cache}}))
    assert not parent_cache_mod.materialize_cached_parent(
        state_path=state_path,
        parent_weight=weight,
        dev_seed_file=seed_file,
        output_dir=tmp_path / "bad-mcts",
        mcts_sims=2000,
        sim_head=sim_head,
    )

    cache["mcts_sims"] = 2000
    cache["runs"][0]["timeout_count"] = 1
    state_path.write_text(json.dumps({"v14_eval_caches": {"dev_parent": cache}}))
    assert not parent_cache_mod.materialize_cached_parent(
        state_path=state_path,
        parent_weight=weight,
        dev_seed_file=seed_file,
        output_dir=tmp_path / "unsafe",
        mcts_sims=2000,
        sim_head=sim_head,
    )


def test_parent_fast_cache_rejects_seed_pin_drift(tmp_path):
    import json

    weight = tmp_path / "parent.pt"
    weight.write_bytes(b"parent-weight")
    seeds = list(range(1, 51))
    runs = [_complete_eval_run(seed) for seed in seeds[:30]]
    sim_head = "7476a81954020087da31d41d16fddf475746ec2d"
    cache = {
        "schema_version": parent_cache_mod.CACHE_SCHEMA,
        "weight_sha256": parent_cache_mod.sha256(weight),
        "seeds": seeds,
        "mcts_sims": 2000,
        "simulator_id": parent_cache_mod.simulator_id(sim_head),
        "runs": runs,
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"v14_eval_caches": {"dev_parent": cache}}))
    seed_file = tmp_path / "dev-seeds.txt"
    drifted = list(seeds)
    drifted[-1] = 999999
    seed_file.write_text("\n".join(str(seed) for seed in drifted) + "\n", encoding="utf-8")

    assert not parent_cache_mod.materialize_cached_parent(
        state_path=state_path,
        parent_weight=weight,
        dev_seed_file=seed_file,
        output_dir=tmp_path / "seed-drift",
        mcts_sims=2000,
        sim_head=sim_head,
    )


def test_v16_stagnation_rescue_uses_stronger_teacher_signal():
    state = {
        "schema_version": "sts1-armg-ppo-v14-loop-state-v1",
        "round_index": 200,
        "stagnation_count": 54,
        "last_decision": "HOLD_PARENT",
        "last_dev_gate": {
            "win_delta": -3,
            "mean_paired_floor_delta": -0.8,
        },
    }
    order = [
        "attempt",
        "retry_count",
        "games_per_worker",
        "mcts_sims",
        "max_rounds",
        "ppo_version",
        "strategy_teacher_coef",
        "strategy_teacher_max_examples",
        "strategy_teacher_required",
        "candidate_threads",
    ]
    control = {
        "attempt": "200",
        "retry_count": "0",
        "games_per_worker": "80",
        "mcts_sims": "2000",
        "max_rounds": "0",
        "ppo_version": "1.5",
        "strategy_teacher_coef": "0.01",
        "strategy_teacher_max_examples": "512",
        "strategy_teacher_required": "1",
        "candidate_threads": "4",
    }

    updated, report = adapt_mod.adapt(
        state=state,
        control_order=order,
        control=control,
    )

    assert report["rescue_mode"] is True
    assert report["profile"]["name"] == "teacher_reanchor"
    assert updated["strategy_teacher_coef"] == "0.05"
    assert updated["strategy_teacher_max_examples"] == "1024"
    assert updated["mcts_sims"] == "2000"
    assert report["production_champion_changed"] is False


REPLAY_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v14_replay.py"
REPLAY_SPEC = importlib.util.spec_from_file_location(
    "sts1_armg_ppo_v14_replay_test",
    REPLAY_SCRIPT,
)
assert REPLAY_SPEC and REPLAY_SPEC.loader
replay_mod = importlib.util.module_from_spec(REPLAY_SPEC)
sys.modules[REPLAY_SPEC.name] = replay_mod
REPLAY_SPEC.loader.exec_module(replay_mod)


def _replay_game(seed: int, *, victory: bool, floor: int, source_round: int):
    return {
        "seed": seed,
        "checkpoint_id": "parent",
        "source_round": source_round,
        "temperature": 1.0,
        "victory": victory,
        "final_floor": floor,
        "action": [0],
    }


def test_v17_rescue_replay_mixes_wins_boss_failures_and_recent_data():
    games = [
        _replay_game(1, victory=True, floor=51, source_round=1),
        _replay_game(2, victory=True, floor=51, source_round=2),
        _replay_game(3, victory=True, floor=51, source_round=3),
        _replay_game(4, victory=False, floor=50, source_round=1),
        _replay_game(5, victory=False, floor=49, source_round=2),
        _replay_game(6, victory=False, floor=48, source_round=3),
        _replay_game(7, victory=False, floor=12, source_round=100),
        _replay_game(8, victory=False, floor=10, source_round=101),
        _replay_game(9, victory=False, floor=8, source_round=102),
    ]

    selected = replay_mod._select_rescue_replay(
        games,
        max_games=6,
        decision_budget=6,
    )

    seeds = {int(game["seed"]) for game in selected}
    assert len(selected) == 6
    assert len(seeds) == 6
    assert sum(bool(game["victory"]) for game in selected) >= 2
    assert sum(replay_mod._is_boss_failure(game) for game in selected) >= 2
    assert any(int(game["source_round"]) >= 101 and not game["victory"] for game in selected)


GATE_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v14_gate.py"
GATE_SPEC = importlib.util.spec_from_file_location(
    "sts1_armg_ppo_v14_gate_v16_test",
    GATE_SCRIPT,
)
assert GATE_SPEC and GATE_SPEC.loader
gate_mod = importlib.util.module_from_spec(GATE_SPEC)
sys.modules[GATE_SPEC.name] = gate_mod
GATE_SPEC.loader.exec_module(gate_mod)


def _gate_run(seed: int, *, win: bool, floor: int):
    return {
        "seed": seed,
        "result": "PASS_SIMULATOR_COMPLETE_RUN",
        "outcome": "victory" if win else "defeat",
        "final_floor": floor,
        "illegal_action_count": 0,
        "crash_count": 0,
        "timeout_count": 0,
        "remote_error_count": 0,
    }


def test_v16_dev_gate_does_not_promote_floor_only_improvement():
    parent = [_gate_run(seed, win=seed <= 4, floor=30) for seed in range(1, 31)]
    candidate = [_gate_run(seed, win=seed <= 4, floor=45) for seed in range(1, 31)]
    result = gate_mod._dev_gate(parent, candidate)
    assert result["win_delta"] == 0
    assert result["mean_paired_floor_delta"] > 0
    assert result["status"] == "HOLD"
    assert "no_dev_win_improvement" in result["reasons"]


def test_v16_dev_gate_promotes_real_win_improvement():
    parent = [_gate_run(seed, win=seed <= 4, floor=30) for seed in range(1, 31)]
    candidate = [_gate_run(seed, win=seed <= 5, floor=30) for seed in range(1, 31)]
    result = gate_mod._dev_gate(parent, candidate)
    assert result["win_delta"] == 1
    assert result["status"] == "PASS"
    assert result["decision"] == "ADOPT_PARENT"


ROLLOUT_V17_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_rollout_v14.py"
ROLLOUT_V17_SPEC = importlib.util.spec_from_file_location(
    "sts1_armg_ppo_rollout_v17_test",
    ROLLOUT_V17_SCRIPT,
)
assert ROLLOUT_V17_SPEC and ROLLOUT_V17_SPEC.loader
rollout_v17 = importlib.util.module_from_spec(ROLLOUT_V17_SPEC)
sys.modules[ROLLOUT_V17_SPEC.name] = rollout_v17
ROLLOUT_V17_SPEC.loader.exec_module(rollout_v17)


def test_v17_terminal_reward_is_victory_first():
    defeat = rollout_v17._v17_terminal_bonus(50, victory=False)
    victory = rollout_v17._v17_terminal_bonus(50, victory=True)
    early_defeat = rollout_v17._v17_terminal_bonus(16, victory=False)
    assert victory - defeat == 3.25
    assert victory > 10 * max(0.01, defeat)
    assert defeat > early_defeat


def test_v17_strategy_teacher_filters_low_confidence_and_low_margin(tmp_path):
    import json

    path = tmp_path / "strategy-replay.jsonl"
    strong = _strategy_row("map", current=0, teacher=1, priority=3.0)
    strong["confidence_weight"] = 0.75
    strong["teacher_margin"] = 4.0

    weak_conf = _strategy_row("shop", current=0, teacher=1, priority=4.0)
    weak_conf["confidence_weight"] = 0.50
    weak_conf["teacher_margin"] = 9.0

    weak_margin = _strategy_row("rest", current=0, teacher=1, priority=5.0)
    weak_margin["confidence_weight"] = 1.0
    weak_margin["teacher_margin"] = 1.0

    path.write_text(
        "".join(json.dumps(row) + "\n" for row in [strong, weak_conf, weak_margin]),
        encoding="utf-8",
    )

    selected = train_mod.load_strategy_teacher_examples(
        path,
        max_examples=16,
        combat_policy="mcts_2000",
        min_confidence_weight=0.75,
        min_teacher_margin=3.0,
    )
    assert len(selected) == 1
    assert selected[0]["kind"] == "map"


def test_v17_strategy_teacher_can_require_consensus_metadata(tmp_path):
    import json

    path = tmp_path / "strategy-replay.jsonl"
    accepted = _strategy_row("map", current=0, teacher=1, priority=3.0)
    accepted["teacher_consensus_fraction"] = 0.75
    rejected = _strategy_row("shop", current=0, teacher=1, priority=3.0)
    rejected["teacher_consensus_fraction"] = 0.50
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in [accepted, rejected]),
        encoding="utf-8",
    )
    selected = train_mod.load_strategy_teacher_examples(
        path,
        max_examples=16,
        combat_policy="mcts_2000",
        min_consensus_fraction=0.75,
    )
    assert len(selected) == 1
    assert selected[0]["kind"] == "map"
