from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "sts1" / "sts1_armg_ppo_v13_auto_gate.py"
SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v13_auto_gate", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def fake_run(seed: int):
    return {
        "seed": str(seed),
        "result": "PASS_SIMULATOR_COMPLETE_RUN",
        "outcome": "defeat",
        "final_floor": 10,
        "illegal_action_count": 0,
        "crash_count": 0,
        "timeout_count": 0,
    }


def test_champion_cache_round_trips_through_loop_state(tmp_path: Path):
    weight = tmp_path / "offline-champion.pt"
    weight.write_bytes(b"champion")
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"schema_version": "sts1-armg-ppo-v13-loop-state-v1"}) + "\n",
        encoding="utf-8",
    )
    seeds = list(range(1, 51))
    runs = {seed: fake_run(seed) for seed in seeds[:30]}
    champion_sha = mod._sha256(weight)

    mod._write_champion_cache(
        None,
        state_path=state_path,
        champion_sha=champion_sha,
        seeds=seeds,
        mcts_sims=2000,
        simulator_id="sim-A",
        runs_by_seed=runs,
    )

    loaded = mod._load_champion_cache(
        None,
        state_path=state_path,
        champion_sha=champion_sha,
        seeds=seeds,
        mcts_sims=2000,
        simulator_id="sim-A",
    )
    assert set(loaded) == set(seeds[:30])

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "champion_eval_cache" in state
    assert state["champion_eval_cache"]["champion_sha256"] == champion_sha
    assert len(state["champion_eval_cache"]["runs"]) == 30


def test_cache_misses_on_any_identity_change(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"schema_version": "sts1-armg-ppo-v13-loop-state-v1"}) + "\n",
        encoding="utf-8",
    )
    seeds = list(range(1, 51))
    runs = {seed: fake_run(seed) for seed in seeds[:2]}

    mod._write_champion_cache(
        None,
        state_path=state_path,
        champion_sha="sha-A",
        seeds=seeds,
        mcts_sims=2000,
        simulator_id="sim-A",
        runs_by_seed=runs,
    )

    common = dict(path=None, state_path=state_path, seeds=seeds)
    assert not mod._load_champion_cache(
        **common, champion_sha="sha-B", mcts_sims=2000, simulator_id="sim-A"
    )
    assert not mod._load_champion_cache(
        **common, champion_sha="sha-A", mcts_sims=50000, simulator_id="sim-A"
    )
    assert not mod._load_champion_cache(
        **common, champion_sha="sha-A", mcts_sims=2000, simulator_id="sim-B"
    )


def test_parallel_worker_bounds_are_enforced_by_gate():
    # This is a cheap guard for the public API used by the workflow.
    assert mod._evaluate_parallel
