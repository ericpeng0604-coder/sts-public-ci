from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
GATE_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v14_gate.py"
STATE_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v14_state.py"

SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v14_gate", GATE_SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def run(seed: int, floor: int, *, win: bool = False):
    return {
        "seed": seed,
        "result": "PASS_SIMULATOR_COMPLETE_RUN",
        "outcome": "victory" if win else "defeat",
        "final_floor": floor,
        "illegal_action_count": 0,
        "timeout_count": 0,
        "crash_count": 0,
        "remote_error_count": 0,
    }


def test_dev_gate_accepts_small_paired_floor_progress_without_win_regression():
    parent = [run(i, 30 + (i % 5), win=i in {1, 2}) for i in range(30)]
    candidate = [run(i, 31 + (i % 5), win=i in {1, 2}) for i in range(30)]
    result = gate._dev_gate(parent, candidate)
    assert result["status"] == "PASS"
    assert result["decision"] == "ADOPT_PARENT"
    assert result["win_delta"] == 0
    assert result["mean_paired_floor_delta"] == 1.0


def test_dev_gate_accepts_one_more_win():
    parent = [run(i, 35, win=i == 1) for i in range(30)]
    candidate = [run(i, 35, win=i in {1, 2}) for i in range(30)]
    result = gate._dev_gate(parent, candidate)
    assert result["status"] == "PASS"
    assert result["win_delta"] == 1


def test_dev_gate_rejects_win_regression_even_if_floor_rises():
    parent = [run(i, 30, win=i in {1, 2}) for i in range(30)]
    candidate = [run(i, 35, win=i == 1) for i in range(30)]
    result = gate._dev_gate(parent, candidate)
    assert result["status"] == "HOLD"
    assert "dev_wins_regressed" in result["reasons"]


def test_dev_gate_rejects_noise_level_floor_change():
    parent = [run(i, 35) for i in range(30)]
    candidate = [run(i, 35 + (1 if i < 10 else 0)) for i in range(30)]
    result = gate._dev_gate(parent, candidate)
    assert result["status"] == "HOLD"
    assert "no_material_dev_improvement" in result["reasons"]


def test_v14_state_persists_parent_critic_and_pending_real_game(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "offline-champion.pt").write_bytes(b"old-parent")
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": "sts1-armg-ppo-v13-loop-state-v1",
                "round_index": 3,
                "generation": 0,
                "stagnation_count": 4,
                "production_champion_replaced": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.pt"
    critic = tmp_path / "critic.pt"
    candidate.write_bytes(b"new-parent")
    critic.write_bytes(b"persistent-critic")
    summary = tmp_path / "gate.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "sts1-armg-ppo-v14-two-level-gate-v1",
                "decision": "ADOPT_PARENT",
                "final_readiness": "READY_FOR_REAL_GAME",
                "dev_gate": {"status": "PASS", "win_delta": 1},
                "final_gate_30": {"status": "PASS"},
                "final_gate_50": {"status": "PASS"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(STATE_SCRIPT),
            "--state-dir",
            str(state_dir),
            "--candidate",
            str(candidate),
            "--candidate-critic",
            str(critic),
            "--gate-summary",
            str(summary),
            "--run-id",
            "123",
        ],
        check=True,
    )

    state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert state["schema_version"] == "sts1-armg-ppo-v14-loop-state-v1"
    assert state["last_decision"] == "ADOPT_PARENT"
    assert state["stagnation_count"] == 0
    assert state["parent_generation"] == 1
    assert state["pending_real_game_validation"] is True
    assert (state_dir / "offline-champion.pt").read_bytes() == b"new-parent"
    assert (state_dir / "training-critic.pt").read_bytes() == b"persistent-critic"
    assert (state_dir / "pending-real-game.pt").read_bytes() == b"new-parent"
