from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
GATE_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v14_gate.py"
STATE_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v14_state.py"
TRAIN_SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_train_v14_sharded.py"

SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v14_gate", GATE_SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


TRAIN_SPEC = importlib.util.spec_from_file_location(
    "sts1_armg_ppo_v14_train",
    TRAIN_SCRIPT,
)
assert TRAIN_SPEC and TRAIN_SPEC.loader
train = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train
TRAIN_SPEC.loader.exec_module(train)


def test_gae_targets_are_deterministic_and_episode_bounded():
    reward = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    done = np.array([False, True, False, True])
    values = np.array([0.5, 0.25, 1.0, 0.75], np.float32)
    adv1, ret1 = train.gae_targets(
        reward,
        done,
        values,
        gamma=0.99,
        lam=0.95,
    )
    adv2, ret2 = train.gae_targets(
        reward,
        done,
        values,
        gamma=0.99,
        lam=0.95,
    )
    np.testing.assert_allclose(adv1, adv2)
    np.testing.assert_allclose(ret1, ret2)
    assert ret1[1] == np.float32(2.0)
    assert ret1[3] == np.float32(4.0)


def test_explained_variance_rewards_better_predictions():
    target = np.array([0.0, 1.0, 2.0, 3.0], np.float32)
    poor = np.zeros(4, np.float32)
    good = np.array([0.0, 0.9, 2.1, 3.0], np.float32)
    assert train.explained_variance(good, target) > train.explained_variance(
        poor,
        target,
    )


def test_discounted_returns_reset_at_episode_boundaries():
    reward = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    done = np.array([False, True, False, True])
    got = train.discounted_returns(reward, done, gamma=0.5)
    np.testing.assert_allclose(
        got,
        np.array([2.0, 2.0, 5.0, 4.0], np.float32),
    )


def test_poisoned_persistent_critic_requests_reset():
    assert train.should_reset_critic(
        critic_loaded=True,
        health_explained_variance=-0.11,
    )
    assert not train.should_reset_critic(
        critic_loaded=True,
        health_explained_variance=-0.05,
    )
    assert not train.should_reset_critic(
        critic_loaded=False,
        health_explained_variance=-100.0,
    )


def test_curriculum_focus_uses_dominant_failure_band_and_ignores_wins():
    focus, counts = train.choose_curriculum_focus(
        [10, 12, 20, 30, 45, 50],
        [False, False, False, False, True, False],
    )
    assert counts == {"act1": 2, "act2": 2, "act3": 1}
    assert focus == "act2"


def test_curriculum_focus_is_none_when_every_game_wins():
    focus, counts = train.choose_curriculum_focus(
        [51, 51, 51],
        [True, True, True],
    )
    assert counts == {"act1": 0, "act2": 0, "act3": 0}
    assert focus == "none"



def run(seed: int, floor: int, *, win: bool = False, illegal: int = 0):
    return {
        "seed": seed,
        "result": "PASS_SIMULATOR_COMPLETE_RUN",
        "outcome": "victory" if win else "defeat",
        "final_floor": floor,
        "illegal_action_count": illegal,
        "timeout_count": 0,
        "crash_count": 0,
        "remote_error_count": 0,
    }


def test_identical_candidate_is_not_adopted():
    parent = [run(i, 35, win=i in {1, 2}) for i in range(30)]
    candidate = [dict(row) for row in parent]
    result = gate._dev_gate(parent, candidate)
    assert result["status"] == "HOLD"
    assert result["decision"] == "HOLD_PARENT"
    assert result["win_delta"] == 0
    assert result["mean_paired_floor_delta"] == 0.0


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


def test_dev_gate_rejects_safety_failure_even_with_extra_win():
    parent = [run(i, 35, win=i == 1) for i in range(30)]
    candidate = [run(i, 35, win=i in {1, 2}) for i in range(30)]
    candidate[3]["illegal_action_count"] = 1
    result = gate._dev_gate(parent, candidate)
    assert result["status"] == "HOLD"
    assert "candidate_safety_failure" in result["reasons"]


def test_final_seed_file_is_disjoint_from_upstream_dev_set():
    final_path = ROOT / "control" / "sts1-v14-final-promotion-seeds-50.txt"
    final = {
        int(x)
        for x in final_path.read_text().splitlines()
        if x.strip() and not x.startswith("#")
    }
    dev = {
        36478011,37788393,74056706,74327186,82434163,85904294,118957286,
        139632048,148131968,174304798,190567680,220682708,274997078,
        292341318,321103947,341684210,357136893,361929013,367225452,
        382121796,388733061,407994558,434145318,434519731,469494371,
        523373928,572162032,620908546,668174005,680768576,711099439,
        748271783,753056956,808101866,815119230,845474044,848824510,
        857886683,879595393,889064010,909136291,919159433,936753763,
        937579329,946665409,950386020,951573210,952299191,981274517,
        993164492,
    }
    assert len(final) == 50
    assert not (final & dev)


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
    assert state["production_champion_replaced"] is False
    assert (state_dir / "offline-champion.pt").read_bytes() == b"new-parent"
    assert (state_dir / "training-critic.pt").read_bytes() == b"persistent-critic"
    assert (state_dir / "pending-real-game.pt").read_bytes() == b"new-parent"


def test_rejected_actor_still_keeps_new_critic(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "offline-champion.pt").write_bytes(b"parent")
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": "sts1-armg-ppo-v14-loop-state-v1",
                "round_index": 1,
                "generation": 0,
                "parent_generation": 0,
                "stagnation_count": 0,
            }
        ) + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.pt"
    critic = tmp_path / "critic.pt"
    candidate.write_bytes(b"bad-candidate")
    critic.write_bytes(b"better-critic")
    summary = tmp_path / "gate.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "sts1-armg-ppo-v14-two-level-gate-v1",
                "decision": "HOLD_PARENT",
                "final_readiness": "NOT_READY",
                "dev_gate": {"status": "HOLD", "win_delta": 0},
                "final_gate_30": {"status": "SKIPPED"},
                "final_gate_50": {"status": "SKIPPED"},
            }
        ) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable, str(STATE_SCRIPT),
            "--state-dir", str(state_dir),
            "--candidate", str(candidate),
            "--candidate-critic", str(critic),
            "--gate-summary", str(summary),
            "--run-id", "124",
        ],
        check=True,
    )
    assert (state_dir / "offline-champion.pt").read_bytes() == b"parent"
    assert (state_dir / "training-critic.pt").read_bytes() == b"better-critic"



def test_parent_promotion_clears_ppo_replay_but_keeps_elite_archive(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "offline-champion.pt").write_bytes(b"parent")
    (state_dir / "ppo-replay.npz").write_bytes(b"same-parent-replay")
    (state_dir / "elite-replay.npz").write_bytes(b"cross-generation-elite")
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": "sts1-armg-ppo-v14-loop-state-v1",
                "round_index": 4,
                "generation": 1,
                "parent_generation": 1,
                "stagnation_count": 2,
            }
        ) + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.pt"
    critic = tmp_path / "critic.pt"
    candidate.write_bytes(b"new-parent")
    critic.write_bytes(b"critic")
    summary = tmp_path / "gate.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "sts1-armg-ppo-v14-two-level-gate-v1",
                "decision": "ADOPT_PARENT",
                "final_readiness": "NOT_READY",
                "dev_gate": {"status": "PASS", "win_delta": 1},
                "final_gate_30": {"status": "HOLD"},
                "final_gate_50": {"status": "SKIPPED"},
            }
        ) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(STATE_SCRIPT),
            "--state-dir", str(state_dir),
            "--candidate", str(candidate),
            "--candidate-critic", str(critic),
            "--gate-summary", str(summary),
            "--run-id", "125",
        ],
        check=True,
    )

    assert not (state_dir / "ppo-replay.npz").exists()
    assert (state_dir / "elite-replay.npz").read_bytes() == b"cross-generation-elite"


def test_parent_hold_preserves_both_replay_archives(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "offline-champion.pt").write_bytes(b"parent")
    (state_dir / "ppo-replay.npz").write_bytes(b"same-parent-replay")
    (state_dir / "elite-replay.npz").write_bytes(b"elite")
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": "sts1-armg-ppo-v14-loop-state-v1",
                "round_index": 5,
                "generation": 1,
                "parent_generation": 1,
                "stagnation_count": 1,
            }
        ) + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.pt"
    critic = tmp_path / "critic.pt"
    candidate.write_bytes(b"candidate")
    critic.write_bytes(b"critic")
    summary = tmp_path / "gate.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "sts1-armg-ppo-v14-two-level-gate-v1",
                "decision": "HOLD_PARENT",
                "final_readiness": "NOT_READY",
                "dev_gate": {"status": "HOLD", "win_delta": 0},
                "final_gate_30": {"status": "SKIPPED"},
                "final_gate_50": {"status": "SKIPPED"},
            }
        ) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(STATE_SCRIPT),
            "--state-dir", str(state_dir),
            "--candidate", str(candidate),
            "--candidate-critic", str(critic),
            "--gate-summary", str(summary),
            "--run-id", "126",
        ],
        check=True,
    )

    assert (state_dir / "ppo-replay.npz").read_bytes() == b"same-parent-replay"
    assert (state_dir / "elite-replay.npz").read_bytes() == b"elite"
