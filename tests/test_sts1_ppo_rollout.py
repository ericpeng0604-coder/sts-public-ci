from __future__ import annotations

import hashlib

import pytest

from roguelike_ai.sts1_phase3.ppo_rollout import (
    PPORolloutError,
    PPOEpisode,
    episode_from_simulator_evidence,
    make_transition,
    public_progress_reward,
    read_rollout_shard,
    write_rollout_shard,
)
from roguelike_ai.sts1_phase3.self_improve_loop import RolloutIdentity


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(generation: int = 1) -> RolloutIdentity:
    return RolloutIdentity(
        generation=generation,
        policy_sha256=_sha("student-v1"),
        simulator_sha="7476a81954020087da31d41d16fddf475746ec2d",
        public_state_schema="sts1-public-state-v1",
        action_schema="sts1-public-action-v1",
    )


def _state() -> dict:
    return {
        "hp": 70,
        "max_hp": 80,
        "block": 0,
        "energy": 3,
        "hand": [{"id": "Strike_R", "cost": 1}],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [{"name": "Jaw Worm", "hp": 40, "intent": "ATTACK"}],
        "turn": 1,
        "combat_active": True,
        "relics": [],
        "potions": [],
        "gold": 99,
        "floor": 1,
        "act": 1,
        "character": "IRONCLAD",
        "ascension_level": 0,
        "room": "COMBAT",
        "screen_type": "NONE",
        "screen_choices": [],
        "rewards": [],
        "map_choices": [],
        "legal_actions": [
            {"kind": "play_card", "hand_index": 1, "target_index": 0, "command": "play 1 0"},
            {"kind": "end_turn", "command": "end"},
        ],
    }


def _episode() -> PPOEpisode:
    state = _state()
    first = make_transition(
        state,
        episode_id="ep-1",
        step_index=0,
        selected_action_index=0,
        reward=0.1,
        done=False,
        old_log_prob=-0.4,
        old_value=0.2,
    )
    second = make_transition(
        state,
        episode_id="ep-1",
        step_index=1,
        selected_action_index=1,
        reward=1.0,
        done=True,
        old_log_prob=-0.6,
        old_value=0.4,
    )
    return PPOEpisode("ep-1", (first, second), "victory", 51)


def test_transition_strips_transport_command_and_keeps_public_state_only() -> None:
    transition = _episode().transitions[0]
    assert "command" not in transition.legal_actions[0]
    assert "seed" not in transition.observation
    assert transition.selected_action_id == transition.legal_action_ids[0]


def test_transition_rejects_hidden_seed() -> None:
    state = _state()
    state["seed"] = 123
    with pytest.raises(PPORolloutError, match="forbidden hidden/provenance key"):
        make_transition(
            state,
            episode_id="ep",
            step_index=0,
            selected_action_index=0,
            reward=0,
            done=True,
            old_log_prob=-0.2,
            old_value=0,
        )


def test_episode_requires_contiguous_steps_and_terminal_done() -> None:
    transition = make_transition(
        _state(),
        episode_id="ep",
        step_index=1,
        selected_action_index=0,
        reward=0,
        done=True,
        old_log_prob=-0.2,
        old_value=0,
    )
    with pytest.raises(PPORolloutError, match="contiguous"):
        PPOEpisode("ep", (transition,), "defeat", 5)


def test_rollout_shard_round_trip_and_checksum(tmp_path) -> None:
    payload = tmp_path / "rollout.jsonl"
    manifest = tmp_path / "manifest.json"
    identity = _identity()
    write_rollout_shard(
        payload,
        manifest,
        identity=identity,
        shard_id="worker-01",
        episodes=[_episode()],
    )
    loaded = read_rollout_shard(payload, manifest, expected_identity=identity)
    assert loaded == (_episode(),)

    payload.write_text(payload.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(PPORolloutError, match="checksum mismatch"):
        read_rollout_shard(payload, manifest, expected_identity=identity)


def test_rollout_shard_rejects_stale_generation(tmp_path) -> None:
    payload = tmp_path / "rollout.jsonl"
    manifest = tmp_path / "manifest.json"
    write_rollout_shard(
        payload,
        manifest,
        identity=_identity(generation=1),
        shard_id="worker-01",
        episodes=[_episode()],
    )
    with pytest.raises(PPORolloutError, match="stale rollout rejected"):
        read_rollout_shard(payload, manifest, expected_identity=_identity(generation=2))


def test_simulator_evidence_becomes_complete_public_episode(tmp_path) -> None:
    state = _state()
    state2 = _state()
    state2["hp"] = 65
    state2["floor"] = 2
    evidence = tmp_path / "sim.jsonl"
    rows = [
        {
            "type": "ppo_decision",
            "public_state": state,
            "action_index": 0,
            "old_log_prob": -0.5,
            "old_value": 0.1,
        },
        {
            "type": "ppo_decision",
            "public_state": state2,
            "action_index": 1,
            "old_log_prob": -0.6,
            "old_value": 0.2,
        },
        {
            "type": "summary",
            "result": "PASS_SIMULATOR_COMPLETE_RUN",
            "ppo_collection": True,
            "outcome": "victory",
            "final_hp": 68,
            "final_floor": 3,
        },
    ]
    evidence.write_text(
        "".join(__import__("json").dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    episode = episode_from_simulator_evidence(evidence, episode_id="sim-1")
    assert episode.outcome == "victory"
    assert len(episode.transitions) == 2
    assert episode.transitions[0].done is False
    assert episode.transitions[-1].done is True
    assert episode.transitions[-1].reward > 1.0


def test_blocked_simulator_evidence_is_never_training_data(tmp_path) -> None:
    evidence = tmp_path / "blocked.jsonl"
    evidence.write_text(
        __import__("json").dumps({
            "type": "summary",
            "result": "BLOCKED_SIMULATOR",
            "ppo_collection": True,
            "outcome": "unknown",
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PPORolloutError, match="cannot become PPO training data"):
        episode_from_simulator_evidence(evidence, episode_id="blocked")


def test_dense_reward_prefers_enemy_damage_and_penalizes_player_damage() -> None:
    before = _state()
    after_attack = _state()
    after_attack["enemies"] = [{"name": "Jaw Worm", "hp": 34, "intent": "ATTACK"}]

    noop = public_progress_reward(
        before,
        after_hp=70,
        after_floor=1,
        after_state=_state(),
    )
    attack = public_progress_reward(
        before,
        after_hp=70,
        after_floor=1,
        after_state=after_attack,
    )
    hurt = public_progress_reward(
        before,
        after_hp=64,
        after_floor=1,
        after_state=_state(),
    )

    assert attack > noop
    assert hurt < noop
