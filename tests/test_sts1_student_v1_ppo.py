from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch

from roguelike_ai.sts1_phase3.frozen_student import (
    FrozenStudentV0,
    normalize_action_payload,
    sha256_json,
)
from roguelike_ai.sts1_phase3.ppo_rollout import PPOEpisode, make_transition
from roguelike_ai.sts1_phase3.student_v1_ppo import (
    StudentV1Config,
    StudentV1PPO,
    file_sha256,
    ppo_update,
    teacher_bc_update,
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


def _baseline_prefers_end_turn() -> FrozenStudentV0:
    action = normalize_action_payload(_state()["legal_actions"][1])
    action_id = sha256_json(action)
    return FrozenStudentV0(
        weights={f"action.id={action_id}": 5.0},
        artifact_sha256=hashlib.sha256(b"synthetic-v0").hexdigest(),
    )


def _episode(policy: StudentV1PPO) -> PPOEpisode:
    state = _state()
    d0 = policy.sample_action(state, deterministic=False, require_command=False)
    t0 = make_transition(
        state,
        episode_id="ep-1",
        step_index=0,
        selected_action_index=d0.action_index,
        reward=0.0,
        done=False,
        old_log_prob=d0.log_prob,
        old_value=d0.value,
    )
    d1 = policy.sample_action(state, deterministic=False, require_command=False)
    t1 = make_transition(
        state,
        episode_id="ep-1",
        step_index=1,
        selected_action_index=d1.action_index,
        reward=1.0,
        done=True,
        old_log_prob=d1.log_prob,
        old_value=d1.value,
    )
    return PPOEpisode("ep-1", (t0, t1), "victory", 51)


def test_untrained_student_v1_preserves_frozen_v0_deterministic_choice() -> None:
    baseline = _baseline_prefers_end_turn()
    v1 = StudentV1PPO(
        baseline,
        config=StudentV1Config(state_dim=64, action_dim=32, hidden_dim=16),
    )
    v0_decision = baseline.select_action(_state(), require_command=False)
    v1_decision = v1.select_action(_state(), require_command=False)
    assert v0_decision.action_index == 1
    assert v1_decision.action_index == v0_decision.action_index
    assert v1_decision.action_id == v0_decision.action_id


def test_ppo_update_changes_residual_policy_without_mutating_v0() -> None:
    torch.manual_seed(7)
    baseline = _baseline_prefers_end_turn()
    before_weights = dict(baseline.weights)
    policy = StudentV1PPO(
        baseline,
        config=StudentV1Config(
            state_dim=64,
            action_dim=32,
            hidden_dim=16,
            learning_rate=1e-3,
            epochs=2,
            batch_size=2,
            baseline_anchor_coef=0.01,
        ),
    )
    episode = _episode(policy)
    before = {
        key: value.detach().clone()
        for key, value in policy.model.state_dict().items()
    }
    stats = ppo_update(policy, [episode])
    after = policy.model.state_dict()

    assert stats["transitions"] == 2
    assert stats["total_updates"] == 1
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert dict(baseline.weights) == before_weights


def test_student_v1_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(9)
    baseline = _baseline_prefers_end_turn()
    config = StudentV1Config(state_dim=64, action_dim=32, hidden_dim=16)
    policy = StudentV1PPO(baseline, config=config, generation=3)
    ppo_update(policy, [_episode(policy)])

    path = tmp_path / "candidate.pt"
    policy.save(path)
    assert len(file_sha256(path)) == 64

    restored = StudentV1PPO.load(path, baseline)
    assert restored.generation == 3
    assert restored.total_updates == policy.total_updates
    left = policy.select_action(_state(), require_command=False)
    right = restored.select_action(_state(), require_command=False)
    assert left.action_index == right.action_index
    assert left.action_id == right.action_id


def test_teacher_bc_warm_start_improves_teacher_agreement_without_mutating_v0() -> None:
    torch.manual_seed(11)
    baseline = _baseline_prefers_end_turn()
    frozen = dict(baseline.weights)
    policy = StudentV1PPO(
        baseline,
        config=StudentV1Config(
            state_dim=64,
            action_dim=32,
            hidden_dim=16,
            learning_rate=5e-3,
            epochs=1,
            batch_size=2,
        ),
    )

    state = _state()
    actions = tuple(
        normalize_action_payload(action)
        for action in state["legal_actions"]
    )
    teacher_examples = tuple(
        SimpleNamespace(
            observation={key: value for key, value in state.items() if key != "legal_actions"},
            action_payloads=actions,
            selected_index=0,
            tie_indices=(0,),
        )
        for _ in range(24)
    )

    stats = teacher_bc_update(policy, teacher_examples, epochs=20)
    assert stats["after_top1_accuracy"] > stats["before_top1_accuracy"]
    assert stats["after_tie_aware_accuracy"] >= stats["after_top1_accuracy"]
    assert dict(baseline.weights) == frozen


def test_teacher_bc_accepts_frozen_teacher_tie_set() -> None:
    torch.manual_seed(12)
    baseline = _baseline_prefers_end_turn()
    policy = StudentV1PPO(
        baseline,
        config=StudentV1Config(
            state_dim=64,
            action_dim=32,
            hidden_dim=16,
            learning_rate=1e-3,
            epochs=1,
            batch_size=2,
        ),
    )
    state = _state()
    actions = tuple(
        normalize_action_payload(action)
        for action in state["legal_actions"]
    )
    example = SimpleNamespace(
        observation={key: value for key, value in state.items() if key != "legal_actions"},
        action_payloads=actions,
        selected_index=0,
        tie_indices=(0, 1),
    )
    stats = teacher_bc_update(policy, [example], epochs=2)
    assert stats["examples"] == 1
    assert stats["updates"] == 2
