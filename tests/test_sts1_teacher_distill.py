from __future__ import annotations

import hashlib
import json

from roguelike_ai.sts1_phase3.frozen_student import (
    FrozenStudentV0,
    normalize_action_payload,
    sha256_json,
)
from roguelike_ai.sts1_phase3.student_v1_ppo import StudentV1Config, StudentV1PPO
from roguelike_ai.sts1_phase3.teacher_distill import (
    distill_mcts_teacher,
    read_teacher_evidence,
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


def test_mcts_teacher_distillation_moves_policy_toward_teacher(tmp_path) -> None:
    state = _state()
    teacher_payload = normalize_action_payload(state["legal_actions"][1])
    teacher_id = sha256_json(teacher_payload)
    evidence = tmp_path / "teacher.ndjson"
    row = {
        "type": "mcts_teacher_decision",
        "public_state": state,
        "teacher_action_index": 1,
        "teacher_action_id": teacher_id,
        "mcts_sims": 2000,
    }
    evidence.write_text(
        "".join(json.dumps(row) + "\n" for _ in range(24)),
        encoding="utf-8",
    )

    examples = read_teacher_evidence(evidence)
    baseline = FrozenStudentV0(
        weights={},
        artifact_sha256=hashlib.sha256(b"synthetic-v0").hexdigest(),
    )
    policy = StudentV1PPO(
        baseline,
        config=StudentV1Config(
            state_dim=64,
            action_dim=32,
            hidden_dim=16,
            learning_rate=0.02,
            baseline_anchor_coef=0.0,
            max_grad_norm=1.0,
        ),
    )

    before = policy.select_action(state, require_command=False)
    stats = distill_mcts_teacher(
        policy,
        examples,
        epochs=10,
        batch_size=8,
        seed=7,
    )
    after = policy.select_action(state, require_command=False)

    assert before.action_index == 0
    assert after.action_index == 1
    assert stats["examples"] == 24
    assert stats["teacher_top1_accuracy"] == 1.0


def test_teacher_evidence_strips_transport_and_keeps_action_identity(tmp_path) -> None:
    state = _state()
    teacher_payload = normalize_action_payload(state["legal_actions"][0])
    evidence = tmp_path / "teacher.ndjson"
    evidence.write_text(
        json.dumps({
            "type": "mcts_teacher_decision",
            "public_state": state,
            "teacher_action_index": 0,
            "teacher_action_id": sha256_json(teacher_payload),
        }) + "\n",
        encoding="utf-8",
    )
    example = read_teacher_evidence(evidence)[0]
    assert "command" not in example.legal_actions[0]
    assert "legal_actions" not in example.observation
    assert example.teacher_action_id == example.legal_action_ids[0]
