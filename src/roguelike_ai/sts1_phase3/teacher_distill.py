"""Public-state MCTS Teacher distillation for STS1 Student v1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .base_policy import STS1_BASE_POLICY_ID
from .frozen_student import (
    FrozenStudentError,
    normalize_action_payload,
    project_policy_observation,
    sha256_json,
)
from .student_v1_ppo import StudentV1Error, StudentV1PPO


TEACHER_DATASET_SCHEMA_VERSION = "sts1-mcts-teacher-public-v1"


class TeacherDistillError(ValueError):
    """Teacher evidence or distillation contract was violated."""


@dataclass(frozen=True)
class TeacherExample:
    observation: Mapping[str, Any]
    legal_actions: tuple[Mapping[str, Any], ...]
    legal_action_ids: tuple[str, ...]
    teacher_action_index: int
    teacher_action_id: str

    def __post_init__(self) -> None:
        if not self.legal_actions:
            raise TeacherDistillError("teacher example must contain legal actions")
        if len(self.legal_actions) != len(self.legal_action_ids):
            raise TeacherDistillError("teacher legal action/id length mismatch")
        if not 0 <= self.teacher_action_index < len(self.legal_actions):
            raise TeacherDistillError("teacher action index outside legal actions")
        if self.teacher_action_id != self.legal_action_ids[self.teacher_action_index]:
            raise TeacherDistillError("teacher action identity mismatch")


def _example_from_row(row: Mapping[str, Any]) -> TeacherExample:
    state = row.get("public_state")
    if not isinstance(state, Mapping):
        raise TeacherDistillError("teacher row is missing public_state")
    try:
        observation = project_policy_observation(state)
    except FrozenStudentError as exc:
        raise TeacherDistillError(str(exc)) from exc

    raw_actions = state.get("legal_actions")
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes, bytearray)):
        raise TeacherDistillError("teacher public_state legal_actions must be an array")
    actions: list[Mapping[str, Any]] = []
    action_ids: list[str] = []
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise TeacherDistillError("teacher legal action must be an object")
        try:
            payload = normalize_action_payload(raw)
        except FrozenStudentError as exc:
            raise TeacherDistillError(str(exc)) from exc
        actions.append(payload)
        action_ids.append(sha256_json(payload))
    if not actions:
        raise TeacherDistillError("teacher legal_actions must be non-empty")

    try:
        index = int(row["teacher_action_index"])
        action_id = str(row["teacher_action_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TeacherDistillError(f"invalid teacher action label: {exc}") from exc

    return TeacherExample(
        observation=observation,
        legal_actions=tuple(actions),
        legal_action_ids=tuple(action_ids),
        teacher_action_index=index,
        teacher_action_id=action_id,
    )


def read_teacher_evidence(path: Path) -> tuple[TeacherExample, ...]:
    examples: list[TeacherExample] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TeacherDistillError(f"could not read teacher evidence: {exc}") from exc

    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TeacherDistillError(f"invalid teacher evidence JSONL: {exc}") from exc
        if not isinstance(row, Mapping):
            raise TeacherDistillError("teacher evidence row must be an object")
        if row.get("type") == "mcts_teacher_decision":
            examples.append(_example_from_row(row))

    if not examples:
        raise TeacherDistillError("teacher evidence contains no MCTS decisions")
    return tuple(examples)


def _teacher_loss(
    policy: StudentV1PPO,
    example: TeacherExample,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    _, base_logits, logits, _, _ = policy._policy_tensors(
        example.observation,
        example.legal_actions,
    )
    target = torch.tensor(
        [example.teacher_action_index],
        dtype=torch.long,
        device=policy.device,
    )
    ce = F.cross_entropy(logits.unsqueeze(0), target)

    with torch.no_grad():
        baseline_probs = torch.softmax(base_logits, dim=0)
    current_log_probs = torch.log_softmax(logits, dim=0)
    anchor_kl = torch.sum(
        baseline_probs
        * (
            torch.log(baseline_probs.clamp_min(1e-12))
            - current_log_probs
        )
    )
    correct = int(torch.argmax(logits).item()) == example.teacher_action_index
    return ce, anchor_kl, correct


def distill_mcts_teacher(
    policy: StudentV1PPO,
    examples: Sequence[TeacherExample],
    *,
    epochs: int = 2,
    batch_size: int = 32,
    seed: int = 0,
) -> dict[str, float | int]:
    """Supervise the residual actor toward public-state MCTS decisions."""

    if not examples:
        raise StudentV1Error("Teacher distillation requires examples")
    if epochs < 1 or batch_size < 1:
        raise StudentV1Error("Teacher distillation epochs/batch_size must be positive")

    rng = random.Random(seed)
    total_ce = 0.0
    total_anchor = 0.0
    batches = 0

    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            losses: list[torch.Tensor] = []
            ce_rows: list[torch.Tensor] = []
            anchor_rows: list[torch.Tensor] = []
            for index in indices:
                ce, anchor, _ = _teacher_loss(policy, examples[index])
                ce_rows.append(ce)
                anchor_rows.append(anchor)
                losses.append(
                    ce + policy.config.baseline_anchor_coef * anchor
                )
            loss = torch.stack(losses).mean()
            policy.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                policy.model.parameters(),
                policy.config.max_grad_norm,
            )
            policy.optimizer.step()
            total_ce += float(torch.stack(ce_rows).mean().item())
            total_anchor += float(torch.stack(anchor_rows).mean().item())
            batches += 1

    if batches == 0:
        raise StudentV1Error("Teacher distillation produced no batches")

    correct = 0
    with torch.no_grad():
        for example in examples:
            _, _, is_correct = _teacher_loss(policy, example)
            correct += int(is_correct)

    return {
        "schema_version": TEACHER_DATASET_SCHEMA_VERSION,
        "base_policy_id": STS1_BASE_POLICY_ID,
        "examples": len(examples),
        "epochs": epochs,
        "batches": batches,
        "teacher_top1_accuracy": correct / len(examples),
        "cross_entropy": total_ce / batches,
        "baseline_anchor_kl": total_anchor / batches,
    }


__all__ = [
    "TEACHER_DATASET_SCHEMA_VERSION",
    "TeacherDistillError",
    "TeacherExample",
    "distill_mcts_teacher",
    "read_teacher_evidence",
]
