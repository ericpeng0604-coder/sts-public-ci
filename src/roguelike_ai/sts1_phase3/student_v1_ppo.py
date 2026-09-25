"""Student v1 residual PPO policy.

Student v0 stays frozen. Student v1 adds a trainable residual actor and critic
on top of the exact v0 action scores. The residual actor's final layer starts
at zero, so an untrained v1 has the same deterministic action choice as v0.

The PPO update follows the useful parts of AscensionAI:
- old log-prob/value rollouts
- GAE advantages
- clipped policy objective
- value loss + entropy bonus
- target-KL early stop
- frozen-baseline policy anchor to reduce catastrophic forgetting
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from .base_policy import STS1_BASE_POLICY_ID
from .frozen_student import (
    EXPECTED_MODEL_SHA256,
    FrozenStudentError,
    FrozenStudentV0,
    StudentDecision,
    _flatten,
    normalize_action_payload,
    project_policy_observation,
    sha256_json,
)
from .ppo_rollout import PPOEpisode, PPOTransition


STUDENT_V1_SCHEMA_VERSION = "sts1-student-v1-residual-ppo-v1"
CHECKPOINT_SCHEMA_VERSION = "sts1-student-v1-checkpoint-v1"


class StudentV1Error(ValueError):
    """Student v1 policy/checkpoint/training contract was violated."""


@dataclass(frozen=True)
class StudentV1Config:
    state_dim: int = 512
    action_dim: int = 256
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    baseline_anchor_coef: float = 0.01
    target_kl: float = 0.03
    max_grad_norm: float = 0.5
    epochs: int = 4
    batch_size: int = 64

    def __post_init__(self) -> None:
        for name in ("state_dim", "action_dim", "hidden_dim", "epochs", "batch_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "learning_rate", "gamma", "gae_lambda", "clip_range",
            "value_coef", "max_grad_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("entropy_coef", "baseline_anchor_coef", "target_kl"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def config_hash(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class StudentV1Decision:
    action_index: int
    action_id: str
    command: str | None
    score: float
    legal_action_count: int
    log_prob: float
    value: float

    def as_student_decision(self) -> StudentDecision:
        return StudentDecision(
            action_index=self.action_index,
            action_id=self.action_id,
            command=self.command,
            score=self.score,
            legal_action_count=self.legal_action_count,
        )


def _hash_features(features: Mapping[str, float], dim: int) -> torch.Tensor:
    vector = torch.zeros(dim, dtype=torch.float32)
    for key in sorted(features):
        value = float(features[key])
        if value == 0.0:
            continue
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dim
        sign = 1.0 if (digest[8] & 1) == 0 else -1.0
        vector[index] += sign * value
    return vector


def encode_state(observation: Mapping[str, Any], dim: int) -> torch.Tensor:
    projected = project_policy_observation(observation)
    features: dict[str, float] = {}
    _flatten(projected, "state", features)
    return _hash_features(features, dim)


def encode_action(action: Mapping[str, Any], dim: int) -> torch.Tensor:
    payload = normalize_action_payload(action)
    features: dict[str, float] = {}
    _flatten(payload, "action", features)
    features[f"action.id={sha256_json(payload)}"] = 1.0
    return _hash_features(features, dim)


def _baseline_scores(
    baseline: FrozenStudentV0,
    observation: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
) -> tuple[list[str], torch.Tensor]:
    projected = project_policy_observation(observation)
    state_features: dict[str, float] = {}
    _flatten(projected, "state", state_features)

    ids: list[str] = []
    scores: list[float] = []
    seen: set[str] = set()
    for action in legal_actions:
        payload = normalize_action_payload(action)
        action_id = sha256_json(payload)
        if action_id in seen:
            raise StudentV1Error("duplicate legal action identity")
        seen.add(action_id)
        ids.append(action_id)

        action_features: dict[str, float] = {f"action.id={action_id}": 1.0}
        _flatten(payload, "action", action_features)
        row = dict(action_features)
        for state_key, state_value in state_features.items():
            for action_key, action_value in action_features.items():
                product = state_value * action_value
                if product != 0.0:
                    row[f"cross::{state_key}::{action_key}"] = product
        scores.append(
            sum(float(baseline.weights.get(key, 0.0)) * value for key, value in row.items())
        )
    return ids, torch.tensor(scores, dtype=torch.float32)


class _ResidualActorCritic(nn.Module):
    def __init__(self, config: StudentV1Config) -> None:
        super().__init__()
        pair_dim = config.state_dim + config.action_dim
        self.actor = nn.Sequential(
            nn.Linear(pair_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        # Exact v0 deterministic behavior at initialization.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def residual_logits(
        self,
        state_vector: torch.Tensor,
        action_vectors: torch.Tensor,
    ) -> torch.Tensor:
        repeated = state_vector.unsqueeze(0).expand(action_vectors.shape[0], -1)
        pair = torch.cat((repeated, action_vectors), dim=1)
        return self.actor(pair).squeeze(-1)

    def value(self, state_vector: torch.Tensor) -> torch.Tensor:
        return self.critic(state_vector).squeeze(-1)


class StudentV1PPO:
    def __init__(
        self,
        baseline: FrozenStudentV0,
        *,
        config: StudentV1Config | None = None,
        generation: int = 0,
        device: str = "cpu",
    ) -> None:
        self.baseline = baseline
        self.config = config or StudentV1Config()
        self.generation = int(generation)
        if self.generation < 0:
            raise StudentV1Error("generation must be non-negative")
        self.device = torch.device(device)
        self.model = _ResidualActorCritic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.total_updates = 0

    def _policy_tensors(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not legal_actions:
            raise StudentV1Error("legal_actions must be non-empty")
        action_ids, base_logits = _baseline_scores(
            self.baseline, observation, legal_actions
        )
        state = encode_state(observation, self.config.state_dim).to(self.device)
        actions = torch.stack(
            [encode_action(action, self.config.action_dim) for action in legal_actions]
        ).to(self.device)
        base_logits = base_logits.to(self.device)
        residual = self.model.residual_logits(state, actions)
        logits = base_logits + residual
        value = self.model.value(state)
        return action_ids, base_logits, logits, value, state

    def sample_action(
        self,
        public_state: Mapping[str, Any],
        *,
        deterministic: bool = False,
        require_command: bool = True,
    ) -> StudentV1Decision:
        raw_actions = public_state.get("legal_actions")
        if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes, bytearray)):
            raise StudentV1Error("Student v1 requires legal_actions")
        actions = [row for row in raw_actions if isinstance(row, Mapping)]
        if len(actions) != len(raw_actions) or not actions:
            raise StudentV1Error("Student v1 legal actions must be non-empty objects")

        commands: list[str | None] = []
        canonical_actions: list[Mapping[str, Any]] = []
        for raw in actions:
            payload = normalize_action_payload(raw)
            canonical_actions.append(payload)
            command = raw.get("command")
            command = command.strip() if isinstance(command, str) and command.strip() else None
            if require_command and command is None:
                raise StudentV1Error("real-game legal action is missing command")
            commands.append(command)

        try:
            observation = project_policy_observation(public_state)
        except FrozenStudentError as exc:
            raise StudentV1Error(str(exc)) from exc

        with torch.no_grad():
            action_ids, _, logits, value, _ = self._policy_tensors(
                observation, canonical_actions
            )
            dist = Categorical(logits=logits)
            if deterministic:
                index_t = torch.argmax(logits)
            else:
                index_t = dist.sample()
            log_prob = dist.log_prob(index_t)

        index = int(index_t.item())
        return StudentV1Decision(
            action_index=index,
            action_id=action_ids[index],
            command=commands[index],
            score=float(logits[index].item()),
            legal_action_count=len(actions),
            log_prob=float(log_prob.item()),
            value=float(value.item()),
        )

    def select_action(
        self,
        public_state: Mapping[str, Any],
        *,
        require_command: bool = True,
    ) -> StudentDecision:
        return self.sample_action(
            public_state,
            deterministic=True,
            require_command=require_command,
        ).as_student_decision()

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "student_schema_version": STUDENT_V1_SCHEMA_VERSION,
            "baseline_model_sha256": EXPECTED_MODEL_SHA256,
            "baseline_artifact_sha256": self.baseline.artifact_sha256,
            "foundation_policy_id": STS1_BASE_POLICY_ID,
            "config": asdict(self.config),
            "config_hash": self.config.config_hash,
            "generation": self.generation,
            "total_updates": self.total_updates,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        torch.save(self.checkpoint_payload(), temp)
        temp.replace(path)

    @classmethod
    def load(
        cls,
        path: Path,
        baseline: FrozenStudentV0,
        *,
        device: str = "cpu",
    ) -> "StudentV1PPO":
        payload = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise StudentV1Error("Student v1 checkpoint must be a mapping")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise StudentV1Error("Student v1 checkpoint schema mismatch")
        if payload.get("student_schema_version") != STUDENT_V1_SCHEMA_VERSION:
            raise StudentV1Error("Student v1 policy schema mismatch")
        if payload.get("baseline_model_sha256") != EXPECTED_MODEL_SHA256:
            raise StudentV1Error("Student v1 baseline model identity drift")
        if payload.get("baseline_artifact_sha256") != baseline.artifact_sha256:
            raise StudentV1Error("Student v1 baseline artifact identity drift")
        if payload.get("foundation_policy_id") != STS1_BASE_POLICY_ID:
            raise StudentV1Error("Student v1 foundation/base policy identity drift")
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise StudentV1Error("Student v1 checkpoint config missing")
        config = StudentV1Config(**dict(raw_config))
        if payload.get("config_hash") != config.config_hash:
            raise StudentV1Error("Student v1 checkpoint config hash mismatch")
        policy = cls(
            baseline,
            config=config,
            generation=int(payload.get("generation", 0)),
            device=device,
        )
        policy.model.load_state_dict(payload["model_state"])
        try:
            policy.optimizer.load_state_dict(payload["optimizer_state"])
        except (KeyError, ValueError, RuntimeError):
            pass
        policy.total_updates = int(payload.get("total_updates", 0))
        return policy


def teacher_bc_update(
    policy: StudentV1PPO,
    examples: Sequence[Any],
    *,
    epochs: int = 12,
) -> dict[str, float | int]:
    """Warm-start Student v1 from frozen public-state Teacher labels.

    Only train examples are accepted by the caller. Tie labels are treated as
    equally valid targets, so the residual actor is not forced to invent a
    preference inside a frozen Teacher tie set.
    """

    if epochs < 1:
        raise StudentV1Error("teacher BC epochs must be positive")
    if not examples:
        raise StudentV1Error("teacher BC requires at least one example")

    def evaluate() -> tuple[float, float]:
        exact = 0
        tie_aware = 0
        with torch.no_grad():
            for example in examples:
                observation = getattr(example, "observation", None)
                actions = getattr(example, "action_payloads", None)
                selected_index = int(getattr(example, "selected_index", -1))
                tie_indices = tuple(int(x) for x in getattr(example, "tie_indices", ()))
                if not isinstance(observation, Mapping) or not isinstance(actions, Sequence) or not actions:
                    raise StudentV1Error("invalid Teacher BC example")
                if not 0 <= selected_index < len(actions):
                    raise StudentV1Error("Teacher selected_index is outside legal actions")
                if not tie_indices or any(index < 0 or index >= len(actions) for index in tie_indices):
                    raise StudentV1Error("Teacher tie_indices are invalid")
                _, _, logits, _, _ = policy._policy_tensors(observation, actions)
                pred = int(torch.argmax(logits).item())
                exact += int(pred == selected_index)
                tie_aware += int(pred in tie_indices)
        total = len(examples)
        return exact / total, tie_aware / total

    before_exact, before_tie = evaluate()
    losses: list[float] = []
    updates = 0

    for _ in range(epochs):
        order = torch.randperm(len(examples)).tolist()
        for index in order:
            example = examples[index]
            observation = getattr(example, "observation", None)
            actions = getattr(example, "action_payloads", None)
            tie_indices = tuple(int(x) for x in getattr(example, "tie_indices", ()))
            if not isinstance(observation, Mapping) or not isinstance(actions, Sequence) or not actions:
                raise StudentV1Error("invalid Teacher BC example")
            if not tie_indices or any(i < 0 or i >= len(actions) for i in tie_indices):
                raise StudentV1Error("Teacher tie_indices are invalid")

            _, _, logits, _, _ = policy._policy_tensors(observation, actions)
            log_probs = torch.log_softmax(logits, dim=0)
            target = torch.tensor(tie_indices, dtype=torch.long, device=policy.device)
            loss = -log_probs[target].mean()

            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.model.actor.parameters(), policy.config.max_grad_norm
            )
            policy.optimizer.step()
            losses.append(float(loss.item()))
            updates += 1

    after_exact, after_tie = evaluate()
    return {
        "examples": len(examples),
        "epochs": epochs,
        "updates": updates,
        "mean_loss": sum(losses) / len(losses),
        "before_top1_accuracy": before_exact,
        "after_top1_accuracy": after_exact,
        "before_tie_aware_accuracy": before_tie,
        "after_tie_aware_accuracy": after_tie,
    }


def _transition_policy(
    policy: StudentV1PPO,
    transition: PPOTransition,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, base_logits, logits, value, _ = policy._policy_tensors(
        transition.observation,
        transition.legal_actions,
    )
    selected = torch.tensor(
        transition.selected_action_index, dtype=torch.long, device=policy.device
    )
    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(selected)
    entropy = dist.entropy()

    # Anchor the residual policy to the frozen v0 distribution. This is the
    # v0-preserving equivalent of AscensionAI's BC anchor.
    with torch.no_grad():
        baseline_probs = torch.softmax(base_logits, dim=0)
    current_log_probs = torch.log_softmax(logits, dim=0)
    anchor_kl = torch.sum(
        baseline_probs * (torch.log(baseline_probs.clamp_min(1e-12)) - current_log_probs)
    )
    return log_prob, value, entropy, anchor_kl


def _gae_for_episodes(
    episodes: Sequence[PPOEpisode],
    config: StudentV1Config,
) -> tuple[list[PPOTransition], torch.Tensor, torch.Tensor]:
    transitions: list[PPOTransition] = []
    advantages: list[float] = []
    returns: list[float] = []

    for episode in episodes:
        ep = list(episode.transitions)
        ep_adv = [0.0] * len(ep)
        last_gae = 0.0
        for index in reversed(range(len(ep))):
            row = ep[index]
            if index == len(ep) - 1:
                next_value = 0.0
            else:
                next_value = ep[index + 1].old_value
            nonterminal = 0.0 if row.done else 1.0
            delta = (
                row.reward
                + config.gamma * next_value * nonterminal
                - row.old_value
            )
            last_gae = (
                delta
                + config.gamma * config.gae_lambda * nonterminal * last_gae
            )
            ep_adv[index] = last_gae
        for row, advantage in zip(ep, ep_adv):
            transitions.append(row)
            advantages.append(advantage)
            returns.append(advantage + row.old_value)

    adv = torch.tensor(advantages, dtype=torch.float32)
    ret = torch.tensor(returns, dtype=torch.float32)
    if len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return transitions, adv, ret


def ppo_update(
    policy: StudentV1PPO,
    episodes: Sequence[PPOEpisode],
) -> dict[str, float | int]:
    """Run one bounded PPO update over complete, public-state episodes."""

    if not episodes:
        raise StudentV1Error("PPO update requires at least one episode")
    transitions, advantages, returns = _gae_for_episodes(
        episodes, policy.config
    )
    if len(transitions) < 2:
        raise StudentV1Error("PPO update requires at least two transitions")

    old_log_probs = torch.tensor(
        [row.old_log_prob for row in transitions], dtype=torch.float32
    )
    advantages = advantages.to(policy.device)
    returns = returns.to(policy.device)
    old_log_probs = old_log_probs.to(policy.device)

    count = len(transitions)
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "anchor_kl": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    batches = 0
    early_stop = False

    for _epoch in range(policy.config.epochs):
        permutation = torch.randperm(count).tolist()
        epoch_kls: list[float] = []
        for start in range(0, count, policy.config.batch_size):
            indices = permutation[start:start + policy.config.batch_size]
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            anchors: list[torch.Tensor] = []
            for index in indices:
                lp, value, entropy, anchor = _transition_policy(
                    policy, transitions[index]
                )
                log_probs.append(lp)
                values.append(value)
                entropies.append(entropy)
                anchors.append(anchor)

            new_lp = torch.stack(log_probs)
            value_t = torch.stack(values)
            entropy_t = torch.stack(entropies).mean()
            anchor_t = torch.stack(anchors).mean()
            idx_t = torch.tensor(indices, dtype=torch.long, device=policy.device)
            old_lp = old_log_probs[idx_t]
            adv = advantages[idx_t]
            ret = returns[idx_t]

            ratio = torch.exp(new_lp - old_lp)
            unclipped = -adv * ratio
            clipped = -adv * torch.clamp(
                ratio,
                1.0 - policy.config.clip_range,
                1.0 + policy.config.clip_range,
            )
            policy_loss = torch.max(unclipped, clipped).mean()
            value_loss = torch.mean((value_t - ret) ** 2)
            loss = (
                policy_loss
                + policy.config.value_coef * value_loss
                - policy.config.entropy_coef * entropy_t
                + policy.config.baseline_anchor_coef * anchor_t
            )

            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.model.parameters(), policy.config.max_grad_norm
            )
            policy.optimizer.step()

            with torch.no_grad():
                approx_kl = torch.mean(old_lp - new_lp).item()
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > policy.config.clip_range).float()
                ).item()
            epoch_kls.append(approx_kl)
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy_t.item())
            totals["anchor_kl"] += float(anchor_t.item())
            totals["approx_kl"] += float(approx_kl)
            totals["clip_fraction"] += float(clip_fraction)
            batches += 1

        if (
            policy.config.target_kl > 0
            and epoch_kls
            and sum(epoch_kls) / len(epoch_kls) > policy.config.target_kl
        ):
            early_stop = True
            break

    if batches == 0:
        raise StudentV1Error("PPO update produced no batches")
    policy.total_updates += 1
    return {
        **{key: value / batches for key, value in totals.items()},
        "batches": batches,
        "transitions": count,
        "total_updates": policy.total_updates,
        "early_stop": int(early_stop),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "STUDENT_V1_SCHEMA_VERSION",
    "StudentV1Config",
    "StudentV1Decision",
    "StudentV1Error",
    "StudentV1PPO",
    "encode_action",
    "encode_state",
    "file_sha256",
    "ppo_update",
    "teacher_bc_update",
]
