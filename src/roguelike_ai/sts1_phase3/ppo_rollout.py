"""Auditable public-state PPO rollout format for STS1 Student v1+."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .frozen_student import (
    FrozenStudentError,
    normalize_action_payload,
    project_policy_observation,
    sha256_json,
)
from .self_improve_loop import (
    RolloutIdentity,
    StaleRolloutError,
    accept_current_rollouts,
    build_rollout_manifest,
)


TRANSITION_SCHEMA_VERSION = "sts1-ppo-transition-v1"
EPISODE_SCHEMA_VERSION = "sts1-ppo-episode-v1"


class PPORolloutError(ValueError):
    """One PPO rollout violates the public-state or policy-evidence contract."""


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PPORolloutError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PPORolloutError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class PPOTransition:
    episode_id: str
    step_index: int
    observation: Mapping[str, Any]
    legal_actions: tuple[Mapping[str, Any], ...]
    legal_action_ids: tuple[str, ...]
    selected_action_index: int
    selected_action_id: str
    reward: float
    done: bool
    old_log_prob: float
    old_value: float
    schema_version: str = TRANSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRANSITION_SCHEMA_VERSION:
            raise PPORolloutError(f"unsupported transition schema: {self.schema_version}")
        if not self.episode_id.strip():
            raise PPORolloutError("episode_id must be non-empty")
        if self.step_index < 0:
            raise PPORolloutError("step_index must be non-negative")
        if not self.legal_actions:
            raise PPORolloutError("legal_actions must be non-empty")
        if len(self.legal_actions) != len(self.legal_action_ids):
            raise PPORolloutError("legal action/id length mismatch")
        if len(set(self.legal_action_ids)) != len(self.legal_action_ids):
            raise PPORolloutError("duplicate legal action identity")
        if not 0 <= self.selected_action_index < len(self.legal_actions):
            raise PPORolloutError("selected_action_index is outside legal actions")
        if self.selected_action_id != self.legal_action_ids[self.selected_action_index]:
            raise PPORolloutError("selected_action_id/index mismatch")
        _finite(self.reward, "reward")
        _finite(self.old_log_prob, "old_log_prob")
        _finite(self.old_value, "old_value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "observation": self.observation,
            "legal_actions": list(self.legal_actions),
            "legal_action_ids": list(self.legal_action_ids),
            "selected_action_index": self.selected_action_index,
            "selected_action_id": self.selected_action_id,
            "reward": self.reward,
            "done": self.done,
            "old_log_prob": self.old_log_prob,
            "old_value": self.old_value,
        }


def make_transition(
    public_state: Mapping[str, Any],
    *,
    episode_id: str,
    step_index: int,
    selected_action_index: int,
    reward: float,
    done: bool,
    old_log_prob: float,
    old_value: float,
) -> PPOTransition:
    """Create one PPO transition after stripping transport and hidden data."""

    try:
        observation = project_policy_observation(public_state)
    except FrozenStudentError as exc:
        raise PPORolloutError(str(exc)) from exc

    raw_actions = public_state.get("legal_actions")
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes, bytearray)):
        raise PPORolloutError("public_state legal_actions must be an array")
    actions: list[Mapping[str, Any]] = []
    action_ids: list[str] = []
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise PPORolloutError("legal action must be an object")
        try:
            payload = normalize_action_payload(raw)
        except FrozenStudentError as exc:
            raise PPORolloutError(str(exc)) from exc
        action_id = sha256_json(payload)
        actions.append(payload)
        action_ids.append(action_id)

    if not actions:
        raise PPORolloutError("public_state legal_actions must be non-empty")
    if not 0 <= selected_action_index < len(actions):
        raise PPORolloutError("selected action is outside legal actions")

    return PPOTransition(
        episode_id=episode_id,
        step_index=step_index,
        observation=observation,
        legal_actions=tuple(actions),
        legal_action_ids=tuple(action_ids),
        selected_action_index=selected_action_index,
        selected_action_id=action_ids[selected_action_index],
        reward=_finite(reward, "reward"),
        done=bool(done),
        old_log_prob=_finite(old_log_prob, "old_log_prob"),
        old_value=_finite(old_value, "old_value"),
    )


@dataclass(frozen=True)
class PPOEpisode:
    episode_id: str
    transitions: tuple[PPOTransition, ...]
    outcome: str
    final_floor: int | None

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise PPORolloutError("episode_id must be non-empty")
        if not self.transitions:
            raise PPORolloutError("episode must contain transitions")
        if self.outcome not in {"victory", "defeat"}:
            raise PPORolloutError("episode outcome must be victory or defeat")
        for index, transition in enumerate(self.transitions):
            if transition.episode_id != self.episode_id:
                raise PPORolloutError("transition episode_id mismatch")
            if transition.step_index != index:
                raise PPORolloutError("transition step_index must be contiguous")
            if transition.done != (index == len(self.transitions) - 1):
                raise PPORolloutError("only the final transition may be done=True")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "outcome": self.outcome,
            "final_floor": self.final_floor,
            "transitions": [row.to_dict() for row in self.transitions],
        }


def _metric(state: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = state.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(default)
    result = float(value)
    return result if math.isfinite(result) else float(default)


def _enemy_stats(state: Mapping[str, Any]) -> tuple[float, int]:
    enemies = state.get("enemies")
    if not isinstance(enemies, Sequence) or isinstance(enemies, (str, bytes, bytearray)):
        return 0.0, 0
    total_hp = 0.0
    alive = 0
    for enemy in enemies:
        if not isinstance(enemy, Mapping):
            continue
        hp = _metric(enemy, "hp", 0.0)
        if hp > 0.0:
            total_hp += hp
            alive += 1
    return total_hp, alive


def public_progress_reward(
    before: Mapping[str, Any],
    after_hp: float,
    after_floor: float,
    *,
    after_state: Mapping[str, Any] | None = None,
    terminal_outcome: str | None = None,
) -> float:
    """Dense public-only combat shaping plus terminal win/loss reward.

    No seed/RNG/future information is used. Same-combat enemy HP loss gives
    immediate credit, while player HP loss is penalized and floor progress is
    rewarded. Cross-combat transitions rely on floor/HP rather than pretending
    the next room's enemies are the previous enemies.
    """

    hp_delta = after_hp - _metric(before, "hp")
    floor_delta = max(0.0, after_floor - _metric(before, "floor"))
    enemy_damage = 0.0
    enemy_kills = 0
    if after_state is not None and floor_delta == 0.0:
        before_enemy_hp, before_alive = _enemy_stats(before)
        after_enemy_hp, after_alive = _enemy_stats(after_state)
        enemy_damage = max(0.0, before_enemy_hp - after_enemy_hp)
        enemy_kills = max(0, before_alive - after_alive)

    reward = (
        0.006 * enemy_damage
        + 0.04 * enemy_kills
        + 0.004 * hp_delta
        + 0.03 * floor_delta
    )
    if terminal_outcome == "victory":
        reward += 1.5
    elif terminal_outcome == "defeat":
        reward -= 1.0
    return max(-2.0, min(2.0, reward))


def episode_from_simulator_evidence(
    evidence_path: Path,
    *,
    episode_id: str,
) -> PPOEpisode:
    """Turn one completed --collect-ppo simulator evidence file into an episode."""

    decisions: list[Mapping[str, Any]] = []
    summary: Mapping[str, Any] | None = None
    try:
        lines = evidence_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PPORolloutError(f"could not read simulator evidence: {exc}") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PPORolloutError(f"invalid simulator evidence JSONL: {exc}") from exc
        if not isinstance(row, Mapping):
            raise PPORolloutError("simulator evidence row must be an object")
        if row.get("type") == "ppo_decision":
            decisions.append(row)
        elif row.get("type") == "summary":
            summary = row

    if summary is None:
        raise PPORolloutError("simulator evidence is missing summary")
    if summary.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
        raise PPORolloutError("incomplete/blocked simulator run cannot become PPO training data")
    if summary.get("ppo_collection") is not True:
        raise PPORolloutError("simulator evidence was not collected in PPO mode")
    outcome = str(summary.get("outcome", ""))
    if outcome not in {"victory", "defeat"}:
        raise PPORolloutError("PPO episode requires terminal victory/defeat")
    if not decisions:
        raise PPORolloutError("simulator evidence contains no PPO decisions")

    transitions: list[PPOTransition] = []
    for index, row in enumerate(decisions):
        state = row.get("public_state")
        if not isinstance(state, Mapping):
            raise PPORolloutError("PPO decision is missing public_state")

        final = index == len(decisions) - 1
        if final:
            after_hp = _finite(summary.get("final_hp", 0.0), "summary.final_hp")
            after_floor = _finite(summary.get("final_floor", 0.0), "summary.final_floor")
            terminal = outcome
        else:
            next_state = decisions[index + 1].get("public_state")
            if not isinstance(next_state, Mapping):
                raise PPORolloutError("next PPO decision is missing public_state")
            after_hp = _metric(next_state, "hp", _metric(state, "hp"))
            after_floor = _metric(next_state, "floor", _metric(state, "floor"))
            terminal = None

        reward = public_progress_reward(
            state,
            after_hp,
            after_floor,
            after_state=(next_state if not final else None),
            terminal_outcome=terminal,
        )
        transitions.append(
            make_transition(
                state,
                episode_id=episode_id,
                step_index=index,
                selected_action_index=int(row.get("action_index", -1)),
                reward=reward,
                done=final,
                old_log_prob=_finite(row.get("old_log_prob"), "old_log_prob"),
                old_value=_finite(row.get("old_value"), "old_value"),
            )
        )

    final_floor_raw = summary.get("final_floor")
    final_floor = (
        int(final_floor_raw)
        if isinstance(final_floor_raw, (int, float)) and not isinstance(final_floor_raw, bool)
        else None
    )
    return PPOEpisode(
        episode_id=episode_id,
        transitions=tuple(transitions),
        outcome=outcome,
        final_floor=final_floor,
    )


def _payload_bytes(episodes: Sequence[PPOEpisode]) -> bytes:
    rows = [episode.to_dict() for episode in episodes]
    encoded = "\n".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for row in rows
    ) + "\n"
    return encoded.encode("utf-8")


def write_rollout_shard(
    payload_path: Path,
    manifest_path: Path,
    *,
    identity: RolloutIdentity,
    shard_id: str,
    episodes: Sequence[PPOEpisode],
) -> dict[str, Any]:
    if not episodes:
        raise PPORolloutError("rollout shard must contain episodes")
    episode_ids = [episode.episode_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise PPORolloutError("duplicate episode_id in rollout shard")
    payload = _payload_bytes(episodes)
    payload_sha = hashlib.sha256(payload).hexdigest()
    manifest = build_rollout_manifest(
        identity,
        shard_id=shard_id,
        episode_count=len(episodes),
        decision_count=sum(len(episode.transitions) for episode in episodes),
        payload_sha256=payload_sha,
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(payload)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _transition_from_dict(value: Mapping[str, Any]) -> PPOTransition:
    try:
        observation = value["observation"]
        legal_actions = value["legal_actions"]
        legal_action_ids = value["legal_action_ids"]
        if not isinstance(observation, Mapping):
            raise PPORolloutError("transition observation must be an object")
        # Re-run the hidden-information validator even for serialized evidence.
        projected = project_policy_observation(observation)
        if dict(projected) != dict(observation):
            raise PPORolloutError("serialized observation is not canonical public policy state")
        if not isinstance(legal_actions, list) or not all(isinstance(row, Mapping) for row in legal_actions):
            raise PPORolloutError("serialized legal_actions must be objects")
        canonical_actions = tuple(normalize_action_payload(row) for row in legal_actions)
        canonical_ids = tuple(sha256_json(row) for row in canonical_actions)
        if tuple(str(x) for x in legal_action_ids) != canonical_ids:
            raise PPORolloutError("serialized legal_action_ids drifted from action payloads")
        return PPOTransition(
            episode_id=str(value["episode_id"]),
            step_index=int(value["step_index"]),
            observation=projected,
            legal_actions=canonical_actions,
            legal_action_ids=canonical_ids,
            selected_action_index=int(value["selected_action_index"]),
            selected_action_id=str(value["selected_action_id"]),
            reward=_finite(value["reward"], "reward"),
            done=bool(value["done"]),
            old_log_prob=_finite(value["old_log_prob"], "old_log_prob"),
            old_value=_finite(value["old_value"], "old_value"),
            schema_version=str(value.get("schema_version", "")),
        )
    except (KeyError, TypeError, ValueError, FrozenStudentError) as exc:
        if isinstance(exc, PPORolloutError):
            raise
        raise PPORolloutError(f"invalid transition: {exc}") from exc


def read_rollout_shard(
    payload_path: Path,
    manifest_path: Path,
    *,
    expected_identity: RolloutIdentity,
) -> tuple[PPOEpisode, ...]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PPORolloutError(f"invalid rollout manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise PPORolloutError("rollout manifest must be an object")
    try:
        accept_current_rollouts([manifest], expected=expected_identity)
    except StaleRolloutError as exc:
        raise PPORolloutError(str(exc)) from exc

    payload = payload_path.read_bytes()
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != manifest.get("payload_sha256"):
        raise PPORolloutError("rollout payload checksum mismatch")

    episodes: list[PPOEpisode] = []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PPORolloutError(f"invalid rollout JSONL: {exc}") from exc
        if not isinstance(raw, Mapping) or raw.get("schema_version") != EPISODE_SCHEMA_VERSION:
            raise PPORolloutError("invalid episode schema")
        rows = raw.get("transitions")
        if not isinstance(rows, list):
            raise PPORolloutError("episode transitions must be an array")
        transitions = tuple(_transition_from_dict(row) for row in rows if isinstance(row, Mapping))
        if len(transitions) != len(rows):
            raise PPORolloutError("episode contains a non-object transition")
        episode = PPOEpisode(
            episode_id=str(raw.get("episode_id", "")),
            transitions=transitions,
            outcome=str(raw.get("outcome", "")),
            final_floor=(
                int(raw["final_floor"]) if raw.get("final_floor") is not None else None
            ),
        )
        episodes.append(episode)

    if len(episodes) != int(manifest.get("episode_count", -1)):
        raise PPORolloutError("episode count does not match rollout manifest")
    if sum(len(ep.transitions) for ep in episodes) != int(manifest.get("decision_count", -1)):
        raise PPORolloutError("decision count does not match rollout manifest")
    return tuple(episodes)


__all__ = [
    "EPISODE_SCHEMA_VERSION",
    "PPORolloutError",
    "PPOEpisode",
    "PPOTransition",
    "TRANSITION_SCHEMA_VERSION",
    "episode_from_simulator_evidence",
    "make_transition",
    "public_progress_reward",
    "read_rollout_shard",
    "write_rollout_shard",
]
