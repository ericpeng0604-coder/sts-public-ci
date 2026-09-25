"""Resumable state for the STS1 self-improvement training loop.

This is the AscensionAI-style safety layer around rollouts/checkpoints:
- every rollout is bound to an exact Champion generation and simulator identity,
- stale/mixed rollouts fail closed,
- a Candidate cannot replace the Champion until the promotion gates say PROMOTE,
- checkpoint writes use atomic replacement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ROLLOUT_SCHEMA_VERSION = "sts1-self-improve-rollout-v1"
CHECKPOINT_SCHEMA_VERSION = "sts1-self-improve-checkpoint-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SelfImproveLoopError(RuntimeError):
    """The self-improvement loop cannot safely continue."""


class StaleRolloutError(SelfImproveLoopError):
    """A rollout was produced by the wrong model/simulator generation."""


def _require_sha256(value: str, field: str) -> str:
    normalized = value.lower().strip()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase/uppercase SHA-256 hex digest")
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RolloutIdentity:
    generation: int
    policy_sha256: str
    simulator_sha: str
    public_state_schema: str
    action_schema: str
    rollout_schema: str = ROLLOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        object.__setattr__(self, "policy_sha256", _require_sha256(self.policy_sha256, "policy_sha256"))
        if not self.simulator_sha.strip():
            raise ValueError("simulator_sha must be non-empty")
        if not self.public_state_schema.strip() or not self.action_schema.strip():
            raise ValueError("state/action schema must be non-empty")
        if self.rollout_schema != ROLLOUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported rollout schema: {self.rollout_schema}")

    @property
    def identity_hash(self) -> str:
        return _sha256_json(asdict(self))


def build_rollout_manifest(
    identity: RolloutIdentity,
    *,
    shard_id: str,
    episode_count: int,
    decision_count: int,
    payload_sha256: str,
) -> dict[str, Any]:
    if not shard_id.strip():
        raise ValueError("shard_id must be non-empty")
    if episode_count < 1 or decision_count < 1:
        raise ValueError("rollout shard must contain episodes and decisions")
    payload_sha256 = _require_sha256(payload_sha256, "payload_sha256")
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "identity": asdict(identity),
        "identity_hash": identity.identity_hash,
        "shard_id": shard_id,
        "episode_count": episode_count,
        "decision_count": decision_count,
        "payload_sha256": payload_sha256,
    }


def _manifest_identity(manifest: Mapping[str, Any]) -> RolloutIdentity:
    if manifest.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
        raise StaleRolloutError("rollout schema mismatch")
    payload = manifest.get("identity")
    if not isinstance(payload, Mapping):
        raise StaleRolloutError("rollout identity missing")
    try:
        identity = RolloutIdentity(
            generation=int(payload["generation"]),
            policy_sha256=str(payload["policy_sha256"]),
            simulator_sha=str(payload["simulator_sha"]),
            public_state_schema=str(payload["public_state_schema"]),
            action_schema=str(payload["action_schema"]),
            rollout_schema=str(payload.get("rollout_schema", ROLLOUT_SCHEMA_VERSION)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StaleRolloutError(f"invalid rollout identity: {exc}") from exc
    if manifest.get("identity_hash") != identity.identity_hash:
        raise StaleRolloutError("rollout identity hash mismatch")
    return identity


def accept_current_rollouts(
    manifests: Sequence[Mapping[str, Any]],
    *,
    expected: RolloutIdentity,
) -> tuple[Mapping[str, Any], ...]:
    """Return manifests only when every shard belongs to the exact current Champion."""

    if not manifests:
        raise StaleRolloutError("no rollout manifests supplied")
    seen_shards: set[str] = set()
    accepted: list[Mapping[str, Any]] = []
    for manifest in manifests:
        identity = _manifest_identity(manifest)
        if identity != expected:
            raise StaleRolloutError(
                "stale rollout rejected: "
                f"expected={expected.identity_hash} got={identity.identity_hash}"
            )
        shard_id = manifest.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id.strip():
            raise StaleRolloutError("rollout shard_id missing")
        if shard_id in seen_shards:
            raise StaleRolloutError(f"duplicate rollout shard: {shard_id}")
        seen_shards.add(shard_id)
        accepted.append(manifest)
    return tuple(accepted)


@dataclass(frozen=True)
class LoopCheckpoint:
    generation: int
    champion_sha256: str
    phase: str = "collect"
    candidate_sha256: str | None = None
    rollout_identity_hash: str | None = None
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema: {self.schema_version}")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        object.__setattr__(
            self, "champion_sha256", _require_sha256(self.champion_sha256, "champion_sha256")
        )
        if self.candidate_sha256 is not None:
            object.__setattr__(
                self, "candidate_sha256", _require_sha256(self.candidate_sha256, "candidate_sha256")
            )
        if not self.phase.strip():
            raise ValueError("phase must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_checkpoint_atomic(path: Path, checkpoint: LoopCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    encoded = json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n"
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_checkpoint(path: Path) -> LoopCheckpoint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SelfImproveLoopError("checkpoint must be a JSON object")
    try:
        return LoopCheckpoint(
            generation=int(payload["generation"]),
            champion_sha256=str(payload["champion_sha256"]),
            phase=str(payload.get("phase", "collect")),
            candidate_sha256=(
                str(payload["candidate_sha256"])
                if payload.get("candidate_sha256") is not None
                else None
            ),
            rollout_identity_hash=(
                str(payload["rollout_identity_hash"])
                if payload.get("rollout_identity_hash") is not None
                else None
            ),
            schema_version=str(payload.get("schema_version", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SelfImproveLoopError(f"invalid checkpoint: {exc}") from exc


def attach_candidate(checkpoint: LoopCheckpoint, candidate_sha256: str) -> LoopCheckpoint:
    candidate_sha256 = _require_sha256(candidate_sha256, "candidate_sha256")
    if candidate_sha256 == checkpoint.champion_sha256:
        raise SelfImproveLoopError("Candidate must differ from current Champion")
    return replace(
        checkpoint,
        candidate_sha256=candidate_sha256,
        phase="gate_30",
    )


def apply_promotion_decision(
    checkpoint: LoopCheckpoint,
    promotion: Mapping[str, Any],
) -> LoopCheckpoint:
    """Advance to the next generation only after an explicit PROMOTE decision."""

    if checkpoint.candidate_sha256 is None:
        raise SelfImproveLoopError("no Candidate is attached")
    if promotion.get("decision") != "PROMOTE" or promotion.get("all_gates_passed") is not True:
        return replace(checkpoint, phase="hold")
    return LoopCheckpoint(
        generation=checkpoint.generation + 1,
        champion_sha256=checkpoint.candidate_sha256,
        phase="collect",
        candidate_sha256=None,
        rollout_identity_hash=None,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "ROLLOUT_SCHEMA_VERSION",
    "LoopCheckpoint",
    "RolloutIdentity",
    "SelfImproveLoopError",
    "StaleRolloutError",
    "accept_current_rollouts",
    "apply_promotion_decision",
    "attach_candidate",
    "build_rollout_manifest",
    "load_checkpoint",
    "write_checkpoint_atomic",
]
