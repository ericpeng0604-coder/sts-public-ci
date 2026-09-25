from __future__ import annotations

import hashlib

import pytest

from roguelike_ai.sts1_phase3.self_improve_loop import (
    LoopCheckpoint,
    RolloutIdentity,
    SelfImproveLoopError,
    StaleRolloutError,
    accept_current_rollouts,
    apply_promotion_decision,
    attach_candidate,
    build_rollout_manifest,
    load_checkpoint,
    write_checkpoint_atomic,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(generation: int = 3, policy: str = "champion") -> RolloutIdentity:
    return RolloutIdentity(
        generation=generation,
        policy_sha256=_sha(policy),
        simulator_sha="7476a81954020087da31d41d16fddf475746ec2d",
        public_state_schema="sts1-public-state-v1",
        action_schema="sts1-action-v1",
    )


def _manifest(identity: RolloutIdentity, shard: str):
    return build_rollout_manifest(
        identity,
        shard_id=shard,
        episode_count=10,
        decision_count=200,
        payload_sha256=_sha(f"payload-{shard}"),
    )


def test_accept_current_rollouts_rejects_stale_generation() -> None:
    expected = _identity(generation=4)
    stale = _identity(generation=3)
    with pytest.raises(StaleRolloutError, match="stale rollout rejected"):
        accept_current_rollouts([_manifest(stale, "worker-01")], expected=expected)


def test_accept_current_rollouts_rejects_wrong_policy_and_duplicate_shards() -> None:
    expected = _identity(policy="champion")
    wrong = _identity(policy="old-champion")
    with pytest.raises(StaleRolloutError, match="stale rollout rejected"):
        accept_current_rollouts([_manifest(wrong, "worker-01")], expected=expected)

    manifest = _manifest(expected, "worker-01")
    with pytest.raises(StaleRolloutError, match="duplicate rollout shard"):
        accept_current_rollouts([manifest, manifest], expected=expected)


def test_current_rollouts_are_accepted_as_one_generation() -> None:
    expected = _identity()
    accepted = accept_current_rollouts(
        [_manifest(expected, "worker-01"), _manifest(expected, "worker-02")],
        expected=expected,
    )
    assert len(accepted) == 2


def test_checkpoint_round_trip_is_atomic_and_resumable(tmp_path) -> None:
    path = tmp_path / "loop-state.json"
    checkpoint = LoopCheckpoint(generation=2, champion_sha256=_sha("champion"))
    write_checkpoint_atomic(path, checkpoint)
    assert load_checkpoint(path) == checkpoint
    assert not path.with_name(path.name + ".tmp").exists()


def test_candidate_does_not_replace_champion_on_hold() -> None:
    checkpoint = LoopCheckpoint(generation=2, champion_sha256=_sha("champion"))
    candidate = attach_candidate(checkpoint, _sha("candidate"))
    held = apply_promotion_decision(
        candidate,
        {"decision": "HOLD", "all_gates_passed": False},
    )
    assert held.generation == 2
    assert held.champion_sha256 == checkpoint.champion_sha256
    assert held.candidate_sha256 == _sha("candidate")
    assert held.phase == "hold"


def test_candidate_promotes_only_after_all_gates_pass() -> None:
    checkpoint = LoopCheckpoint(generation=2, champion_sha256=_sha("champion"))
    candidate = attach_candidate(checkpoint, _sha("candidate"))
    promoted = apply_promotion_decision(
        candidate,
        {"decision": "PROMOTE", "all_gates_passed": True},
    )
    assert promoted.generation == 3
    assert promoted.champion_sha256 == _sha("candidate")
    assert promoted.candidate_sha256 is None
    assert promoted.phase == "collect"


def test_candidate_must_differ_from_champion() -> None:
    checkpoint = LoopCheckpoint(generation=2, champion_sha256=_sha("same"))
    with pytest.raises(SelfImproveLoopError, match="must differ"):
        attach_candidate(checkpoint, _sha("same"))
