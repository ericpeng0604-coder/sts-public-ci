"""Shared frozen STS1 Phase-3 protocol identity.

Kept independent from the real-game bridge so offline/public compute workers can
use the exact same seed/model/schema contract without importing CommunicationMod.
"""

from __future__ import annotations

from typing import Any

from .frozen_student import (
    ACTION_SCHEMA_VERSION,
    EXPECTED_CONFIG_HASH,
    EXPECTED_DATASET_HASH,
    EXPECTED_MODEL_SHA256,
    EXPECTED_SOURCE_SHA256,
    PREDECESSOR_HEAD,
    PUBLIC_STATE_SCHEMA_VERSION,
    sha256_json,
)


A0_PROTOCOL_VERSION = "sts1-phase3-a0-full-run-v1"
REAL_GAME_SCHEMA_VERSION = "sts1-real-game-public-state-v1"
FROZEN_SIMULATOR_SHA = "7476a81954020087da31d41d16fddf475746ec2d"
A0_FROZEN_SEEDS_V1 = (
    "347001", "347002", "347003", "347004", "347005",
    "347006", "347007", "347008", "347009", "347010",
)
DEFAULT_MAX_DECISIONS = 2500


def frozen_a0_manifest() -> dict[str, Any]:
    seed_payload = {
        "schema_version": "sts1-phase3-a0-seeds-v1",
        "seeds": list(A0_FROZEN_SEEDS_V1),
    }
    return {
        "schema_version": A0_PROTOCOL_VERSION,
        "phase": "A0",
        "frozen_before_outcomes": True,
        "student": {
            "predecessor_head": PREDECESSOR_HEAD,
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "config_hash": EXPECTED_CONFIG_HASH,
            "model_sha256": EXPECTED_MODEL_SHA256,
            "dataset_hash": EXPECTED_DATASET_HASH,
        },
        "simulator_sha": FROZEN_SIMULATOR_SHA,
        "public_state_schema": PUBLIC_STATE_SCHEMA_VERSION,
        "action_schema": ACTION_SCHEMA_VERSION,
        "real_game_adapter_schema": REAL_GAME_SCHEMA_VERSION,
        "seed_manifest": seed_payload,
        "seed_manifest_sha256": sha256_json(seed_payload),
        "max_decisions_per_run": DEFAULT_MAX_DECISIONS,
        "combat_policy": "exact-frozen-student-v0",
        "noncombat_policy": "deterministic-public-legal-action-fallback-v1",
        "noncombat_priority": ["choose", "proceed", "cancel"],
        "required_metrics": [
            "outcome", "final_floor", "max_act", "illegal_action_count", "fallback_count",
            "student_action_count", "mean_inference_latency_ms", "max_inference_latency_ms",
            "timeout_count", "crash_count", "remote_error_count",
        ],
        "promotion_verdict": "NOT_VERIFIED",
    }


__all__ = [
    "A0_FROZEN_SEEDS_V1",
    "A0_PROTOCOL_VERSION",
    "DEFAULT_MAX_DECISIONS",
    "FROZEN_SIMULATOR_SHA",
    "REAL_GAME_SCHEMA_VERSION",
    "frozen_a0_manifest",
]
