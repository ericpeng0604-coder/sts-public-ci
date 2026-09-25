"""Public STS1 Phase-3 evaluation and self-improvement helpers."""

from .champion_gate import (
    ChampionGateError,
    FAST_GATE_POLICY,
    FORMAL_GATE_POLICY,
    evaluate_fixed_seed_gate,
    evaluate_real_game_gate,
    promotion_decision,
)
from .self_improve_loop import (
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

__all__ = [
    "ChampionGateError",
    "FAST_GATE_POLICY",
    "FORMAL_GATE_POLICY",
    "LoopCheckpoint",
    "RolloutIdentity",
    "SelfImproveLoopError",
    "StaleRolloutError",
    "accept_current_rollouts",
    "apply_promotion_decision",
    "attach_candidate",
    "build_rollout_manifest",
    "evaluate_fixed_seed_gate",
    "evaluate_real_game_gate",
    "load_checkpoint",
    "promotion_decision",
    "write_checkpoint_atomic",
]
