#!/usr/bin/env python3
"""Prove PublicStateSearch can use the first native public-only rollout backend."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
BUILD = ROOT / "external" / "sts_lightspeed" / "source" / "build-probe"
SCRIPT_DIR = ROOT / "scripts" / "sts1"
for path in (SRC, BUILD, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import slaythespire as sts

from verify_phase1_public_reconstruction_native import public_state, run_state
from roguelike_ai.sts1_teacher.native_rollout import NativePublicJawWormRolloutV1
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig


def main() -> None:
    state = public_state()
    admitted_state = attach_reconstruction_capabilities(state, run_state=run_state(state))
    context = require_public_reconstruction(admitted_state)

    backend = NativePublicJawWormRolloutV1(admitted_state, sts)
    evaluator = PairedPublicSampleEvaluator(backend)
    config = SearchConfig(
        samples_per_semantic_action=3,
        rollout_budget=32,
        node_budget=4096,
        max_depth=12,
        timeout_ms=10_000,
        tie_tolerance=1e-9,
        sampling_seed=20260829,
    )
    search = PublicStateSearch(evaluator, config)

    first = search.run(context)
    second = search.run(context)

    assert first.resolved is True, first
    assert second.resolved is True, second
    assert first.timed_out is False
    assert second.timed_out is False
    assert first.unresolved_action_ids == ()
    assert second.unresolved_action_ids == ()
    assert first.rollout_count == second.rollout_count
    assert first.candidate_scores == second.candidate_scores
    assert first.tie_action_ids == second.tie_action_ids
    assert first.unique_best_action_id == second.unique_best_action_id
    assert first.evidence_hash == second.evidence_hash
    assert first.rollout_count == 9, first.rollout_count  # 3 semantic actions x 3 paired samples
    assert evaluator.uses_hidden_information is False
    assert backend.uses_hidden_information is False

    print("PUBLIC_STATE_SEARCH_NATIVE_V1 = PASS")
    print("PAIRED_PUBLIC_SAMPLING = PASS")
    print("SEARCH_REPEAT_DETERMINISM = PASS")
    print(f"ROLLOUT_COUNT = {first.rollout_count}")
    print(f"TIE_COUNT = {len(first.tie_action_ids)}")
    print(f"UNIQUE_BEST = {first.unique_best_action_id}")
    print(f"EVIDENCE_HASH = {first.evidence_hash}")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
