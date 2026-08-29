#!/usr/bin/env python3
"""Discovery-only Gremlin Nob Search V2 experiment.

This script MUST NOT read the frozen 340xxx V1 holdout. It generates a separate
32-case development set from committed 350xxx seeds, then evaluates a simple
public-only confidence fallback: use Search when its top semantic score is
clearly ahead; otherwise use the existing simple public heuristic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
BUILD = ROOT / "external" / "sts_lightspeed" / "source" / "build-probe"
for path in (SRC, BUILD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import slaythespire as sts

from roguelike_ai.sts1_teacher.baselines import deterministic_random_legal, simple_public_heuristic
from roguelike_ai.sts1_teacher.benchmark import conservative_tie_agreement, oracle_ties, single_action_agreement
from roguelike_ai.sts1_teacher.contract import canonical_json
from roguelike_ai.sts1_teacher.native_rollout import NativePublicGremlinNobRolloutV1
from roguelike_ai.sts1_teacher.public_run_state import simulator_public_run_state
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator, public_sample
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter

SCHEMA = "sts1-phase1-public-search-v2-gremlin-nob-discovery-v1"
SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v2_gremlin_nob_discovery_seeds_64.txt"
TARGET_CASES = 32
SEARCH_SAMPLE_COUNT = 8
ORACLE_SAMPLE_COUNT = 4
ORACLE_MCTS_SIMS = 2000
SEARCH_SAMPLING_SEED = 20260830
THRESHOLDS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_seeds() -> list[int]:
    seeds = [int(x.strip()) for x in SEED_FILE.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if len(seeds) != 64 or len(set(seeds)) != 64:
        raise RuntimeError("nob_v2_discovery_seed_contract_failed")
    if any(seed < 350001 or seed > 350064 for seed in seeds):
        raise RuntimeError("nob_v2_discovery_seed_range_drift")
    return seeds


def run_state(gc) -> dict:
    run = simulator_public_run_state(gc)
    run.update({
        "floor": 1, "act": 1, "character": "IRONCLAD", "ascension_level": 0,
        "room": "COMBAT", "screen_type": "NONE", "screen_choices": [],
        "rewards": [], "map_choices": [],
    })
    return run


def make_cases() -> list[dict]:
    adapter = SimulatorCombatAdapter()
    cases = []
    seen = set()
    for seed in load_seeds():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
        gc.floor_num = 1
        gc.cur_room = sts.Room.MONSTER
        bc = sts.BattleContext()
        bc.init_encounter(gc, sts.MonsterEncounter.GREMLIN_NOB)
        native_actions = list(sts.get_legal_actions(bc))
        run = run_state(gc)
        projected = adapter.adapt(bc, legal_actions=native_actions, run_state=run)
        state = attach_reconstruction_capabilities(projected, run_state=run)
        context = require_public_reconstruction(state)
        enemy = state["enemies"][0]
        if state.get("turn") != 0 or enemy.get("name") != "GREMLIN_NOB" or enemy.get("intent") != "GREMLIN_NOB_BELLOW":
            raise RuntimeError(f"nob_v2_discovery_scope_drift:{seed}")
        if context.decision_signature in seen:
            continue
        seen.add(context.decision_signature)
        cases.append({
            "case_id": f"nob-v2-dev-{len(cases):02d}",
            "source_seed_provenance_only": seed,
            "decision_signature": context.decision_signature,
            "public_state": state,
        })
        if len(cases) == TARGET_CASES:
            break
    if len(cases) != TARGET_CASES:
        raise RuntimeError(f"nob_v2_discovery_insufficient_cases:{len(cases)}")
    return cases


def oracle_scores(context, backend, config) -> dict[str, float]:
    groups = {}
    for action in context.legal_actions:
        groups.setdefault(action.semantic_key, []).append(action)
    totals = {action.action_id: 0.0 for action in context.legal_actions}
    for sample_index in range(ORACLE_SAMPLE_COUNT):
        sample = public_sample(context, sample_index=sample_index, config=config)
        for semantic_key in sorted(groups):
            members = sorted(groups[semantic_key], key=lambda x: x.action_id)
            representative = members[0]
            bc = backend._build(context, sample)
            native_action = backend._native_action(bc, representative)
            score = float(sts.judge_branch_action(bc, native_action, ORACLE_MCTS_SIMS)["score"])
            for member in members:
                totals[member.action_id] += score
    return {k: v / ORACLE_SAMPLE_COUNT for k, v in totals.items()}


def select_with_fallback(context, search_result, simple, threshold: float) -> tuple[str, float, bool]:
    group_scores = {}
    group_ids = {}
    for item in search_result.candidate_scores:
        group_scores[item.semantic_key] = item.score
        group_ids.setdefault(item.semantic_key, []).append(item.action_id)
    ranked = sorted(group_scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if not ranked:
        return simple.action_id, 0.0, True
    top_key, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
    margin = top_score - second_score
    tied_top = len(ranked) > 1 and abs(margin) <= 1e-9
    fallback = tied_top or margin <= threshold
    if fallback:
        return simple.action_id, margin, True
    simple_key = context.action_by_id(simple.action_id).semantic_key
    if simple_key == top_key:
        return simple.action_id, margin, False
    return sorted(group_ids[top_key])[0], margin, False


def run_once(cases: list[dict]) -> tuple[list[dict], dict]:
    config = SearchConfig(
        samples_per_semantic_action=SEARCH_SAMPLE_COUNT,
        rollout_budget=256,
        node_budget=16384,
        max_depth=30,
        timeout_ms=60_000,
        tie_tolerance=1e-9,
        sampling_seed=SEARCH_SAMPLING_SEED,
    )
    counts = {"search": 0, "simple": 0, "random": 0}
    candidate_counts = {str(t): 0 for t in THRESHOLDS}
    fallback_counts = {str(t): 0 for t in THRESHOLDS}
    rows = []
    unresolved = 0

    for record in cases:
        state = record["public_state"]
        context = require_public_reconstruction(state)
        backend = NativePublicGremlinNobRolloutV1(state, sts)
        search_result = PublicStateSearch(PairedPublicSampleEvaluator(backend), config).run(context)
        if not search_result.resolved or search_result.timed_out or search_result.unresolved_action_ids:
            unresolved += 1
        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError("nob_v2_discovery_baseline_missing")
        scores = oracle_scores(context, backend, config)
        ties = oracle_ties(scores)
        base_agreement = {
            "search": conservative_tie_agreement(tuple(sorted(search_result.tie_action_ids)), ties),
            "simple": single_action_agreement(simple.action_id, ties),
            "random": single_action_agreement(random_action.action_id, ties),
        }
        for key in counts:
            counts[key] += int(base_agreement[key])

        candidates = {}
        for threshold in THRESHOLDS:
            selected, margin, fallback = select_with_fallback(context, search_result, simple, threshold)
            ok = single_action_agreement(selected, ties)
            candidate_counts[str(threshold)] += int(ok)
            fallback_counts[str(threshold)] += int(fallback)
            candidates[str(threshold)] = {
                "selected_action_id": selected,
                "agreement": ok,
                "semantic_margin": margin,
                "used_simple_fallback": fallback,
            }
        rows.append({
            "case_id": record["case_id"],
            "source_seed_provenance_only": record["source_seed_provenance_only"],
            "decision_signature": context.decision_signature,
            "oracle_tie_ids": list(ties),
            "baseline_agreement": base_agreement,
            "candidates": candidates,
        })

    eligible = [
        t for t in THRESHOLDS
        if candidate_counts[str(t)] > counts["search"] and candidate_counts[str(t)] > counts["simple"]
    ]
    if eligible:
        best_count = max(candidate_counts[str(t)] for t in eligible)
        chosen = min(t for t in eligible if candidate_counts[str(t)] == best_count)
        verdict = "PASS"
        reasons = []
    else:
        chosen = None
        verdict = "FAIL"
        reasons = ["no_threshold_strictly_beats_both_search_and_simple_on_discovery"]

    summary = {
        "schema_version": SCHEMA,
        "scope": "Gremlin Nob V2 discovery only; 340xxx frozen holdout not read",
        "phase1_gate_claimed": False,
        "case_count": len(cases),
        "seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "baseline_counts": counts,
        "candidate_counts": candidate_counts,
        "fallback_counts": fallback_counts,
        "chosen_threshold": chosen,
        "discovery_verdict": verdict,
        "reasons": reasons,
        "search_unresolved": unresolved,
        "rows_digest": digest(rows),
    }
    return rows, summary


def main() -> None:
    if not hasattr(sts, "judge_branch_action"):
        raise RuntimeError("diagnostic_oracle_binding_missing")
    cases = make_cases()
    rows1, summary1 = run_once(cases)
    rows2, summary2 = run_once(cases)
    if rows1 != rows2 or summary1 != summary2:
        raise RuntimeError("nob_v2_discovery_not_deterministic")
    out = ROOT / "evidence" / "sts1" / "phase1_nob_search_v2_discovery"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
    (out / "rows.json").write_text(json.dumps(rows1, indent=2, sort_keys=True) + "\n")
    final = dict(summary1)
    final.update({"full_repeats": 2, "deterministic": True})
    (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True))
    print(f"NOB_SEARCH_V2_DISCOVERY = {final['discovery_verdict']}")
    print("FROZEN_340XXX_HOLDOUT_READ = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
