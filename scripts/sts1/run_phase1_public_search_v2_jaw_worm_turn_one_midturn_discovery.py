#!/usr/bin/env python3
"""Discovery-only Jaw Worm turn-1 midturn Search V2 experiment.

This script MUST NOT read the retired 420xxx holdout. It creates a separate
32-case development set from committed 430xxx seeds using the same public-only
one-card generation rule, then evaluates several public-only fallback policies.
A candidate is eligible only if it strictly beats raw Search, Simple, and Random
on this discovery set. Discovery PASS is not a Phase-1 or holdout PASS.
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
from roguelike_ai.sts1_teacher.contract import DecisionContext, canonical_json
from roguelike_ai.sts1_teacher.native_rollout import NativePublicJawWormRolloutV1
from roguelike_ai.sts1_teacher.public_run_state import simulator_public_run_state
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.sampling import PairedPublicSampleEvaluator, public_sample
from roguelike_ai.sts1_teacher.search import PublicStateSearch, SearchConfig
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter

SCHEMA = "sts1-phase1-public-search-v2-jaw-worm-turn-one-midturn-discovery-v1"
SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v2_jaw_worm_turn_one_midturn_discovery_seeds_64.txt"
RETIRED_HOLDOUT_SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v1_jaw_worm_turn_one_midturn_source_seeds_64.txt"
TARGET_CASES = 32
SEARCH_SAMPLE_COUNT = 8
ORACLE_SAMPLE_COUNT = 4
ORACLE_MCTS_SIMS = 2000
SEARCH_SAMPLING_SEED = 20260830
MARGIN_THRESHOLDS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
STATIC_CANDIDATES = (
    "bellow_else_search",
    "block_positive_else_search",
    "bellow_or_block_positive_else_search",
)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_seed_file(path: Path) -> list[int]:
    return [
        int(x.strip())
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]


def load_discovery_seeds() -> list[int]:
    seeds = load_seed_file(SEED_FILE)
    if len(seeds) != 64 or len(set(seeds)) != 64:
        raise RuntimeError("midturn_v2_discovery_seed_contract_failed")
    if min(seeds) != 430001 or max(seeds) != 430064:
        raise RuntimeError("midturn_v2_discovery_seed_range_drift")
    retired = set(load_seed_file(RETIRED_HOLDOUT_SEED_FILE))
    if retired.intersection(seeds):
        raise RuntimeError("midturn_v2_discovery_overlaps_retired_420xxx")
    return seeds


def run_state(gc) -> dict:
    run = simulator_public_run_state(gc)
    run.update({
        "floor": 1,
        "act": 1,
        "character": "IRONCLAD",
        "ascension_level": 0,
        "room": "COMBAT",
        "screen_type": "NONE",
        "screen_choices": [],
        "rewards": [],
        "map_choices": [],
    })
    return run


def enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name if isinstance(name, str) else value).rsplit(".", 1)[-1].upper()


def end_first_turn(bc) -> None:
    action = next(
        (a for a in sts.get_legal_actions(bc) if enum_name(a.action_type) == "END_TURN"),
        None,
    )
    if action is None:
        raise RuntimeError("midturn_v2_opening_end_turn_missing")
    action.execute(bc)


def project(adapter, bc, run: dict) -> dict:
    return adapter.adapt(bc, legal_actions=list(sts.get_legal_actions(bc)), run_state=run)


def choose_generation_card(context: DecisionContext):
    cards = sorted(
        (a for a in context.legal_actions if a.payload.get("kind") == "play_card"),
        key=lambda a: a.action_id,
    )
    if not cards:
        raise RuntimeError("midturn_v2_no_public_generation_card")
    idx = int(context.decision_signature[:16], 16) % len(cards)
    return cards[idx]


def execute_public_action(bc, action) -> str:
    hand_idx = int(action.payload["hand_index"]) - 1
    target = action.payload.get("target_index")
    hand = list(bc.hand)
    played_name = str(hand[hand_idx].name).upper()
    for native in sts.get_legal_actions(bc):
        if enum_name(native.action_type) != "CARD":
            continue
        if int(native.source_idx) != hand_idx:
            continue
        if target is not None and int(native.target_idx) != int(target):
            continue
        native.execute(bc)
        return played_name
    raise RuntimeError("midturn_v2_generation_native_action_missing")


def make_cases() -> list[dict]:
    adapter = SimulatorCombatAdapter()
    cases: list[dict] = []
    seen: set[str] = set()
    for seed in load_discovery_seeds():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
        gc.floor_num = 1
        gc.cur_room = sts.Room.MONSTER
        bc = sts.BattleContext()
        bc.init_encounter(gc, sts.MonsterEncounter.JAW_WORM)
        if int(bc.turn) != 0:
            raise RuntimeError(f"midturn_v2_opening_turn_drift:{seed}:{bc.turn}")
        end_first_turn(bc)
        run = run_state(gc)
        boundary = project(adapter, bc, run)
        boundary_state = attach_reconstruction_capabilities(boundary, run_state=run)
        boundary_context = require_public_reconstruction(boundary_state)
        generation_action = choose_generation_card(boundary_context)
        played_card = execute_public_action(bc, generation_action)
        midturn = project(adapter, bc, run)
        state = attach_reconstruction_capabilities(midturn, run_state=run)
        context = require_public_reconstruction(state)
        if context.state.get("turn") != 1 or len(context.state.get("hand", [])) != 4:
            raise RuntimeError(f"midturn_v2_scope_drift:{seed}")
        enemy = context.state["enemies"][0]
        if enemy.get("name") != "JAW_WORM":
            raise RuntimeError(f"midturn_v2_enemy_drift:{seed}")
        if context.decision_signature in seen:
            continue
        seen.add(context.decision_signature)
        cases.append({
            "case_id": f"jaw-midturn-v2-dev-{len(cases):02d}",
            "source_seed_provenance_only": seed,
            "generation_played_card": played_card,
            "decision_signature": context.decision_signature,
            "public_state": state,
        })
        if len(cases) == TARGET_CASES:
            break
    if len(cases) != TARGET_CASES:
        raise RuntimeError(f"midturn_v2_discovery_insufficient_cases:{len(cases)}")
    return cases


def oracle_scores(context, backend, config) -> dict[str, float]:
    groups: dict[str, list] = {}
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


def search_top(context, search_result, simple) -> tuple[str, float, bool]:
    scores: dict[str, float] = {}
    ids: dict[str, list[str]] = {}
    for item in search_result.candidate_scores:
        scores[item.semantic_key] = item.score
        ids.setdefault(item.semantic_key, []).append(item.action_id)
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if not ranked:
        return simple.action_id, 0.0, True
    top_key, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
    margin = top_score - second_score
    tied = len(ranked) > 1 and abs(margin) <= 1e-9
    if tied:
        return simple.action_id, margin, True
    simple_key = context.action_by_id(simple.action_id).semantic_key
    if simple_key == top_key:
        return simple.action_id, margin, False
    return sorted(ids[top_key])[0], margin, False


def choose_static_candidate(name: str, context, search_result, simple) -> tuple[str, bool, float]:
    search_id, margin, search_fell_back_for_tie = search_top(context, search_result, simple)
    enemy = context.state["enemies"][0]
    intent = str(enemy.get("intent", ""))
    block = int(context.state.get("block", 0))
    if name == "bellow_else_search":
        use_simple = intent == "JAW_WORM_BELLOW"
    elif name == "block_positive_else_search":
        use_simple = block > 0
    elif name == "bellow_or_block_positive_else_search":
        use_simple = intent == "JAW_WORM_BELLOW" or block > 0
    else:
        raise RuntimeError(f"unknown_static_candidate:{name}")
    if use_simple:
        return simple.action_id, True, margin
    return search_id, search_fell_back_for_tie, margin


def choose_margin_candidate(context, search_result, simple, threshold: float) -> tuple[str, bool, float]:
    search_id, margin, tied_fallback = search_top(context, search_result, simple)
    use_simple = tied_fallback or margin <= threshold
    return (simple.action_id if use_simple else search_id), use_simple, margin


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
    baseline_counts = {"search": 0, "simple": 0, "random": 0}
    candidate_names = list(STATIC_CANDIDATES) + [f"margin_{t:g}" for t in MARGIN_THRESHOLDS]
    candidate_counts = {name: 0 for name in candidate_names}
    fallback_counts = {name: 0 for name in candidate_names}
    rows: list[dict] = []
    unresolved = 0

    for record in cases:
        state = record["public_state"]
        context = require_public_reconstruction(state)
        backend = NativePublicJawWormRolloutV1(state, sts)
        search_result = PublicStateSearch(PairedPublicSampleEvaluator(backend), config).run(context)
        if not search_result.resolved or search_result.timed_out or search_result.unresolved_action_ids:
            unresolved += 1
        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError("midturn_v2_discovery_baseline_missing")
        scores = oracle_scores(context, backend, config)
        ties = oracle_ties(scores)
        base = {
            "search": conservative_tie_agreement(tuple(sorted(search_result.tie_action_ids)), ties),
            "simple": single_action_agreement(simple.action_id, ties),
            "random": single_action_agreement(random_action.action_id, ties),
        }
        for key in baseline_counts:
            baseline_counts[key] += int(base[key])

        candidates: dict[str, dict] = {}
        for name in STATIC_CANDIDATES:
            selected, fallback, margin = choose_static_candidate(name, context, search_result, simple)
            ok = single_action_agreement(selected, ties)
            candidate_counts[name] += int(ok)
            fallback_counts[name] += int(fallback)
            candidates[name] = {
                "selected_action_id": selected,
                "agreement": ok,
                "used_simple_fallback": fallback,
                "semantic_margin": margin,
            }
        for threshold in MARGIN_THRESHOLDS:
            name = f"margin_{threshold:g}"
            selected, fallback, margin = choose_margin_candidate(context, search_result, simple, threshold)
            ok = single_action_agreement(selected, ties)
            candidate_counts[name] += int(ok)
            fallback_counts[name] += int(fallback)
            candidates[name] = {
                "selected_action_id": selected,
                "agreement": ok,
                "used_simple_fallback": fallback,
                "semantic_margin": margin,
            }

        rows.append({
            "case_id": record["case_id"],
            "source_seed_provenance_only": record["source_seed_provenance_only"],
            "generation_played_card_diagnostic_only": record["generation_played_card"],
            "public_intent": str(context.state["enemies"][0].get("intent", "")),
            "public_block": int(context.state.get("block", 0)),
            "decision_signature": context.decision_signature,
            "oracle_tie_ids": list(ties),
            "baseline_agreement": base,
            "candidates": candidates,
        })

    eligible = [
        name for name in candidate_names
        if candidate_counts[name] > baseline_counts["search"]
        and candidate_counts[name] > baseline_counts["simple"]
        and candidate_counts[name] > baseline_counts["random"]
    ]
    if eligible:
        best = max(candidate_counts[name] for name in eligible)
        # Stable tie break: fewer fallbacks first, then lexical candidate name.
        chosen = min(
            (name for name in eligible if candidate_counts[name] == best),
            key=lambda name: (fallback_counts[name], name),
        )
        verdict = "PASS"
        reasons: list[str] = []
    else:
        chosen = None
        verdict = "FAIL"
        reasons = ["no_public_candidate_strictly_beats_search_simple_and_random_on_discovery"]

    summary = {
        "schema_version": SCHEMA,
        "scope": "Jaw Worm turn-1 one-card midturn V2 discovery only; retired 420xxx holdout not read",
        "phase1_gate_claimed": False,
        "case_count": len(cases),
        "seed_file_sha256": hashlib.sha256(SEED_FILE.read_bytes()).hexdigest(),
        "retired_420xxx_overlap": 0,
        "retired_420xxx_quality_rows_read": 0,
        "baseline_counts": baseline_counts,
        "candidate_counts": candidate_counts,
        "fallback_counts": fallback_counts,
        "chosen_candidate": chosen,
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
        raise RuntimeError("midturn_v2_discovery_not_deterministic")
    out = ROOT / "evidence" / "sts1" / "phase1_jaw_worm_turn_one_midturn_v2_discovery"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(rows1, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final = dict(summary1)
    final.update({"full_repeats": 2, "deterministic": True})
    (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True))
    print(f"JAW_WORM_TURN_ONE_MIDTURN_V2_DISCOVERY = {final['discovery_verdict']}")
    print("RETIRED_420XXX_HOLDOUT_READ = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
