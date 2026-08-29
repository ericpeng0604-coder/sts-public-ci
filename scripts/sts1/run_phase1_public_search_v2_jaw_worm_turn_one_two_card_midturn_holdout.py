#!/usr/bin/env python3
"""Fresh holdout for the frozen two-card Jaw Worm midturn V2 candidate.

The candidate was chosen on 460xxx discovery and frozen before this script is
allowed to observe 470xxx outcomes. This runner never reads 450xxx V1 quality
rows or 460xxx discovery quality rows. It regenerates public states from the
fresh precommitted 470xxx source seeds and evaluates the frozen margin-zero
Search->Simple tie fallback against Raw Search, Simple, and Random.
"""
from __future__ import annotations

import argparse
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
from roguelike_ai.sts1_teacher.two_card_midturn_policy_v2 import (
    MARGIN_THRESHOLD,
    POLICY_ID,
    choose_two_card_midturn_v2,
)

SCHEMA = "sts1-phase1-public-search-v2-jaw-worm-turn-one-two-card-midturn-holdout-v1"
SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v2_jaw_worm_turn_one_two_card_midturn_holdout_seeds_64.txt"
FROZEN_V1_SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v1_jaw_worm_turn_one_two_card_midturn_source_seeds_64.txt"
DISCOVERY_SEED_FILE = ROOT / "tests" / "data" / "sts1_phase1_public_search_v2_jaw_worm_turn_one_two_card_midturn_discovery_seeds_64.txt"
TARGET_CASES = 24
SEARCH_SAMPLE_COUNT = 8
ORACLE_SAMPLE_COUNT = 4
ORACLE_MCTS_SIMS = 2000
SEARCH_SAMPLING_SEED = 20260830
STARTER_NAMES = frozenset({"STRIKE", "DEFEND", "BASH"})
# Reuse the exact discovery generation salt/mechanism; only source seeds change.
GENERATION_SALT = "two-card-v2-discovery-generation"


def sha_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_seed_file(path: Path) -> list[int]:
    return [
        int(x.strip())
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]


def load_holdout_seeds() -> list[int]:
    seeds = load_seed_file(SEED_FILE)
    if len(seeds) != 64 or len(set(seeds)) != 64:
        raise RuntimeError("two_card_v2_holdout_seed_contract_failed")
    if min(seeds) != 470001 or max(seeds) != 470064:
        raise RuntimeError("two_card_v2_holdout_seed_range_drift")
    frozen = set(load_seed_file(FROZEN_V1_SEED_FILE))
    discovery = set(load_seed_file(DISCOVERY_SEED_FILE))
    if set(seeds).intersection(frozen):
        raise RuntimeError("two_card_v2_holdout_overlaps_450xxx")
    if set(seeds).intersection(discovery):
        raise RuntimeError("two_card_v2_holdout_overlaps_460xxx")
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
    action = next((a for a in sts.get_legal_actions(bc) if enum_name(a.action_type) == "END_TURN"), None)
    if action is None:
        raise RuntimeError("two_card_v2_holdout_opening_end_turn_missing")
    action.execute(bc)


def project(adapter, bc, run: dict) -> dict:
    return adapter.adapt(bc, legal_actions=list(sts.get_legal_actions(bc)), run_state=run)


def admitted_context(adapter, bc, run: dict) -> tuple[dict, DecisionContext]:
    projected = project(adapter, bc, run)
    state = attach_reconstruction_capabilities(projected, run_state=run)
    return state, require_public_reconstruction(state)


def choose_generation_card(context: DecisionContext, ordinal: int):
    cards = sorted(
        (a for a in context.legal_actions if a.payload.get("kind") == "play_card"),
        key=lambda a: a.action_id,
    )
    if not cards:
        raise RuntimeError(f"two_card_v2_holdout_no_public_generation_card:{ordinal}")
    selector = hashlib.sha256(
        f"{GENERATION_SALT}:{ordinal}:{context.decision_signature}".encode("utf-8")
    ).hexdigest()
    return cards[int(selector[:16], 16) % len(cards)]


def execute_public_action(bc, action) -> str:
    hand_idx = int(action.payload["hand_index"]) - 1
    target = action.payload.get("target_index")
    hand = list(bc.hand)
    played_name = str(hand[hand_idx].name).upper()
    if played_name not in STARTER_NAMES:
        raise RuntimeError(f"two_card_v2_holdout_generation_nonstarter:{played_name}")
    for native in sts.get_legal_actions(bc):
        if enum_name(native.action_type) != "CARD":
            continue
        if int(native.source_idx) != hand_idx:
            continue
        if target is not None and int(native.target_idx) != int(target):
            continue
        native.execute(bc)
        return played_name
    raise RuntimeError("two_card_v2_holdout_generation_native_action_missing")


def make_cases() -> list[dict]:
    adapter = SimulatorCombatAdapter()
    cases: list[dict] = []
    seen: set[str] = set()
    for seed in load_holdout_seeds():
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
        gc.floor_num = 1
        gc.cur_room = sts.Room.MONSTER
        bc = sts.BattleContext()
        bc.init_encounter(gc, sts.MonsterEncounter.JAW_WORM)
        if int(bc.turn) != 0:
            raise RuntimeError(f"two_card_v2_holdout_opening_turn_drift:{seed}:{bc.turn}")
        end_first_turn(bc)
        run = run_state(gc)

        _, context0 = admitted_context(adapter, bc, run)
        action1 = choose_generation_card(context0, 1)
        card1 = execute_public_action(bc, action1)
        _, context1 = admitted_context(adapter, bc, run)
        action2 = choose_generation_card(context1, 2)
        card2 = execute_public_action(bc, action2)

        state, context = admitted_context(adapter, bc, run)
        if context.state.get("turn") != 1 or len(context.state.get("hand", [])) != 3:
            raise RuntimeError(f"two_card_v2_holdout_scope_drift:{seed}")
        if context.state.get("draw_pile") != [] or len(context.state.get("discard_pile", [])) != 7:
            raise RuntimeError(f"two_card_v2_holdout_pile_drift:{seed}")
        if context.state["enemies"][0].get("name") != "JAW_WORM":
            raise RuntimeError(f"two_card_v2_holdout_enemy_drift:{seed}")
        if context.decision_signature in seen:
            continue
        seen.add(context.decision_signature)
        cases.append({
            "case_id": f"jaw-two-card-midturn-v2-holdout-{len(cases):02d}",
            "source_seed_provenance_only": seed,
            "generation_sequence": f"{card1}->{card2}",
            "decision_signature": context.decision_signature,
            "public_state": state,
        })
        if len(cases) == TARGET_CASES:
            break
    if len(cases) != TARGET_CASES:
        raise RuntimeError(f"two_card_v2_holdout_insufficient_cases:{len(cases)}")
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
    counts = {"v2": 0, "search": 0, "simple": 0, "random": 0}
    fallback_count = 0
    unresolved = 0
    rows: list[dict] = []

    for record in cases:
        state = record["public_state"]
        context = require_public_reconstruction(state)
        if context.decision_signature != record["decision_signature"]:
            raise RuntimeError(f"two_card_v2_holdout_signature_drift:{record['case_id']}")
        backend = NativePublicJawWormRolloutV1(state, sts)
        result = PublicStateSearch(PairedPublicSampleEvaluator(backend), config).run(context)
        if not result.resolved or result.timed_out or result.unresolved_action_ids:
            unresolved += 1

        simple = simple_public_heuristic(context)
        random_action = deterministic_random_legal(context, benchmark_seed=0)
        if simple is None or random_action is None:
            raise RuntimeError(f"two_card_v2_holdout_baseline_missing:{record['case_id']}")

        ties = oracle_ties(oracle_scores(context, backend, config))
        v2 = choose_two_card_midturn_v2(context, result, simple.action_id)
        fallback_count += int(v2.used_simple_fallback)
        agreement = {
            "v2": single_action_agreement(v2.action_id, ties),
            "search": conservative_tie_agreement(tuple(sorted(result.tie_action_ids)), ties),
            "simple": single_action_agreement(simple.action_id, ties),
            "random": single_action_agreement(random_action.action_id, ties),
        }
        for key in counts:
            counts[key] += int(agreement[key])

        rows.append({
            "case_id": record["case_id"],
            "source_seed_provenance_only": record["source_seed_provenance_only"],
            "generation_sequence": record["generation_sequence"],
            "decision_signature": context.decision_signature,
            "oracle_tie_ids": list(ties),
            "v2_action_id": v2.action_id,
            "v2_used_simple_fallback": v2.used_simple_fallback,
            "v2_semantic_margin": v2.semantic_margin,
            "search_tie_ids": list(sorted(result.tie_action_ids)),
            "simple_action_id": simple.action_id,
            "random_action_id": random_action.action_id,
            "agreement": agreement,
        })

    reasons: list[str] = []
    if unresolved:
        reasons.append(f"search_unresolved:{unresolved}")
    for baseline in ("search", "simple", "random"):
        if not counts["v2"] > counts[baseline]:
            reasons.append(f"v2_not_strictly_better_than_{baseline}")

    summary = {
        "schema_version": SCHEMA,
        "scope": "Jaw Worm turn-1 exactly-two-starter-card midturn V2 fresh holdout",
        "candidate_policy_id": POLICY_ID,
        "candidate_margin_threshold": MARGIN_THRESHOLD,
        "candidate_frozen_before_holdout_results": True,
        "phase1_gate_claimed": False,
        "case_count": len(cases),
        "agreement_counts": counts,
        "agreement_rates": {k: counts[k] / len(cases) for k in counts},
        "fallback_count": fallback_count,
        "search_unresolved": unresolved,
        "quality_verdict": "PASS" if not reasons else "FAIL",
        "quality_reasons": reasons,
        "holdout_seed_file_sha256": sha_bytes(SEED_FILE),
        "frozen_450xxx_overlap": 0,
        "discovery_460xxx_overlap": 0,
        "frozen_450xxx_quality_rows_read": 0,
        "discovery_460xxx_quality_rows_read": 0,
        "hand_size_shortcut_generalized": False,
        "nonstarter_cards_admitted": 0,
        "suite_digest": digest([
            {
                "case_id": item["case_id"],
                "decision_signature": item["decision_signature"],
                "generation_sequence": item["generation_sequence"],
                "source_seed_provenance_only": item["source_seed_provenance_only"],
            }
            for item in cases
        ]),
        "rows_digest": digest(rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if POLICY_ID != "jaw-worm-turn1-two-card-midturn-margin0-else-simple-v1":
        raise RuntimeError(f"unexpected_candidate_policy:{POLICY_ID}")
    if MARGIN_THRESHOLD != 0.0:
        raise RuntimeError(f"candidate_threshold_drift:{MARGIN_THRESHOLD}")

    cases = make_cases()
    rows1, summary1 = run_once(cases)
    rows2, summary2 = run_once(cases)
    if rows1 != rows2 or summary1 != summary2:
        raise RuntimeError("two_card_v2_holdout_not_deterministic")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "decisions.ndjson").write_text(
        "".join(canonical_json(row) + "\n" for row in rows1), encoding="utf-8"
    )
    final_summary = dict(summary1)
    final_summary["full_repeats"] = 2
    final_summary["deterministic"] = True
    (out / "summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(final_summary, indent=2, sort_keys=True))
    print("TWO_CARD_MIDTURN_V2_HOLDOUT_EXECUTION = PASS")
    print(f"TWO_CARD_MIDTURN_V2_HOLDOUT_VERDICT = {final_summary['quality_verdict']}")
    print(f"TWO_CARD_MIDTURN_V2_POLICY_ID = {POLICY_ID}")
    print("FROZEN_450XXX_QUALITY_ROWS_READ = 0")
    print("DISCOVERY_460XXX_QUALITY_ROWS_READ = 0")
    print("MIDTURN_HAND_SIZE_SHORTCUT_GENERALIZED = 0")
    print("NONSTARTER_CARDS_ADMITTED = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
