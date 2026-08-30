#!/usr/bin/env python3
"""Freeze blind pristine Cultist turn-1 discovery/holdout suites.

Each source seed creates a real Cultist opening. We end turn 0 without playing
a card, let the pinned simulator execute Incantation, and project only the
resulting public turn-1 observation. Source BattleContext, RNG, move history,
Ritual storage, counters, and miscInfo are never exported into formal state.
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

from roguelike_ai.sts1_teacher.contract import canonical_json
from roguelike_ai.sts1_teacher.public_run_state import simulator_public_run_state
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter

TARGET_CASES = 24
SUITES = {
    "discovery": {
        "seed_file": ROOT / "tests" / "data" / "sts1_phase1_cultist_turn1_discovery_seeds_64.txt",
        "min_seed": 650001,
        "max_seed": 650064,
        "schema": "sts1-phase1-cultist-turn1-pristine-discovery-suite-v1",
    },
    "holdout": {
        "seed_file": ROOT / "tests" / "data" / "sts1_phase1_cultist_turn1_fresh_holdout_seeds_64.txt",
        "min_seed": 660001,
        "max_seed": 660064,
        "schema": "sts1-phase1-cultist-turn1-pristine-fresh-holdout-suite-v1",
    },
}


def load_seeds(spec: dict) -> list[int]:
    path = spec["seed_file"]
    seeds = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != 64 or len(set(seeds)) != 64:
        raise RuntimeError(f"cultist_turn1_source_seed_contract:{len(seeds)}:{len(set(seeds))}")
    if seeds != list(range(spec["min_seed"], spec["max_seed"] + 1)):
        raise RuntimeError("cultist_turn1_source_seed_range_or_order_drift")
    return seeds


def public_run_state(gc) -> dict:
    run = simulator_public_run_state(gc)
    run.update(
        {
            "floor": 1,
            "act": 1,
            "character": "IRONCLAD",
            "ascension_level": 0,
            "room": "COMBAT",
            "screen_type": "NONE",
            "screen_choices": [],
            "rewards": [],
            "map_choices": [],
        }
    )
    return run


def suite_digest(records: list[dict]) -> str:
    payload = "\n".join(canonical_json(record) for record in records) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def end_first_player_turn_without_cards(bc) -> None:
    end_turn = next(
        (action for action in sts.get_legal_actions(bc) if action.action_type == sts.SearchActionType.END_TURN),
        None,
    )
    if end_turn is None:
        raise RuntimeError("cultist_turn1_opening_end_turn_missing")
    end_turn.execute(bc)


def assert_turn_one_public_boundary(state: dict, *, source_seed: int) -> None:
    if state.get("turn") != 1:
        raise RuntimeError(f"cultist_turn1_not_reached:{source_seed}:{state.get('turn')}")
    if state.get("energy") != 3 or state.get("block") != 0:
        raise RuntimeError(
            f"cultist_turn1_player_boundary:{source_seed}:energy={state.get('energy')}:block={state.get('block')}"
        )
    piles = {
        "hand": len(state.get("hand", [])),
        "draw": len(state.get("draw_pile", [])),
        "discard": len(state.get("discard_pile", [])),
        "exhaust": len(state.get("exhaust_pile", [])),
    }
    if piles != {"hand": 5, "draw": 0, "discard": 5, "exhaust": 0}:
        raise RuntimeError(f"cultist_turn1_pile_boundary:{source_seed}:{piles}")
    if state.get("powers") != []:
        raise RuntimeError(f"cultist_turn1_player_powers:{source_seed}:{state.get('powers')}")
    enemies = state.get("enemies", [])
    if len(enemies) != 1:
        raise RuntimeError(f"cultist_turn1_enemy_count:{source_seed}:{len(enemies)}")
    enemy = enemies[0]
    if enemy.get("name") != "CULTIST" or enemy.get("intent") != "CULTIST_DARK_STRIKE":
        raise RuntimeError(f"cultist_turn1_enemy_scope:{source_seed}:{enemies}")
    if enemy.get("hp") != enemy.get("max_hp") or enemy.get("block") != 0:
        raise RuntimeError(f"cultist_turn1_enemy_pristine_boundary:{source_seed}:{enemy}")
    if enemy.get("powers") != []:
        raise RuntimeError(f"cultist_turn1_projected_enemy_powers:{source_seed}:{enemy.get('powers')}")


def generate(kind: str) -> dict:
    spec = SUITES[kind]
    adapter = SimulatorCombatAdapter()
    records: list[dict] = []
    seen_signatures: set[str] = set()
    rejected: list[dict] = []

    for source_seed in load_seeds(spec):
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, source_seed, 0)
        gc.floor_num = 1
        gc.cur_room = sts.Room.MONSTER
        bc = sts.BattleContext()
        bc.init_encounter(gc, sts.MonsterEncounter.CULTIST)
        if int(bc.turn) != 0 or bc.monsters[0].intent != "CULTIST_INCANTATION":
            rejected.append({"source_seed": source_seed, "reason": "unexpected_cultist_opening"})
            continue

        end_first_player_turn_without_cards(bc)
        native_actions = list(sts.get_legal_actions(bc))
        if not native_actions:
            rejected.append({"source_seed": source_seed, "reason": "turn1_no_native_legal_actions"})
            continue

        run = public_run_state(gc)
        projected = adapter.adapt(bc, legal_actions=native_actions, run_state=run)
        try:
            assert_turn_one_public_boundary(projected, source_seed=source_seed)
            admitted_state = attach_reconstruction_capabilities(projected, run_state=run)
            context = require_public_reconstruction(admitted_state)
        except Exception as exc:
            rejected.append({"source_seed": source_seed, "reason": str(exc)})
            continue

        if context.decision_signature in seen_signatures:
            continue
        seen_signatures.add(context.decision_signature)
        records.append(
            {
                "case_id": f"cultist-turn1-{kind}-{len(records):02d}",
                "source_seed_provenance_only": source_seed,
                "decision_signature": context.decision_signature,
                "public_state": admitted_state,
            }
        )
        if len(records) == TARGET_CASES:
            break

    if len(records) != TARGET_CASES:
        raise RuntimeError(
            f"insufficient_unique_admitted_cultist_turn1_cases:{kind}:{len(records)}/{TARGET_CASES}:rejected={rejected}"
        )

    seed_file = spec["seed_file"]
    return {
        "schema_version": spec["schema"],
        "suite_kind": kind,
        "scope": "Cultist turn-1 pristine no-turn0-card-play boundary only",
        "frozen_before_quality_results": True,
        "selection_rule": (
            f"first 24 unique fully admitted public decision signatures after ending Cultist turn 0 without cards "
            f"in committed {spec['min_seed']}..{spec['max_seed']} source-seed order"
        ),
        "source_seed_file": str(seed_file.relative_to(ROOT)),
        "source_seed_file_sha256": hashlib.sha256(seed_file.read_bytes()).hexdigest(),
        "quality_results_observed_during_selection": 0,
        "source_battle_context_exported": 0,
        "source_hidden_rng_exported": 0,
        "source_move_history_exported": 0,
        "source_counter_history_exported": 0,
        "source_hidden_miscinfo_exported": 0,
        "case_count": len(records),
        "suite_digest": suite_digest(records),
        "cases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-kind", choices=sorted(SUITES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = generate(args.suite_kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CULTIST_TURN1_{args.suite_kind.upper()}_SUITE_FREEZE = PASS")
    print(f"CASE_COUNT = {suite['case_count']}")
    print(f"SUITE_DIGEST = {suite['suite_digest']}")
    print(f"SOURCE_SEED_FILE_SHA256 = {suite['source_seed_file_sha256']}")
    print("QUALITY_RESULTS_OBSERVED_DURING_SELECTION = 0")
    print("SOURCE_RAW_BATTLE_CONTEXT_INPUT = 0")
    print("SOURCE_HIDDEN_RNG_ACCESS = 0")
    print("SOURCE_MOVE_HISTORY_ACCESS = 0")
    print("SOURCE_COUNTER_HISTORY_ACCESS = 0")
    print("SOURCE_HIDDEN_MISCINFO_ACCESS = 0")


if __name__ == "__main__":
    main()
