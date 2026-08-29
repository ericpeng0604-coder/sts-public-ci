#!/usr/bin/env python3
"""Run one shard/repeat of the frozen STS1 Phase-1 Arm B oracle benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
AGENT = ROOT / "external" / "sts-rl-agent" / "source"
SIM = ROOT / "external" / "sts_lightspeed" / "source"
BUILD = SIM / "build-probe"
CACHE = ROOT / ".cache" / "sts1-phase1-oracle"
SB = CACHE / "sts-bot"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BUILD) not in sys.path:
    sys.path.insert(0, str(BUILD))
if str(AGENT / "agent") not in sys.path:
    sys.path.insert(0, str(AGENT / "agent"))

from roguelike_ai.sts1_teacher.baselines import deterministic_random_legal, simple_public_heuristic
from roguelike_ai.sts1_teacher.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkContractError,
    BenchmarkDecision,
    ORACLE_MCTS_SIMS,
    exact_native_action_map,
    load_heldout_seeds,
    oracle_ties,
    semantic_score_consistent,
    sha256_file,
    summarize_rows,
    write_json,
    write_ndjson,
)
from roguelike_ai.sts1_teacher.contract import DecisionContext, PublicStateContractError
from roguelike_ai.sts1_teacher.jialeiv_armb import (
    JIALEIV_ARM_B_WEIGHT_GIT_BLOB_SHA,
    JIALEIV_VOCAB_GIT_BLOB_SHA,
    JialeivArmBPolicy,
)
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter

EXPECTED_AGENT = "b20eb2cac2f52b22fbb6c79900c309b51ea0a1db"
EXPECTED_SIM = "7476a81954020087da31d41d16fddf475746ec2d"
SEEDS_PATH = AGENT / "eval" / "eval_seeds_50.txt"
VOCAB_PATH = AGENT / "agent" / "armS_card_vocab.json"
ARMB_WEIGHT_PATH = AGENT / "weights" / "armB_model_B256x256.pt"
ARMG_WEIGHT_PATH = AGENT / "weights" / "armG_model_G128x128_15k.pt"
MAX_GAME_STEPS = 600
MAX_BATTLE_STEPS = 800


def _git_head(path: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _prepare_armg_compat_layout() -> None:
    """Mirror the accepted Baseline-v0 runtime layout before importing Arm G."""

    if CACHE.exists():
        shutil.rmtree(CACHE)
    SB.mkdir(parents=True, exist_ok=True)
    _link(BUILD, SB / "sim" / "sts_lightspeed" / "build312")
    _link(VOCAB_PATH, SB / "armS_card_vocab.json")
    _link(ARMG_WEIGHT_PATH, SB / "armG_model_G128x128_15k.pt")
    os.environ.update(
        {
            "STS_BOT_DIR": str(SB),
            "ARMG": "armG_model_G128x128_15k.pt",
            "ASC": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    if _git_head(AGENT) != EXPECTED_AGENT or _git_head(SIM) != EXPECTED_SIM:
        raise RuntimeError("pinned upstream source mismatch")
    if not BUILD.is_dir():
        raise RuntimeError("native build missing; run scripts/sts1/build_probe.py first")

    _prepare_armg_compat_layout()
    import torch
    import slaythespire as sts
    import armG_train as AG

    torch.set_num_threads(1)
    if not hasattr(sts, "judge_branch_action"):
        raise RuntimeError("phase1 oracle judge binding missing")

    expected_vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    if AG._vocab != expected_vocab or not AG._vocab:
        raise RuntimeError("Arm G vocabulary did not load from the pinned compatibility layout")

    # Freeze Arm G vocabulary lifecycle exactly like the accepted Baseline-v0
    # reproduction so non-combat routing cannot mutate its module-global vocab.
    AG.card_idx = lambda name: AG._vocab.get(name, AG.VOCAB_CAP - 1)
    armg = AG.Scorer((128, 128))
    armg.load_state_dict(torch.load(ARMG_WEIGHT_PATH, weights_only=True, map_location="cpu"))
    armg.eval()
    armb = JialeivArmBPolicy.from_files(vocab_path=VOCAB_PATH, weight_path=ARMB_WEIGHT_PATH)
    return sts, torch, AG, (armg, armb)


def _set_pauses(agent: Any) -> None:
    agent.pause_on_card_reward = True
    agent.pause_on_map = True
    agent.pause_on_rest = True
    agent.pause_on_shop = True
    agent.pause_on_event = True
    agent.pause_on_battle = True


def _armg_step(gc: Any, *, sts: Any, torch: Any, AG: Any, armg: Any) -> bool:
    noncombat = {
        sts.ScreenState.REWARDS,
        sts.ScreenState.MAP_SCREEN,
        sts.ScreenState.REST_ROOM,
        sts.ScreenState.SHOP_ROOM,
        sts.ScreenState.EVENT_SCREEN,
    }
    if gc.screen_state not in noncombat:
        return False
    _kind, descs, execs = AG.build_choices(gc)
    if not descs:
        if gc.screen_state == sts.ScreenState.REWARDS:
            gc.skip_reward_cards()
        return True
    if len(descs) == 1:
        execs[0](gc)
        return True
    with torch.no_grad():
        idx = int(torch.argmax(armg.score(torch.tensor(AG.obs_vec(gc), dtype=torch.float32), descs)).item())
    execs[idx](gc)
    return True


def _public_run_state(gc: Any) -> dict[str, Any]:
    return {
        "floor": int(gc.floor_num),
        "character": "IRONCLAD",
        "ascension_level": 0,
        "room": "COMBAT",
        "screen_type": "NONE",
    }


def _error(
    errors: list[dict[str, Any]],
    *,
    seed: int,
    stage: str,
    floor: int | None = None,
    combat_index: int | None = None,
    decision_index: int | None = None,
    detail: str,
) -> None:
    errors.append(
        {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "seed": seed,
            "floor": floor,
            "combat_index": combat_index,
            "decision_index": decision_index,
            "stage": stage,
            "detail": detail,
        }
    )


def _run_seed(
    seed: int,
    *,
    sts: Any,
    torch: Any,
    AG: Any,
    armg: Any,
    armb: Any,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    adapter = SimulatorCombatAdapter()
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    ag = sts.Agent()
    ag.simulation_count_base = ORACLE_MCTS_SIMS
    _set_pauses(ag)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    counters = {"unresolved": 0, "illegal": 0, "leakage": 0}
    combat_index = 0
    game_steps = 0

    try:
        while gc.outcome == sts.GameOutcome.UNDECIDED and game_steps < MAX_GAME_STEPS:
            game_steps += 1
            ag.playout(gc)
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                break
            if gc.screen_state != sts.ScreenState.BATTLE:
                if not _armg_step(gc, sts=sts, torch=torch, AG=AG, armg=armg):
                    counters["unresolved"] += 1
                    _error(
                        errors,
                        seed=seed,
                        stage="noncombat_screen",
                        floor=int(gc.floor_num),
                        detail=f"unsupported_screen:{gc.screen_state}",
                    )
                    break
                continue

            combat_index += 1
            bc = sts.BattleContext()
            bc.init(gc)
            decision_index = 0
            battle_steps = 0
            abort_battle = False
            while bc.outcome == sts.Outcome.UNDECIDED and battle_steps < MAX_BATTLE_STEPS:
                battle_steps += 1
                native_actions = list(sts.get_legal_actions(bc))
                if not native_actions:
                    counters["unresolved"] += 1
                    _error(
                        errors,
                        seed=seed,
                        stage="legal_actions",
                        floor=int(gc.floor_num),
                        combat_index=combat_index,
                        decision_index=decision_index,
                        detail="no_native_legal_actions_in_undecided_battle",
                    )
                    abort_battle = True
                    break
                if len(native_actions) == 1:
                    native_actions[0].execute(bc)
                    continue

                decision_index += 1
                try:
                    public_state = adapter.adapt(
                        bc,
                        legal_actions=native_actions,
                        run_state=_public_run_state(gc),
                    )
                    context = DecisionContext.from_public_state(public_state)
                    native_by_id = exact_native_action_map(
                        context,
                        public_state["legal_actions"],
                        native_actions,
                    )
                    armb_scores = armb.score_actions(context)
                    expected_ids = {action.action_id for action in context.legal_actions}
                    if set(armb_scores.action_ids) != expected_ids or not armb_scores.tie_action_ids:
                        raise BenchmarkContractError("armb_did_not_score_every_legal_action")

                    oracle_scores: dict[str, float] = {}
                    for action_id in sorted(expected_ids):
                        try:
                            judged = sts.judge_branch_action(bc, native_by_id[action_id], ORACLE_MCTS_SIMS)
                        except Exception as exc:
                            if "judge_first_action_not_legal" in str(exc):
                                counters["illegal"] += 1
                            raise
                        oracle_scores[action_id] = float(judged["score"])
                    if not semantic_score_consistent(context, oracle_scores):
                        raise BenchmarkContractError("semantic_duplicate_oracle_score_drift")
                    oracle_tie_ids = oracle_ties(oracle_scores)
                    random_action = deterministic_random_legal(context, benchmark_seed=0)
                    simple_action = simple_public_heuristic(context)
                    if random_action is None or simple_action is None:
                        raise BenchmarkContractError("baseline_missing_action")
                    trajectory_action_id = min(oracle_tie_ids)
                    row = BenchmarkDecision(
                        seed=seed,
                        floor=int(gc.floor_num),
                        combat_index=combat_index,
                        decision_index=decision_index,
                        decision_signature=context.decision_signature,
                        legal_action_ids=tuple(sorted(expected_ids)),
                        oracle_scores=oracle_scores,
                        oracle_tie_ids=oracle_tie_ids,
                        armb_tie_ids=tuple(sorted(armb_scores.tie_action_ids)),
                        random_action_id=random_action.action_id,
                        simple_action_id=simple_action.action_id,
                        trajectory_action_id=trajectory_action_id,
                    ).to_dict()
                    rows.append(row)
                    native_by_id[trajectory_action_id].execute(bc)
                except PublicStateContractError as exc:
                    if "hidden_information_forbidden" in str(exc):
                        counters["leakage"] += 1
                    counters["unresolved"] += 1
                    _error(
                        errors,
                        seed=seed,
                        stage="public_state_contract",
                        floor=int(gc.floor_num),
                        combat_index=combat_index,
                        decision_index=decision_index,
                        detail=f"{type(exc).__name__}:{exc}",
                    )
                    abort_battle = True
                    break
                except Exception as exc:
                    counters["unresolved"] += 1
                    _error(
                        errors,
                        seed=seed,
                        stage="benchmark_decision",
                        floor=int(gc.floor_num),
                        combat_index=combat_index,
                        decision_index=decision_index,
                        detail=f"{type(exc).__name__}:{exc}",
                    )
                    abort_battle = True
                    break

            if battle_steps >= MAX_BATTLE_STEPS and bc.outcome == sts.Outcome.UNDECIDED:
                counters["unresolved"] += 1
                _error(
                    errors,
                    seed=seed,
                    stage="battle_step_limit",
                    floor=int(gc.floor_num),
                    combat_index=combat_index,
                    decision_index=decision_index,
                    detail=f"battle_steps_reached:{MAX_BATTLE_STEPS}",
                )
                abort_battle = True
            if abort_battle:
                break
            bc.exit_battle(gc)

        if game_steps >= MAX_GAME_STEPS and gc.outcome == sts.GameOutcome.UNDECIDED:
            counters["unresolved"] += 1
            _error(
                errors,
                seed=seed,
                stage="game_step_limit",
                floor=int(gc.floor_num),
                detail=f"game_steps_reached:{MAX_GAME_STEPS}",
            )
    except Exception as exc:
        counters["unresolved"] += 1
        _error(
            errors,
            seed=seed,
            stage="seed_runtime",
            floor=int(getattr(gc, "floor_num", 0)),
            combat_index=combat_index,
            detail=f"{type(exc).__name__}:{exc}",
        )
    return rows, counters, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, choices=(1, 2), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index/count")

    all_seeds = load_heldout_seeds(SEEDS_PATH)
    seeds = tuple(seed for index, seed in enumerate(all_seeds) if index % args.shard_count == args.shard_index)
    sts, torch, AG, models = _load_runtime()
    armg, armb = models
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    totals = {"unresolved": 0, "illegal": 0, "leakage": 0}
    for seed in seeds:
        seed_rows, counters, seed_errors = _run_seed(
            seed,
            sts=sts,
            torch=torch,
            AG=AG,
            armg=armg,
            armb=armb,
        )
        rows.extend(seed_rows)
        errors.extend(seed_errors)
        for key in totals:
            totals[key] += counters[key]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows, seeds=seeds, **totals)
    summary["error_count"] = len(errors)
    provenance = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "repeat": args.repeat,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "agent_sha": EXPECTED_AGENT,
        "simulator_sha": EXPECTED_SIM,
        "heldout_seed_file": str(SEEDS_PATH.relative_to(ROOT)),
        "heldout_seed_file_sha256": sha256_file(SEEDS_PATH),
        "armb_weight_git_blob": JIALEIV_ARM_B_WEIGHT_GIT_BLOB_SHA,
        "armb_vocab_git_blob": JIALEIV_VOCAB_GIT_BLOB_SHA,
        "oracle_mcts_sims": ORACLE_MCTS_SIMS,
        "oracle_role": "benchmark_reference_only",
        "oracle_uses_hidden_rng": True,
        "formal_teacher_uses_oracle": False,
        "trajectory_policy": "lexicographically_smallest_action_id_in_oracle_tie_set",
        "noncombat_policy": "pinned_jialeiv_armg_public_model",
        "armg_runtime_layout": "accepted_baseline_v0_compatibility_layout",
        "github_sha": os.environ.get("GITHUB_SHA"),
    }
    write_ndjson(out / "benchmark_decisions.ndjson", rows)
    write_ndjson(out / "benchmark_errors.ndjson", errors)
    write_json(out / "benchmark_summary.json", summary)
    write_json(out / "provenance.json", provenance)
    print(json.dumps({"summary": summary, "provenance": provenance, "errors": errors[:20]}, indent=2))
    if any(totals.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
