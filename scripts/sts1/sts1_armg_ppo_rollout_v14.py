#!/usr/bin/env python3
"""Collect fresh ArmG PPO v1.4 episodes with bounded per-runner parallelism.

Each game is independent because the ArmG policy is frozen during rollout.
Parallel workers use deterministic per-game Torch seeds, preserve the requested
game-seed order in the merged .npz, and keep combat at MCTS-2000.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, run_simulator_game, _set_pauses


def _v17_terminal_bonus(final_floor: float, *, victory: bool) -> float:
    """Victory-first terminal reward used by PPO v1.7."""
    progress = min(0.50, max(0.0, float(final_floor) / 100.0))
    return progress + (3.0 if victory else -0.25)


class SamplingArmG(ArmGNoncombatPolicy):
    def __init__(self, *args: Any, temperature: float = 1.0, torch_seed: int = 0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.torch.manual_seed(torch_seed)
        # Keep child processes from oversubscribing the hosted runner.
        self.torch.set_num_threads(1)
        self.conversion_states: list[dict[str, Any]] = []

    def capture_conversion_state(
        self,
        *,
        gc: Any,
        sts: Any,
        kind: str,
        selected_index: int,
        descs: list[Any],
        scores: list[float],
    ) -> None:
        """Keep only a tiny in-memory tail of reversible non-combat decisions."""
        if len(descs) < 2 or not callable(getattr(gc, "clone", None)):
            return
        snapshot = self.training_vector_snapshot(gc, descs)
        self.conversion_states.append(
            {
                "gc": gc.clone(),
                "kind": str(kind),
                "selected_index": int(selected_index),
                "scores": [float(v) for v in scores],
                "floor": int(getattr(gc, "floor_num", 0) or 0),
                "act": int(getattr(gc, "act", 0) or 0),
                "hp": int(getattr(gc, "cur_hp", 0) or 0),
                "max_hp": int(getattr(gc, "max_hp", 1) or 1),
                "obs": list(snapshot["obs_412"]),
                "descs": [list(row) for row in snapshot["candidate_desc_368"]],
            }
        )
        # Conversion mining only needs the most recent strategic decisions.
        self.conversion_states = self.conversion_states[-20:]

    def decide(self, gc: Any, sts: Any):
        kind, descs, execs = self.choices(gc)
        if not descs:
            if gc.screen_state == sts.ScreenState.REWARDS:
                return "reward_empty", -1, [], [], []
            raise RuntimeError("no legal ArmG choice")
        if len(descs) == 1:
            return kind, 0, descs, execs, [0.0]
        _, _, scores = self.score_choices(gc)
        raw = [float(x) for x in scores.tolist()]
        probs = self.torch.softmax(scores / self.temperature, 0)
        idx = int(self.torch.multinomial(probs, 1).item())
        return kind, idx, descs, execs, raw


BOSS_FLOORS = (16, 33, 50)
CONVERSION_SCHEMA = "sts1-ppo-v19-multistep-win-conversion-v1"


def _near_boss_failure(final_floor: int, *, tolerance: int = 1) -> bool:
    return any(abs(int(final_floor) - boss) <= tolerance for boss in BOSS_FLOORS)


def _boss_target_floor(final_floor: int) -> int:
    candidates = sorted(BOSS_FLOORS, key=lambda boss: abs(int(final_floor) - boss))
    target = int(candidates[0])
    if abs(int(final_floor) - target) > 1:
        raise RuntimeError(f"floor {final_floor} is not a near-Boss failure")
    return target


def _passed_target_boss(gc: Any, sts: Any, target_floor: int) -> bool:
    if gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
        return True
    return int(getattr(gc, "floor_num", 0) or 0) > int(target_floor)


def _choice_snapshot(policy: SamplingArmG, gc: Any, kind: str, descs: list[Any]) -> dict[str, Any]:
    snap = policy.training_vector_snapshot(gc, descs)
    _, _, scores = policy.score_choices(gc)
    return {
        "kind": str(kind),
        "floor": int(getattr(gc, "floor_num", 0) or 0),
        "act": int(getattr(gc, "act", 0) or 0),
        "obs": list(snap["obs_412"]),
        "descs": [list(row) for row in snap["candidate_desc_368"]],
        "scores": [float(v) for v in scores.tolist()],
    }


def _deterministic_armg_choice(policy: SamplingArmG, gc: Any, sts: Any) -> tuple[str, int, list[Any], list[Any], list[float]]:
    kind, descs, execs = policy.choices(gc)
    if not descs:
        if gc.screen_state == sts.ScreenState.REWARDS:
            return "reward_empty", -1, [], [], []
        if gc.screen_state == sts.ScreenState.MAP_SCREEN:
            legal = list(sts.get_legal_game_actions(gc))
            if len(legal) == 1:
                action = legal[0]
                return (
                    "map_single_legal_fallback",
                    0,
                    [["single_legal_map_transition"]],
                    [lambda context, a=action: a.execute(context)],
                    [0.0],
                )
            if len(legal) == 0:
                return "map_zero_legal_transition", -1, [], [], []
            raise RuntimeError(
                f"conversion replay exposed {len(legal)} legal map actions but ArmG no choice"
            )
        raise RuntimeError(f"conversion replay exposed no ArmG choice: {gc.screen_state}")
    if len(descs) == 1:
        return str(kind), 0, descs, execs, [0.0]
    _, _, scores = policy.score_choices(gc)
    raw = [float(v) for v in scores.tolist()]
    return str(kind), int(policy.torch.argmax(scores).item()), descs, execs, raw


def _advance_zero_legal_map_transition(agent: Any, gc: Any, sts: Any) -> None:
    """Advance a MAP_SCREEN transition that exposes no actual player choice."""
    if gc.screen_state != sts.ScreenState.MAP_SCREEN:
        raise RuntimeError(f"zero-legal transition is only allowed on MAP_SCREEN: {gc.screen_state}")
    legal = list(sts.get_legal_game_actions(gc))
    if legal:
        raise RuntimeError("refusing to auto-advance a map state that has legal player actions")
    before = (
        int(getattr(gc, "floor_num", 0) or 0),
        int(getattr(gc, "act", 0) or 0),
        gc.screen_state,
        gc.outcome,
    )
    agent.pause_on_map = False
    try:
        agent.playout(gc)
    finally:
        agent.pause_on_map = True
    after = (
        int(getattr(gc, "floor_num", 0) or 0),
        int(getattr(gc, "act", 0) or 0),
        gc.screen_state,
        gc.outcome,
    )
    if after == before:
        raise RuntimeError("zero-legal map transition made no progress")


def _finish_conversion_branch(
    gc: Any,
    *,
    sts: Any,
    policy: SamplingArmG,
    mcts_sims: int,
    target_floor: int,
    boss_mcts_sims: int | None = None,
    second_alternative_rank: int | None = None,
    max_game_steps: int = 600,
    max_battle_steps: int = 800,
) -> dict[str, Any]:
    """Finish from a cloned state; optionally alter the next multi-choice decision."""
    agent = sts.Agent()
    _set_pauses(agent)
    game_steps = 0
    second_intervention = None

    while gc.outcome == sts.GameOutcome.UNDECIDED and game_steps < max_game_steps:
        if _passed_target_boss(gc, sts, target_floor):
            break
        game_steps += 1
        agent.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED or _passed_target_boss(gc, sts, target_floor):
            break

        if gc.screen_state == sts.ScreenState.BATTLE:
            battle = sts.BattleContext()
            battle.init(gc)
            battle_steps = 0
            while battle.outcome == sts.Outcome.UNDECIDED and battle_steps < max_battle_steps:
                battle_steps += 1
                legal = list(sts.get_legal_actions(battle))
                if not legal:
                    raise RuntimeError("conversion battle exposed no legal action")
                floor_now = int(getattr(gc, "floor_num", 0) or 0)
                active_sims = (
                    int(boss_mcts_sims)
                    if boss_mcts_sims is not None and floor_now == int(target_floor)
                    else int(mcts_sims)
                )
                chosen = legal[0] if len(legal) == 1 else sts.mcts_recommend(battle, active_sims)
                if chosen is None:
                    raise RuntimeError("conversion MCTS returned no action")
                chosen.execute(battle)
            if battle.outcome == sts.Outcome.UNDECIDED:
                raise RuntimeError("conversion battle step bound reached")
            battle.exit_battle(gc)
            continue

        kind, current, descs, execs, scores = _deterministic_armg_choice(policy, gc, sts)
        if current < 0:
            if gc.screen_state == sts.ScreenState.REWARDS:
                gc.skip_reward_cards()
            elif gc.screen_state == sts.ScreenState.MAP_SCREEN:
                _advance_zero_legal_map_transition(agent, gc, sts)
            else:
                raise RuntimeError(f"conversion replay cannot advance empty choice: {gc.screen_state}")
            continue

        if second_alternative_rank is not None and second_intervention is None and len(descs) > 1:
            alternatives = [i for i in range(len(descs)) if i != current]
            alternatives.sort(
                key=lambda i: float(scores[i]) if i < len(scores) else -1e30,
                reverse=True,
            )
            if alternatives:
                rank = min(int(second_alternative_rank), len(alternatives) - 1)
                forced = int(alternatives[rank])
                snap = _choice_snapshot(policy, gc, kind, descs)
                target = [0.0] * len(descs)
                target[forced] = 1.0
                second_intervention = {
                    **snap,
                    "current_armg_index": int(current),
                    "teacher_best_index": forced,
                    "target_probs": target,
                }
                execs[forced](gc)
                continue

        execs[current](gc)

    passed = _passed_target_boss(gc, sts, target_floor)
    if not passed and gc.outcome == sts.GameOutcome.UNDECIDED and game_steps >= max_game_steps:
        raise RuntimeError("conversion game step bound reached")
    return {
        "victory": bool(gc.outcome == sts.GameOutcome.PLAYER_VICTORY),
        "passed_boss": bool(passed),
        "target_boss_floor": int(target_floor),
        "final_floor": int(getattr(gc, "floor_num", 0) or 0),
        "second_intervention": second_intervention,
    }


def _force_choice_and_finish(
    record: dict[str, Any],
    *,
    choice_index: int,
    sts: Any,
    policy: SamplingArmG,
    mcts_sims: int,
    target_floor: int,
    boss_mcts_sims: int | None = None,
    second_alternative_rank: int | None = None,
) -> dict[str, Any]:
    branch = record["gc"].clone()
    kind, descs, execs = policy.choices(branch)
    desc_rows = [[float(v) for v in row] for row in descs]
    if str(kind) != str(record["kind"]) or desc_rows != record["descs"]:
        raise RuntimeError("conversion clone choice identity drift")
    if not 0 <= int(choice_index) < len(execs):
        raise RuntimeError("conversion forced choice outside legal range")
    execs[int(choice_index)](branch)
    return _finish_conversion_branch(
        branch,
        sts=sts,
        policy=policy,
        mcts_sims=mcts_sims,
        target_floor=target_floor,
        boss_mcts_sims=boss_mcts_sims,
        second_alternative_rank=second_alternative_rank,
    )


def _conversion_row(
    *,
    record: dict[str, Any],
    selected: int,
    alternative: int,
    seed: int | None,
    conversion_type: str,
    priority: float,
    baseline_2k: dict[str, Any],
    baseline_10k: dict[str, Any],
    alt_2k: dict[str, Any],
    alt_10k: dict[str, Any],
    primary_mcts: int,
    confirm_mcts: int,
) -> dict[str, Any]:
    target = [0.0] * len(record["descs"])
    target[int(alternative)] = 1.0
    return {
        "schema_version": "sts1-armg-strategy-branch-dataset-v1",
        "type": conversion_type,
        "source": CONVERSION_SCHEMA,
        "seed": seed,
        "floor": int(record["floor"]),
        "act": int(record["act"]),
        "kind": str(record["kind"]),
        "obs": record["obs"],
        "descs": record["descs"],
        "current_armg_index": int(selected),
        "teacher_best_index": int(alternative),
        "target_probs": target,
        "priority": float(priority),
        "teacher_margin": 12.0,
        "confidence_weight": 1.0,
        "teacher_consensus_fraction": 1.0,
        "combat_policy": f"mcts_{int(primary_mcts)}",
        "confirmation_policy": f"mcts_{int(confirm_mcts)}",
        "baseline_confirmed_loss": {
            f"mcts_{int(primary_mcts)}": baseline_2k,
            f"mcts_{int(confirm_mcts)}": baseline_10k,
        },
        "alternative_confirmed_progress": {
            f"mcts_{int(primary_mcts)}": alt_2k,
            f"mcts_{int(confirm_mcts)}": alt_10k,
        },
    }


def _second_conversion_row(
    *,
    intervention: dict[str, Any],
    seed: int | None,
    target_floor: int,
    primary_mcts: int,
    confirm_mcts: int,
) -> dict[str, Any]:
    return {
        "schema_version": "sts1-armg-strategy-branch-dataset-v1",
        "type": "critical_two_step_boss_conversion_followup",
        "source": CONVERSION_SCHEMA,
        "seed": seed,
        "floor": int(intervention["floor"]),
        "act": int(intervention["act"]),
        "kind": str(intervention["kind"]),
        "obs": intervention["obs"],
        "descs": intervention["descs"],
        "current_armg_index": int(intervention["current_armg_index"]),
        "teacher_best_index": int(intervention["teacher_best_index"]),
        "target_probs": intervention["target_probs"],
        "priority": 4.5,
        "teacher_margin": 10.0,
        "confidence_weight": 1.0,
        "teacher_consensus_fraction": 1.0,
        "combat_policy": f"mcts_{int(primary_mcts)}",
        "confirmation_policy": f"mcts_{int(confirm_mcts)}",
        "target_boss_floor": int(target_floor),
    }


def _mine_win_conversions(
    *,
    policy: SamplingArmG,
    sts: Any,
    final_floor: int,
    seed: int | None = None,
    max_states: int = 6,
    max_alternatives: int = 2,
    max_second_alternatives: int = 2,
    primary_mcts: int = 2000,
    confirm_mcts: int = 10000,
) -> list[dict[str, Any]]:
    """Mine one- and two-step changes that reliably convert a Boss loss into progress."""
    if not _near_boss_failure(final_floor):
        return []

    target_floor = _boss_target_floor(final_floor)
    checked_states = 0

    for record in reversed(policy.conversion_states):
        if checked_states >= max_states:
            break
        if len(record["descs"]) < 2:
            continue
        checked_states += 1
        selected = int(record["selected_index"])

        baseline_2k = _force_choice_and_finish(
            record,
            choice_index=selected,
            sts=sts,
            policy=policy,
            mcts_sims=int(primary_mcts),
            target_floor=target_floor,
        )
        if baseline_2k["passed_boss"]:
            continue
        baseline_10k = _force_choice_and_finish(
            record,
            choice_index=selected,
            sts=sts,
            policy=policy,
            mcts_sims=int(confirm_mcts),
            target_floor=target_floor,
        )
        if baseline_10k["passed_boss"]:
            continue

        alternatives = [i for i in range(len(record["descs"])) if i != selected]
        alternatives.sort(
            key=lambda i: float(record["scores"][i]) if i < len(record["scores"]) else -1e30,
            reverse=True,
        )

        for alternative in alternatives[:max_alternatives]:
            alt_2k = _force_choice_and_finish(
                record,
                choice_index=alternative,
                sts=sts,
                policy=policy,
                mcts_sims=int(primary_mcts),
                target_floor=target_floor,
            )
            alt_10k = _force_choice_and_finish(
                record,
                choice_index=alternative,
                sts=sts,
                policy=policy,
                mcts_sims=int(confirm_mcts),
                target_floor=target_floor,
            )
            if alt_2k["passed_boss"] and alt_10k["passed_boss"]:
                priority = 5.0 if alt_2k["victory"] and alt_10k["victory"] else 4.0
                ctype = (
                    "critical_full_win_conversion"
                    if priority == 5.0
                    else "critical_boss_pass_conversion"
                )
                return [
                    _conversion_row(
                        record=record,
                        selected=selected,
                        alternative=alternative,
                        seed=seed,
                        conversion_type=ctype,
                        priority=priority,
                        baseline_2k=baseline_2k,
                        baseline_10k=baseline_10k,
                        alt_2k=alt_2k,
                        alt_10k=alt_10k,
                        primary_mcts=primary_mcts,
                        confirm_mcts=confirm_mcts,
                    )
                ]

            # If one change is insufficient, alter the next strategic choice too.
            for second_rank in range(max_second_alternatives):
                two_2k = _force_choice_and_finish(
                    record,
                    choice_index=alternative,
                    sts=sts,
                    policy=policy,
                    mcts_sims=int(primary_mcts),
                    target_floor=target_floor,
                    second_alternative_rank=second_rank,
                )
                if not two_2k["passed_boss"]:
                    continue
                two_10k = _force_choice_and_finish(
                    record,
                    choice_index=alternative,
                    sts=sts,
                    policy=policy,
                    mcts_sims=int(confirm_mcts),
                    target_floor=target_floor,
                    second_alternative_rank=second_rank,
                )
                if not two_10k["passed_boss"]:
                    continue

                first = _conversion_row(
                    record=record,
                    selected=selected,
                    alternative=alternative,
                    seed=seed,
                    conversion_type="critical_two_step_boss_conversion",
                    priority=4.5,
                    baseline_2k=baseline_2k,
                    baseline_10k=baseline_10k,
                    alt_2k=two_2k,
                    alt_10k=two_10k,
                    primary_mcts=primary_mcts,
                    confirm_mcts=confirm_mcts,
                )
                second = two_10k.get("second_intervention") or two_2k.get("second_intervention")
                rows = [first]
                if second is not None:
                    rows.append(
                        _second_conversion_row(
                            intervention=second,
                            seed=seed,
                            target_floor=target_floor,
                            primary_mcts=primary_mcts,
                            confirm_mcts=confirm_mcts,
                        )
                    )
                return rows

    return []

def _collect_one(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        module_dir,
        armg_root,
        weight,
        seed,
        temperature,
        checkpoint_id,
        worker,
        torch_seed,
        reward_mode,
        conversion_enabled,
        conversion_max_states,
        conversion_max_alternatives,
        conversion_max_second_alternatives,
        conversion_sample_modulus,
        conversion_primary_mcts,
        conversion_confirm_mcts,
    ) = task

    module_path = Path(module_dir)
    sys.path.insert(0, str(module_path))
    sts = importlib.import_module("slaythespire")
    out = Path(armg_root).parent / "rollout-games" / f"worker-{worker}" / f"seed-{seed}"
    out.mkdir(parents=True, exist_ok=True)

    policy = SamplingArmG(
        root=Path(armg_root),
        weight_path=Path(weight),
        temperature=temperature,
        torch_seed=torch_seed,
    )
    evidence = out / "evidence.ndjson"
    result = run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=evidence,
        armg_policy=policy,
        combat_mcts_sims=2000,
        training_seeds=[seed],
    )
    if result.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
        return {
            "accepted": False,
            "seed": seed,
            "checkpoint_id": checkpoint_id,
            "reward_mode": reward_mode,
            "behavior_temperature": float(temperature),
            "rejection": {
                "result": result.get("result"),
                "error": result.get("error"),
                "final_floor": result.get("final_floor"),
                "illegal_action_count": int(result.get("illegal_action_count", 0) or 0),
                "timeout_count": int(result.get("timeout_count", 0) or 0),
                "crash_count": int(result.get("crash_count", 0) or 0),
                "remote_error_count": int(result.get("remote_error_count", 0) or 0),
            },
        }

    events = [json.loads(line) for line in evidence.read_text().splitlines() if line.strip()]
    decisions = [
        row
        for row in events
        if row.get("type") == "armg_noncombat_decision_v3"
        and row.get("selected_index", -1) >= 0
        and len(row.get("candidate_desc_368", [])) > 1
    ]
    victory = str(result.get("outcome", "")).lower() == "victory"
    rows = []
    for index, row in enumerate(decisions):
        scores = np.asarray(row["choice_scores"], dtype=np.float64) / temperature
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()

        floor = int(row.get("floor") or 0)
        next_floor = (
            int(decisions[index + 1].get("floor") or floor)
            if index + 1 < len(decisions)
            else int(result.get("max_floor") or result.get("final_floor") or floor)
        )
        hp = float(row.get("hp_before") or 0)
        next_hp = (
            float(decisions[index + 1].get("hp_before") or hp)
            if index + 1 < len(decisions)
            else hp
        )
        if reward_mode == "legacy":
            reward = (
                0.10 * max(0, next_floor - floor)
                + 0.01 * (next_hp - hp)
                + (
                    10.0
                    if victory and index == len(decisions) - 1
                    else -2.0
                    if (not victory and index == len(decisions) - 1)
                    else 0.0
                )
            )
        elif reward_mode == "v14_dense":
            # Keep dense shaping small and put most credit on final run quality.
            # This aligns the scale with floor/win evaluation instead of allowing
            # HP fluctuations to dominate the objective.
            reward = (
                0.02 * max(0, next_floor - floor)
                + 0.002 * (next_hp - hp)
            )
            if index == len(decisions) - 1:
                final_floor = float(
                    result.get("final_floor")
                    or result.get("max_floor")
                    or next_floor
                )
                reward += min(1.0, max(0.0, final_floor / 50.0))
                if victory:
                    reward += 1.0
        elif reward_mode == "v17_winrate":
            # Victory-first shaping. Floors/HP still provide a weak learning
            # signal, but a full clear is deliberately worth far more than
            # merely reaching a late boss.
            reward = (
                0.01 * max(0, next_floor - floor)
                + 0.001 * (next_hp - hp)
            )
            if index == len(decisions) - 1:
                final_floor = float(
                    result.get("final_floor")
                    or result.get("max_floor")
                    or next_floor
                )
                reward += _v17_terminal_bonus(final_floor, victory=victory)
        else:
            raise RuntimeError(f"unknown reward mode: {reward_mode}")
        action = int(row["selected_index"])
        rows.append(
            {
                "obs": row["obs_412"],
                "desc": row["candidate_desc_368"],
                "action": action,
                "reward": float(reward),
                "old_logp": float(math.log(max(1e-12, probs[action]))),
                "done": index == len(decisions) - 1,
            }
        )

    final_floor = int(
        result.get("final_floor")
        or result.get("max_floor")
        or 0
    )
    win_conversions: list[dict[str, Any]] = []
    conversion_mining_attempted = bool(
        conversion_enabled
        and (not victory)
        and _near_boss_failure(final_floor)
        and int(seed) % int(conversion_sample_modulus) == 0
    )
    if conversion_mining_attempted:
        win_conversions = _mine_win_conversions(
            policy=policy,
            sts=sts,
            final_floor=final_floor,
            seed=int(seed),
            max_states=int(conversion_max_states),
            max_alternatives=int(conversion_max_alternatives),
            max_second_alternatives=int(conversion_max_second_alternatives),
            primary_mcts=int(conversion_primary_mcts),
            confirm_mcts=int(conversion_confirm_mcts),
        )
        for row in win_conversions:
            row["seed"] = int(seed)

    return {
        "accepted": True,
        "seed": seed,
        "victory": victory,
        "final_floor": final_floor,
        "win_conversions": win_conversions,
        "conversion_mining_attempted": conversion_mining_attempted,
        "rows": rows,
        "checkpoint_id": checkpoint_id,
        "reward_mode": reward_mode,
        "behavior_temperature": float(temperature),
    }


def _next_training_seed(rng: random.Random, seen: set[int]) -> int:
    """Draw one deterministic training-only seed not used by any held-out set."""
    while True:
        seed = rng.randrange(1, 2**31 - 1)
        if seed not in seen:
            seen.add(seed)
            return seed


def _fill_training_games(
    *,
    target_games: int,
    max_rejections: int,
    draw_seed: Any,
    run_batch: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect exactly target_games complete training episodes with bounded refill."""
    games: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    attempt_index = 0
    while len(games) < target_games:
        needed = target_games - len(games)
        seeds = [int(draw_seed()) for _ in range(needed)]
        batch = list(run_batch(seeds, attempt_index))
        if len(batch) != len(seeds):
            raise RuntimeError("training batch result cardinality mismatch")
        attempt_index += len(seeds)
        for game in batch:
            if bool(game.get("accepted", False)):
                games.append(game)
                continue
            rejection = {
                "seed": int(game["seed"]),
                **dict(game.get("rejection") or {}),
            }
            rejected.append(rejection)
            print(
                "PPO_V15_TRAINING_SEED_REJECT",
                json.dumps(rejection, sort_keys=True),
                flush=True,
            )
        if len(rejected) > max_rejections:
            raise RuntimeError(
                "training seed rejection budget exceeded: "
                f"{len(rejected)} > {max_rejections}; "
                f"rejections={rejected}"
            )
    if len(games) != target_games:
        raise RuntimeError(
            f"training rollout completeness mismatch: {len(games)} != {target_games}"
        )
    return games, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("module-dir", "armg-root", "weight", "out", "formal-seed-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--parallel-games", type=int, default=2)
    parser.add_argument("--max-training-seed-rejections", type=int, default=8)
    parser.add_argument("--reward-mode", choices=("legacy", "v14_dense", "v17_winrate"), default="v14_dense")
    parser.add_argument("--final-seed-file", type=Path)
    parser.add_argument("--extra-heldout-seed-file", type=Path)
    parser.add_argument("--win-conversion-enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument("--win-conversion-max-states", type=int, default=6)
    parser.add_argument("--win-conversion-max-alternatives", type=int, default=2)
    parser.add_argument("--win-conversion-max-second-alternatives", type=int, default=2)
    parser.add_argument("--win-conversion-sample-modulus", type=int, default=4)
    parser.add_argument("--win-conversion-primary-mcts", type=int, default=2000)
    parser.add_argument("--win-conversion-confirm-mcts", type=int, default=10000)
    args = parser.parse_args()

    if args.parallel_games < 1 or args.parallel_games > 4:
        raise RuntimeError("parallel-games must be within 1..4")
    if args.max_training_seed_rejections < 0 or args.max_training_seed_rejections > 32:
        raise RuntimeError("max-training-seed-rejections must be within 0..32")
    if not 1 <= args.win_conversion_max_states <= 8:
        raise RuntimeError("win-conversion-max-states must be within 1..8")
    if not 1 <= args.win_conversion_max_alternatives <= 4:
        raise RuntimeError("win-conversion-max-alternatives must be within 1..4")
    if not 1 <= args.win_conversion_max_second_alternatives <= 3:
        raise RuntimeError("win-conversion-max-second-alternatives must be within 1..3")
    if not 1 <= args.win_conversion_sample_modulus <= 16:
        raise RuntimeError("win-conversion-sample-modulus must be within 1..16")
    if args.win_conversion_primary_mcts < 1:
        raise RuntimeError("win-conversion-primary-mcts must be positive")
    if args.win_conversion_confirm_mcts < args.win_conversion_primary_mcts:
        raise RuntimeError("win-conversion-confirm-mcts must be >= primary")

    formal = {
        int(x)
        for x in args.formal_seed_file.read_text().splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    }
    final = set()
    if args.final_seed_file is not None:
        final = {
            int(x)
            for x in args.final_seed_file.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        }
    if formal & final:
        raise RuntimeError("dev/final seed sets must be disjoint")
    extra_heldout = set()
    if args.extra_heldout_seed_file is not None:
        extra_heldout = {
            int(x)
            for x in args.extra_heldout_seed_file.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        }
        if not extra_heldout:
            raise RuntimeError("extra held-out seed file is empty")
        if formal & extra_heldout or final & extra_heldout:
            raise RuntimeError("extra held-out seeds must be disjoint from dev/final")
    rng = random.Random(args.seed_start)
    seen = set(formal) | set(final) | set(extra_heldout)
    with ProcessPoolExecutor(max_workers=min(args.parallel_games, args.games)) as pool:
        def draw_seed() -> int:
            return _next_training_seed(rng, seen)

        def run_batch(seeds: list[int], attempt_index: int) -> list[dict[str, Any]]:
            tasks = [
                (
                    str(args.module_dir),
                    str(args.armg_root),
                    str(args.weight),
                    seed,
                    float(args.temperature),
                    args.checkpoint_id,
                    int(args.worker),
                    int(
                        args.seed_start
                        + args.worker * 1000003
                        + attempt_index
                        + index
                    ),
                    args.reward_mode,
                    bool(args.win_conversion_enabled),
                    int(args.win_conversion_max_states),
                    int(args.win_conversion_max_alternatives),
                    int(args.win_conversion_max_second_alternatives),
                    int(args.win_conversion_sample_modulus),
                    int(args.win_conversion_primary_mcts),
                    int(args.win_conversion_confirm_mcts),
                )
                for index, seed in enumerate(seeds)
            ]
            return list(pool.map(_collect_one, tasks))

        games, rejected = _fill_training_games(
            target_games=args.games,
            max_rejections=args.max_training_seed_rejections,
            draw_seed=draw_seed,
            run_batch=run_batch,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    wins = 0
    for game in games:
        if game["checkpoint_id"] != args.checkpoint_id:
            raise RuntimeError("parallel rollout checkpoint identity drift")
        wins += int(game["victory"])
        for row in game["rows"]:
            rows.append((game["seed"], row))

    if not rows:
        raise RuntimeError("parallel rollout produced no PPO decisions")

    conversion_rows = [
        row
        for game in games
        for row in game.get("win_conversions", [])
    ]
    conversion_path = args.out / "win-conversion.jsonl"
    conversion_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in conversion_rows),
        encoding="utf-8",
    )

    game_by_seed = {int(game["seed"]): game for game in games}
    np.savez_compressed(
        args.out / "rollout.npz",
        obs=np.array([row["obs"] for _, row in rows], np.float32),
        desc=np.array([row["desc"] for _, row in rows], dtype=object),
        action=np.array([row["action"] for _, row in rows]),
        reward=np.array([row["reward"] for _, row in rows], np.float32),
        old_logp=np.array([row["old_logp"] for _, row in rows], np.float32),
        done=np.array([row["done"] for _, row in rows], np.bool_),
        checkpoint_id=np.array([args.checkpoint_id] * len(rows)),
        game_seed=np.array([seed for seed, _ in rows], np.int64),
        behavior_temperature=np.array(
            [game_by_seed[int(seed)]["behavior_temperature"] for seed, _ in rows],
            np.float32,
        ),
        game_victory=np.array(
            [game_by_seed[int(seed)]["victory"] for seed, _ in rows],
            np.bool_,
        ),
        game_final_floor=np.array(
            [game_by_seed[int(seed)]["final_floor"] for seed, _ in rows],
            np.int16,
        ),
    )
    print(
        "ARMG_PPO_V11_ROLLOUT",
        json.dumps(
            {
                "games": len(games),
                "rejected_training_seeds": len(rejected),
                "rejected_seed_ids": [row["seed"] for row in rejected],
                "wins": wins,
                "decisions": len(rows),
                "checkpoint_id": args.checkpoint_id,
                "parallel_games": args.parallel_games,
                "reward_mode": args.reward_mode,
                "near_boss_losses": sum(
                    (not bool(game["victory"])) and _near_boss_failure(int(game["final_floor"]))
                    for game in games
                ),
                "conversion_mining_attempts": sum(
                    bool(game.get("conversion_mining_attempted", False))
                    for game in games
                ),
                "win_conversion_examples": len(conversion_rows),
                "behavior_temperature": args.temperature,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
