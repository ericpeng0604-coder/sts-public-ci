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
        self.conversion_states = self.conversion_states[-6:]

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
CONVERSION_SCHEMA = "sts1-ppo-v18-win-conversion-v1"


def _near_boss_failure(final_floor: int, *, tolerance: int = 1) -> bool:
    return any(abs(int(final_floor) - boss) <= tolerance for boss in BOSS_FLOORS)


def _deterministic_armg_step(policy: SamplingArmG, gc: Any, sts: Any) -> None:
    kind, descs, execs = policy.choices(gc)
    if not descs:
        if gc.screen_state == sts.ScreenState.REWARDS:
            gc.skip_reward_cards()
            return
        raise RuntimeError(f"conversion replay exposed no ArmG choice: {gc.screen_state}")
    if len(descs) == 1:
        index = 0
    else:
        _, _, scores = policy.score_choices(gc)
        index = int(policy.torch.argmax(scores).item())
    execs[index](gc)


def _finish_conversion_branch(
    gc: Any,
    *,
    sts: Any,
    policy: SamplingArmG,
    mcts_sims: int,
    max_game_steps: int = 600,
    max_battle_steps: int = 800,
) -> dict[str, Any]:
    agent = sts.Agent()
    _set_pauses(agent)
    game_steps = 0
    while gc.outcome == sts.GameOutcome.UNDECIDED and game_steps < max_game_steps:
        game_steps += 1
        agent.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
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
                chosen = legal[0] if len(legal) == 1 else sts.mcts_recommend(battle, int(mcts_sims))
                if chosen is None:
                    raise RuntimeError("conversion MCTS returned no action")
                chosen.execute(battle)
            if battle.outcome == sts.Outcome.UNDECIDED:
                raise RuntimeError("conversion battle step bound reached")
            battle.exit_battle(gc)
        else:
            _deterministic_armg_step(policy, gc, sts)

    if gc.outcome == sts.GameOutcome.UNDECIDED:
        raise RuntimeError("conversion game step bound reached")
    return {
        "victory": bool(gc.outcome == sts.GameOutcome.PLAYER_VICTORY),
        "final_floor": int(getattr(gc, "floor_num", 0) or 0),
    }


def _force_choice_and_finish(
    record: dict[str, Any],
    *,
    choice_index: int,
    sts: Any,
    policy: SamplingArmG,
    mcts_sims: int,
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
    )


def _mine_win_conversions(
    *,
    policy: SamplingArmG,
    sts: Any,
    final_floor: int,
    max_states: int = 2,
    max_alternatives: int = 3,
) -> list[dict[str, Any]]:
    """Find high-confidence one-decision changes that flip a near-Boss loss to wins."""
    if not _near_boss_failure(final_floor):
        return []

    mined: list[dict[str, Any]] = []
    checked_states = 0
    for record in reversed(policy.conversion_states):
        if checked_states >= max_states:
            break
        if len(record["descs"]) < 2:
            continue
        checked_states += 1

        selected = int(record["selected_index"])
        baseline_2k = _force_choice_and_finish(
            record, choice_index=selected, sts=sts, policy=policy, mcts_sims=2000
        )
        if baseline_2k["victory"]:
            continue
        baseline_10k = _force_choice_and_finish(
            record, choice_index=selected, sts=sts, policy=policy, mcts_sims=10000
        )
        if baseline_10k["victory"]:
            continue

        alternatives = [i for i in range(len(record["descs"])) if i != selected]
        alternatives.sort(
            key=lambda i: float(record["scores"][i]) if i < len(record["scores"]) else -1e30,
            reverse=True,
        )
        for alternative in alternatives[:max_alternatives]:
            alt_2k = _force_choice_and_finish(
                record, choice_index=alternative, sts=sts, policy=policy, mcts_sims=2000
            )
            if not alt_2k["victory"]:
                continue
            alt_10k = _force_choice_and_finish(
                record, choice_index=alternative, sts=sts, policy=policy, mcts_sims=10000
            )
            if not alt_10k["victory"]:
                continue

            target = [0.0] * len(record["descs"])
            target[alternative] = 1.0
            mined.append(
                {
                    "schema_version": "sts1-armg-strategy-branch-dataset-v1",
                    "type": "critical_win_conversion",
                    "source": CONVERSION_SCHEMA,
                    "seed": None,
                    "floor": int(record["floor"]),
                    "act": int(record["act"]),
                    "kind": str(record["kind"]),
                    "obs": record["obs"],
                    "descs": record["descs"],
                    "current_armg_index": selected,
                    "teacher_best_index": int(alternative),
                    "target_probs": target,
                    "priority": 5.0,
                    "teacher_margin": 12.0,
                    "confidence_weight": 1.0,
                    "teacher_consensus_fraction": 1.0,
                    "combat_policy": "mcts_2000",
                    "confirmation_policy": "mcts_10000",
                    "baseline_confirmed_loss": {
                        "mcts_2000": baseline_2k,
                        "mcts_10000": baseline_10k,
                    },
                    "alternative_confirmed_win": {
                        "mcts_2000": alt_2k,
                        "mcts_10000": alt_10k,
                    },
                }
            )
            # One verified flip per original game is enough; avoid runaway MCTS cost.
            return mined
    return mined


def _collect_one(task: tuple[str, str, str, int, float, str, int, int, str]) -> dict[str, Any]:
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
    if (not victory) and _near_boss_failure(final_floor):
        win_conversions = _mine_win_conversions(
            policy=policy,
            sts=sts,
            final_floor=final_floor,
        )
        for row in win_conversions:
            row["seed"] = int(seed)

    return {
        "accepted": True,
        "seed": seed,
        "victory": victory,
        "final_floor": final_floor,
        "win_conversions": win_conversions,
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
    args = parser.parse_args()

    if args.parallel_games < 1 or args.parallel_games > 4:
        raise RuntimeError("parallel-games must be within 1..4")
    if args.max_training_seed_rejections < 0 or args.max_training_seed_rejections > 32:
        raise RuntimeError("max-training-seed-rejections must be within 0..32")

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
                "win_conversion_examples": len(conversion_rows),
                "behavior_temperature": args.temperature,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
