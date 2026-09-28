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

from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, run_simulator_game


class SamplingArmG(ArmGNoncombatPolicy):
    def __init__(self, *args: Any, temperature: float = 1.0, torch_seed: int = 0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.torch.manual_seed(torch_seed)
        # Keep child processes from oversubscribing the hosted runner.
        self.torch.set_num_threads(1)

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
        raise RuntimeError(f"rollout seed {seed} incomplete: {result}")

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

    return {
        "seed": seed,
        "victory": victory,
        "rows": rows,
        "checkpoint_id": checkpoint_id,
        "reward_mode": reward_mode,
    }


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
    parser.add_argument("--reward-mode", choices=("legacy", "v14_dense"), default="v14_dense")
    args = parser.parse_args()

    if args.parallel_games < 1 or args.parallel_games > 4:
        raise RuntimeError("parallel-games must be within 1..4")

    formal = {
        int(x)
        for x in args.formal_seed_file.read_text().splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    }
    rng = random.Random(args.seed_start)
    seeds: list[int] = []
    seen = set(formal)
    while len(seeds) < args.games:
        seed = rng.randrange(1, 2**31 - 1)
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)

    tasks = [
        (
            str(args.module_dir),
            str(args.armg_root),
            str(args.weight),
            seed,
            float(args.temperature),
            args.checkpoint_id,
            int(args.worker),
            int(args.seed_start + args.worker * 1000003 + index),
            args.reward_mode,
        )
        for index, seed in enumerate(seeds)
    ]

    with ProcessPoolExecutor(max_workers=min(args.parallel_games, len(tasks))) as pool:
        games = list(pool.map(_collect_one, tasks))

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

    np.savez_compressed(
        args.out / "rollout.npz",
        obs=np.array([row["obs"] for _, row in rows], np.float32),
        desc=np.array([row["desc"] for _, row in rows], dtype=object),
        action=np.array([row["action"] for _, row in rows]),
        reward=np.array([row["reward"] for _, row in rows], np.float32),
        old_logp=np.array([row["old_logp"] for _, row in rows], np.float32),
        done=np.array([row["done"] for _, row in rows], np.bool_),
        checkpoint_id=np.array([args.checkpoint_id] * len(rows)),
        game_seed=np.array([seed for seed, _ in rows]),
    )
    print(
        "ARMG_PPO_V11_ROLLOUT",
        json.dumps(
            {
                "games": len(seeds),
                "wins": wins,
                "decisions": len(rows),
                "checkpoint_id": args.checkpoint_id,
                "parallel_games": args.parallel_games,
                "reward_mode": args.reward_mode,
                "behavior_temperature": args.temperature,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
