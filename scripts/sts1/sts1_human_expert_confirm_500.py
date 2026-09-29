#!/usr/bin/env python3
"""500-seed independent confirmation for STS1 Human Expert v1 near-miss."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from roguelike_ai.sts1_phase3.armg_strategy_evolve import (
    StrategyGatePolicy,
    evaluate_strategy_gate,
)
from roguelike_ai.sts1_phase3.human_expert import build_card_prior


LOOP_PATH = Path(__file__).with_name("sts1_human_expert_loop.py")
SPEC = importlib.util.spec_from_file_location("sts1_human_expert_loop_confirm", LOOP_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load Human Expert loop helpers")
loop = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = loop
SPEC.loader.exec_module(loop)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--armg-base-weight", type=Path, required=True)
    parser.add_argument("--dev-seed-file", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-strength", type=float, default=2.0)
    parser.add_argument("--seed-count", type=int, default=500)
    parser.add_argument("--combat-mcts-sims", type=int, default=2000)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--rng-seed", type=int, default=2026092910)
    parser.add_argument("--min-ascension", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=2.0)
    args = parser.parse_args()

    if args.seed_count < 1:
        raise RuntimeError("seed-count must be positive")
    if args.candidate_strength <= 0:
        raise RuntimeError("candidate-strength must be positive")

    state = json.loads(args.state_json.read_text(encoding="utf-8"))
    if state.get("schema_version") != "sts1-human-expert-loop-state-v1":
        raise RuntimeError("Human Expert state schema mismatch")

    dev_seeds = loop._read_seeds(args.dev_seed_file)
    forbidden = {int(v) for v in dev_seeds}
    forbidden.update(int(v) for v in state.get("used_evaluation_seeds", []))
    seeds = loop._fresh_seeds(
        count=args.seed_count,
        rng_seed=args.rng_seed,
        forbidden=forbidden,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = args.output_dir / "expert-prior.json"
    dataset = build_card_prior(
        args.data_root,
        prior_path,
        min_ascension=args.min_ascension,
        alpha=args.alpha,
    )
    (args.output_dir / "confirm-seeds-500.txt").write_text(
        "".join(f"{seed}\n" for seed in seeds),
        encoding="utf-8",
    )

    current = loop._evaluate_policy(
        seeds=seeds,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        base_weight=args.armg_base_weight,
        prior_path=prior_path,
        strength=0.0,
        mcts_sims=args.combat_mcts_sims,
        workers=args.eval_workers,
    )
    candidate = loop._evaluate_policy(
        seeds=seeds,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        base_weight=args.armg_base_weight,
        prior_path=prior_path,
        strength=args.candidate_strength,
        mcts_sims=args.combat_mcts_sims,
        workers=args.eval_workers,
    )

    policy = StrategyGatePolicy(
        name=f"human-expert-confirm-{args.seed_count}",
        expected_seed_count=args.seed_count,
        min_win_delta=1,
        min_floor_delta=0.0,
        max_floor_regression_with_win_gain=60.0,
        max_one_sided_sign_p=0.05,
    )
    gate = evaluate_strategy_gate(
        current,
        candidate,
        policy=policy,
        expected_combat_policy=f"mcts_{args.combat_mcts_sims}",
    )
    report = {
        "schema_version": "sts1-human-expert-confirm-500-v1",
        "candidate_strength": args.candidate_strength,
        "seed_count": args.seed_count,
        "seed_rng": args.rng_seed,
        "forbidden_seed_count": len(forbidden),
        "dataset": {
            "accepted_runs": dataset["accepted_runs"],
            "card_choice_examples": dataset["card_choice_examples"],
            "holdout_top1": dataset["holdout_top1"],
            "corpus_sha256": dataset["corpus_sha256"],
        },
        "gate": gate,
        "production_champion_replaced": False,
    }
    (args.output_dir / "confirm-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
