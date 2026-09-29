#!/usr/bin/env python3
"""STS1 Human Expert v1 fixed-500 confirmation.

Runs five independent 100-seed shards over one frozen, disjoint seed set.
No training occurs here.  Parent (strength=0) and Human Expert v1
(strength=2.0 by default) use identical seeds and frozen MCTS combat.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping, Sequence

from roguelike_ai.sts1_phase3.armg_strategy_evolve import (
    StrategyGatePolicy,
    evaluate_strategy_gate,
)
from roguelike_ai.sts1_phase3.human_expert import (
    HumanExpertPolicy,
    build_card_prior,
)
from roguelike_ai.sts1_phase3.simulator import _load_sts, run_simulator_game


SCHEMA_VERSION = "sts1-human-expert-confirm-500-v1"
SEED_BASE = 1_500_000_000
SEED_COUNT = 500
SHARD_COUNT = 5
SHARD_SIZE = 100


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fixed_seeds() -> tuple[int, ...]:
    return tuple(SEED_BASE + index for index in range(SEED_COUNT))


def shard_seeds(shard_index: int) -> tuple[int, ...]:
    if not 0 <= int(shard_index) < SHARD_COUNT:
        raise ValueError(f"shard index must be 0..{SHARD_COUNT - 1}")
    start = int(shard_index) * SHARD_SIZE
    return fixed_seeds()[start : start + SHARD_SIZE]


def _worker(
    payload: tuple[str, str, str, str, float, int, tuple[int, ...], int],
) -> dict[str, Any]:
    (
        module_dir_text,
        armg_root_text,
        base_weight_text,
        prior_path_text,
        strength,
        seed,
        all_seeds,
        mcts_sims,
    ) = payload
    sts = _load_sts(Path(module_dir_text))
    policy = HumanExpertPolicy(
        root=Path(armg_root_text),
        weight_path=Path(base_weight_text),
        expert_prior_path=Path(prior_path_text),
        expert_strength=float(strength),
    )
    return dict(
        run_simulator_game(
            student=None,
            sts=sts,
            seed=int(seed),
            armg_policy=policy,
            combat_mcts_sims=int(mcts_sims),
            heldout_seeds=all_seeds,
            collect_ppo=False,
            collect_teacher=False,
        )
    )


def _evaluate(
    *,
    seeds: Sequence[int],
    all_seeds: Sequence[int],
    module_dir: Path,
    armg_root: Path,
    base_weight: Path,
    prior_path: Path,
    strength: float,
    mcts_sims: int,
    workers: int,
) -> list[dict[str, Any]]:
    payloads = [
        (
            str(module_dir),
            str(armg_root),
            str(base_weight),
            str(prior_path),
            float(strength),
            int(seed),
            tuple(int(v) for v in all_seeds),
            int(mcts_sims),
        )
        for seed in seeds
    ]
    if workers == 1:
        return [_worker(payload) for payload in payloads]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(workers), len(payloads)),
        mp_context=context,
    ) as pool:
        return list(pool.map(_worker, payloads))


def run_shard(args: argparse.Namespace) -> int:
    seeds = shard_seeds(args.shard_index)
    all_seeds = fixed_seeds()
    prior_path = args.output.parent / f"expert-prior-shard-{args.shard_index}.json"
    dataset = build_card_prior(
        args.data_root,
        prior_path,
        min_ascension=args.min_ascension,
        alpha=args.alpha,
    )
    current = _evaluate(
        seeds=seeds,
        all_seeds=all_seeds,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        base_weight=args.armg_base_weight,
        prior_path=prior_path,
        strength=0.0,
        mcts_sims=args.combat_mcts_sims,
        workers=args.eval_workers,
    )
    candidate = _evaluate(
        seeds=seeds,
        all_seeds=all_seeds,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        base_weight=args.armg_base_weight,
        prior_path=prior_path,
        strength=args.candidate_strength,
        mcts_sims=args.combat_mcts_sims,
        workers=args.eval_workers,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "shard",
        "shard_index": int(args.shard_index),
        "shard_count": SHARD_COUNT,
        "seed_base": SEED_BASE,
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "candidate_strength": float(args.candidate_strength),
        "combat_policy": f"mcts_{args.combat_mcts_sims}",
        "corpus_sha256": dataset["corpus_sha256"],
        "expert_runs": dataset["accepted_runs"],
        "expert_choices": dataset["card_choice_examples"],
        "holdout_top1": dataset["holdout_top1"],
        "current": {"runs": current},
        "candidate": {"runs": candidate},
    }
    _write_json(args.output, payload)
    print(json.dumps({
        "result": "PASS_HUMAN_EXPERT_500_SHARD",
        "shard": args.shard_index,
        "seeds": len(seeds),
    }, sort_keys=True))
    return 0


def _sha_run_set(rows: Sequence[Mapping[str, Any]]) -> str:
    slim = [
        {
            "seed": row.get("seed"),
            "outcome": row.get("outcome"),
            "final_floor": row.get("final_floor"),
            "result": row.get("result"),
        }
        for row in sorted(rows, key=lambda row: int(row.get("seed", 0)))
    ]
    raw = json.dumps(slim, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def aggregate(args: argparse.Namespace) -> int:
    paths = sorted(args.input_dir.rglob("shard-*.json"))
    if len(paths) != SHARD_COUNT:
        raise RuntimeError(f"expected {SHARD_COUNT} shard files, found {len(paths)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("schema_version") != SCHEMA_VERSION for row in payloads):
        raise RuntimeError("shard schema mismatch")
    indices = sorted(int(row["shard_index"]) for row in payloads)
    if indices != list(range(SHARD_COUNT)):
        raise RuntimeError(f"shard indices mismatch: {indices}")
    corpus = {str(row["corpus_sha256"]) for row in payloads}
    strengths = {float(row["candidate_strength"]) for row in payloads}
    combats = {str(row["combat_policy"]) for row in payloads}
    if len(corpus) != 1 or len(strengths) != 1 or len(combats) != 1:
        raise RuntimeError("shards disagree on corpus/strength/combat policy")

    current_runs: list[dict[str, Any]] = []
    candidate_runs: list[dict[str, Any]] = []
    for row in payloads:
        current_runs.extend(list((row.get("current") or {}).get("runs") or []))
        candidate_runs.extend(list((row.get("candidate") or {}).get("runs") or []))

    expected = set(fixed_seeds())
    current_seeds = {int(row["seed"]) for row in current_runs}
    candidate_seeds = {int(row["seed"]) for row in candidate_runs}
    if len(current_runs) != SEED_COUNT or len(candidate_runs) != SEED_COUNT:
        raise RuntimeError("500 confirmation did not produce exactly 500 runs per side")
    if current_seeds != expected or candidate_seeds != expected:
        raise RuntimeError("500 confirmation seed set mismatch")

    combat_policy = next(iter(combats))
    gate_policy = StrategyGatePolicy(
        "human-expert-confirm-500",
        SEED_COUNT,
        min_win_delta=1,
        min_floor_delta=0.0,
        max_floor_regression_with_win_gain=60.0,
        max_one_sided_sign_p=0.10,
    )
    gate = evaluate_strategy_gate(
        {"runs": current_runs},
        {"runs": candidate_runs},
        policy=gate_policy,
        expected_combat_policy=combat_policy,
    )
    confirmed = gate["status"] == "PASS"
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "aggregate",
        "result": "PASS_HUMAN_EXPERT_500_COMPLETE",
        "decision": "CONFIRM_HUMAN_EXPERT" if confirmed else "DO_NOT_CONFIRM_HUMAN_EXPERT",
        "candidate_strength": next(iter(strengths)),
        "combat_policy": combat_policy,
        "corpus_sha256": next(iter(corpus)),
        "seed_base": SEED_BASE,
        "seed_count": SEED_COUNT,
        "seed_set_sha256": hashlib.sha256(
            "\n".join(str(v) for v in fixed_seeds()).encode("utf-8")
        ).hexdigest(),
        "current_runset_sha256": _sha_run_set(current_runs),
        "candidate_runset_sha256": _sha_run_set(candidate_runs),
        "gate": gate,
        "production_champion_replaced": False,
    }
    _write_json(args.output, report)
    print(json.dumps({
        "result": report["result"],
        "decision": report["decision"],
        "current_wins": gate["current"]["victories"],
        "candidate_wins": gate["candidate"]["victories"],
        "win_delta": gate["win_delta"],
        "paired_win_p": gate["paired_wins"]["one_sided_sign_p"],
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    shard = sub.add_parser("shard")
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--data-root", type=Path, required=True)
    shard.add_argument("--module-dir", type=Path, required=True)
    shard.add_argument("--armg-root", type=Path, required=True)
    shard.add_argument("--armg-base-weight", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--candidate-strength", type=float, default=2.0)
    shard.add_argument("--combat-mcts-sims", type=int, default=2000)
    shard.add_argument("--eval-workers", type=int, default=4)
    shard.add_argument("--min-ascension", type=int, default=15)
    shard.add_argument("--alpha", type=float, default=2.0)

    agg = sub.add_parser("aggregate")
    agg.add_argument("--input-dir", type=Path, required=True)
    agg.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.mode == "shard":
        return run_shard(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
