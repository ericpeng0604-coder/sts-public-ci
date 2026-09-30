#!/usr/bin/env python3
"""One-shot Human Expert v5 path benchmark.

Parent and candidate share the exact same card prior and card strength.
The only difference is the candidate's Human Path Prior, so measured win-rate
changes isolate the effect of expert pathing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from roguelike_ai.sts1_phase3.armg_strategy_evolve import (
    CONFIRM_500_STRATEGY_GATE,
    DEV_STRATEGY_GATE,
    FRESH_STRATEGY_GATE,
    HIDDEN_STRATEGY_GATE,
    StrategyGatePolicy,
    evaluate_strategy_gate,
)
from roguelike_ai.sts1_phase3.human_expert import HumanExpertPolicy, build_card_prior
from roguelike_ai.sts1_phase3.human_path_prior import build_path_prior
from roguelike_ai.sts1_phase3.simulator import _load_sts, run_simulator_game


REPORT_SCHEMA = "sts1-human-expert-v5-path-benchmark-v2"

DEV_CONFIRM_20_GATE = StrategyGatePolicy(
    "human-expert-v5-dev-confirm-20",
    20,
    min_win_delta=0,
    min_floor_delta=0.0,
    max_floor_regression_with_win_gain=60.0,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_seeds(path: Path) -> tuple[int, ...]:
    values = tuple(
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(values) != 50 or len(set(values)) != 50:
        raise RuntimeError("v5 requires exactly 50 unique frozen dev seeds")
    return values


def _fresh_seeds(*, count: int, rng_seed: int, forbidden: set[int]) -> tuple[int, ...]:
    rng = random.Random(int(rng_seed))
    used = set(int(v) for v in forbidden)
    out: list[int] = []
    while len(out) < count:
        value = rng.randint(1, 10**9)
        if value in used:
            continue
        used.add(value)
        out.append(value)
    return tuple(out)


def _worker(
    payload: tuple[str, str, str, str, float, str, float, float, float, int, tuple[int, ...], int],
) -> dict[str, Any]:
    (
        module_dir_text,
        armg_root_text,
        weight_text,
        card_prior_text,
        card_strength,
        path_prior_text,
        path_strength,
        path_min_prior_spread,
        path_max_armg_margin,
        seed,
        all_seeds,
        mcts_sims,
    ) = payload
    module_dir = Path(module_dir_text)
    sts = _load_sts(module_dir)
    policy = HumanExpertPolicy(
        root=Path(armg_root_text),
        weight_path=Path(weight_text),
        expert_prior_path=Path(card_prior_text),
        expert_strength=float(card_strength),
        path_prior_path=Path(path_prior_text),
        path_strength=float(path_strength),
        path_min_prior_spread=float(path_min_prior_spread),
        path_max_armg_margin=float(path_max_armg_margin),
    )
    result = dict(
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
    result["v5_path_diagnostics"] = policy.path_diagnostics_snapshot()
    return result


def _evaluate(
    *,
    seeds: Sequence[int],
    module_dir: Path,
    armg_root: Path,
    base_weight: Path,
    card_prior: Path,
    card_strength: float,
    path_prior: Path,
    path_strength: float,
    path_min_prior_spread: float = 0.0,
    path_max_armg_margin: float = float("inf"),
    mcts_sims: int = 2000,
    workers: int,
) -> dict[str, Any]:
    ordered = tuple(int(seed) for seed in seeds)
    payloads = [
        (
            str(module_dir),
            str(armg_root),
            str(base_weight),
            str(card_prior),
            float(card_strength),
            str(path_prior),
            float(path_strength),
            float(path_min_prior_spread),
            float(path_max_armg_margin),
            seed,
            ordered,
            int(mcts_sims),
        )
        for seed in ordered
    ]
    if workers == 1:
        runs = [_worker(payload) for payload in payloads]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(payloads)),
            mp_context=ctx,
        ) as pool:
            runs = list(pool.map(_worker, payloads))
    return {"runs": runs}


def _rank(gate: Mapping[str, Any]) -> tuple[float, ...]:
    paired = gate.get("paired_wins") or {}
    return (
        1.0 if gate.get("status") == "PASS" else 0.0,
        float(gate.get("win_delta", -9999) or 0),
        float(paired.get("candidate_better", 0) or 0)
        - float(paired.get("candidate_worse", 0) or 0),
        -float(paired.get("one_sided_sign_p", 1.0) or 1.0),
    )


def _should_confirm_500(fresh_gate: Mapping[str, Any]) -> bool:
    if fresh_gate.get("status") != "ROLLBACK":
        return False
    reasons = set(str(v) for v in fresh_gate.get("reasons", []))
    if reasons != {"paired_superiority_not_strong_enough"}:
        return False
    paired = fresh_gate.get("paired_wins") or {}
    return (
        int(fresh_gate.get("win_delta", 0) or 0) >= 1
        and int(paired.get("candidate_better", 0) or 0)
        > int(paired.get("candidate_worse", 0) or 0)
    )


def _paired_gate(
    *,
    parent: dict[str, Any],
    candidate: dict[str, Any],
    policy: Any,
    mcts_sims: int,
) -> dict[str, Any]:
    return evaluate_strategy_gate(
        parent,
        candidate,
        policy=policy,
        expected_combat_policy=f"mcts_{mcts_sims}",
    )


def _aggregate_path_diagnostics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    decisions = 0
    flips = 0
    eligible = 0
    blocked_low_prior = 0
    blocked_armg = 0
    spread_sum = 0.0
    raw_margin_sum = 0.0
    max_spread = 0.0
    max_raw_margin = 0.0
    rooms: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for run in list(evaluation.get("runs") or []):
        diag = dict(run.get("v5_path_diagnostics") or {})
        d = int(diag.get("map_decisions", 0) or 0)
        f = int(diag.get("map_flips", 0) or 0)
        decisions += d
        flips += f
        eligible += int(diag.get("eligible_decisions", 0) or 0)
        blocked_low_prior += int(diag.get("blocked_low_prior_confidence", 0) or 0)
        blocked_armg += int(diag.get("blocked_armg_confident", 0) or 0)
        spread_sum += float(diag.get("addition_spread_sum", 0.0) or 0.0)
        raw_margin_sum += float(diag.get("raw_margin_sum", 0.0) or 0.0)
        max_spread = max(max_spread, float(diag.get("addition_spread_max", 0.0) or 0.0))
        max_raw_margin = max(max_raw_margin, float(diag.get("raw_margin_max", 0.0) or 0.0))
        for room, count in dict(diag.get("rooms_seen") or {}).items():
            rooms[str(room)] = rooms.get(str(room), 0) + int(count)
        for row in list(diag.get("flip_examples") or []):
            if len(examples) < 25:
                examples.append(dict(row))
    return {
        "map_decisions": decisions,
        "map_flips": flips,
        "flip_rate": (flips / decisions) if decisions else 0.0,
        "eligible_decisions": eligible,
        "eligible_rate": (eligible / decisions) if decisions else 0.0,
        "blocked_low_prior_confidence": blocked_low_prior,
        "blocked_armg_confident": blocked_armg,
        "mean_addition_spread": (spread_sum / decisions) if decisions else 0.0,
        "max_addition_spread": max_spread,
        "mean_raw_margin": (raw_margin_sum / decisions) if decisions else 0.0,
        "max_raw_margin": max_raw_margin,
        "rooms_seen": dict(sorted(rooms.items())),
        "flip_examples": examples,
    }


def _summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    parent = dict(gate.get("current") or {})
    candidate = dict(gate.get("candidate") or {})
    n = int(candidate.get("seeds", parent.get("seeds", 0)) or 0)
    return {
        "status": gate.get("status"),
        "seeds": n,
        "parent_wins": int(parent.get("victories", 0) or 0),
        "candidate_wins": int(candidate.get("victories", 0) or 0),
        "parent_win_rate": (float(parent.get("victories", 0) or 0) / n) if n else 0.0,
        "candidate_win_rate": (float(candidate.get("victories", 0) or 0) / n) if n else 0.0,
        "win_delta": int(gate.get("win_delta", 0) or 0),
        "paired_wins": dict(gate.get("paired_wins") or {}),
        "reasons": list(gate.get("reasons") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--armg-base-weight", type=Path, required=True)
    parser.add_argument("--dev-seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--card-strength", type=float, default=2.0)
    parser.add_argument("--path-strengths", default="8.00")
    parser.add_argument("--path-min-prior-spreads", default="0.10,0.20")
    parser.add_argument("--path-max-armg-margins", default="1.0,2.0,3.0")
    parser.add_argument("--combat-mcts-sims", type=int, default=2000)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--rng-seed", type=int, default=2026093005)
    parser.add_argument("--min-ascension", type=int, default=15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    card_prior = args.output_dir / "card-prior.json"
    path_prior = args.output_dir / "path-prior.json"
    card_data = build_card_prior(
        args.data_root,
        card_prior,
        min_ascension=args.min_ascension,
        alpha=2.0,
    )
    path_data = build_path_prior(
        args.data_root,
        path_prior,
        min_ascension=args.min_ascension,
        alpha=3.0,
    )

    strengths = tuple(float(v.strip()) for v in args.path_strengths.split(",") if v.strip())
    min_spreads = tuple(float(v.strip()) for v in args.path_min_prior_spreads.split(",") if v.strip())
    max_margins = tuple(float(v.strip()) for v in args.path_max_armg_margins.split(",") if v.strip())
    if not strengths or any(v <= 0 or v > 20.0 for v in strengths):
        raise RuntimeError("path strengths must be within (0, 20]")
    if not min_spreads or any(v < 0 or v > 5.0 for v in min_spreads):
        raise RuntimeError("path minimum prior spreads must be within [0, 5]")
    if not max_margins or any(v < 0 or v > 20.0 for v in max_margins):
        raise RuntimeError("path maximum ArmG margins must be within [0, 20]")

    frozen50 = _read_seeds(args.dev_seed_file)
    dev30 = frozen50[:30]
    dev_confirm20 = frozen50[30:]
    parent_dev = _evaluate(
        seeds=dev30,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        base_weight=args.armg_base_weight,
        card_prior=card_prior,
        card_strength=args.card_strength,
        path_prior=path_prior,
        path_strength=0.0,
        mcts_sims=args.combat_mcts_sims,
        workers=args.eval_workers,
    )

    candidates: list[dict[str, Any]] = []
    for strength in strengths:
        for min_spread in min_spreads:
            for max_margin in max_margins:
                candidate_dev = _evaluate(
                    seeds=dev30,
                    module_dir=args.module_dir,
                    armg_root=args.armg_root,
                    base_weight=args.armg_base_weight,
                    card_prior=card_prior,
                    card_strength=args.card_strength,
                    path_prior=path_prior,
                    path_strength=strength,
                    path_min_prior_spread=min_spread,
                    path_max_armg_margin=max_margin,
                    mcts_sims=args.combat_mcts_sims,
                    workers=args.eval_workers,
                )
                gate = _paired_gate(
                    parent=parent_dev,
                    candidate=candidate_dev,
                    policy=DEV_STRATEGY_GATE,
                    mcts_sims=args.combat_mcts_sims,
                )
                candidates.append({
                    "path_strength": strength,
                    "path_min_prior_spread": min_spread,
                    "path_max_armg_margin": max_margin,
                    "gate": gate,
                    "eval": candidate_dev,
                    "path_diagnostics": _aggregate_path_diagnostics(candidate_dev),
                })
                label = f"s{strength:.2f}-p{min_spread:.2f}-m{max_margin:.2f}".replace(".", "p")
                _write_json(args.output_dir / f"dev30-{label}.json", candidate_dev)

    selected = max(candidates, key=lambda row: _rank(row["gate"]))
    path_strength = float(selected["path_strength"])
    path_min_prior_spread = float(selected["path_min_prior_spread"])
    path_max_armg_margin = float(selected["path_max_armg_margin"])
    dev_gate = dict(selected["gate"])
    _write_json(args.output_dir / "parent-dev30.json", parent_dev)

    dev_confirm_gate: dict[str, Any] = {
        "status": "SKIPPED",
        "reasons": ["dev_30_gate_did_not_pass"],
    }
    if dev_gate.get("status") == "PASS":
        parent_confirm = _evaluate(
            seeds=dev_confirm20,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            base_weight=args.armg_base_weight,
            card_prior=card_prior,
            card_strength=args.card_strength,
            path_prior=path_prior,
            path_strength=0.0,
            mcts_sims=args.combat_mcts_sims,
            workers=args.eval_workers,
        )
        candidate_confirm = _evaluate(
            seeds=dev_confirm20,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            base_weight=args.armg_base_weight,
            card_prior=card_prior,
            card_strength=args.card_strength,
            path_prior=path_prior,
            path_strength=path_strength,
            path_min_prior_spread=path_min_prior_spread,
            path_max_armg_margin=path_max_armg_margin,
            mcts_sims=args.combat_mcts_sims,
            workers=args.eval_workers,
        )
        dev_confirm_gate = _paired_gate(
            parent=parent_confirm,
            candidate=candidate_confirm,
            policy=DEV_CONFIRM_20_GATE,
            mcts_sims=args.combat_mcts_sims,
        )
        _write_json(args.output_dir / "parent-dev-confirm20.json", parent_confirm)
        _write_json(args.output_dir / "candidate-dev-confirm20.json", candidate_confirm)

    hidden_gate: dict[str, Any] = {
        "status":"SKIPPED",
        "reasons":["dev_confirmation_not_passed"],
    }
    fresh_gate: dict[str, Any] = {"status":"SKIPPED","reasons":["hidden_50_gate_not_passed"]}
    confirm_gate: dict[str, Any] = {"status":"SKIPPED","reasons":["fresh_100_did_not_require_confirmation"]}
    hidden_seeds: tuple[int, ...] = ()
    fresh_seeds: tuple[int, ...] = ()
    confirm_seeds: tuple[int, ...] = ()

    forbidden = set(frozen50)
    if dev_gate.get("status") == "PASS":
        hidden_seeds = _fresh_seeds(count=50, rng_seed=args.rng_seed+1, forbidden=forbidden)
        forbidden.update(hidden_seeds)
        parent = _evaluate(
            seeds=hidden_seeds, module_dir=args.module_dir, armg_root=args.armg_root,
            base_weight=args.armg_base_weight, card_prior=card_prior,
            card_strength=args.card_strength, path_prior=path_prior, path_strength=0.0,
            mcts_sims=args.combat_mcts_sims, workers=args.eval_workers,
        )
        candidate = _evaluate(
            seeds=hidden_seeds, module_dir=args.module_dir, armg_root=args.armg_root,
            base_weight=args.armg_base_weight, card_prior=card_prior,
            card_strength=args.card_strength, path_prior=path_prior, path_strength=path_strength,
            path_min_prior_spread=path_min_prior_spread,
            path_max_armg_margin=path_max_armg_margin,
            mcts_sims=args.combat_mcts_sims, workers=args.eval_workers,
        )
        hidden_gate = _paired_gate(
            parent=parent, candidate=candidate,
            policy=HIDDEN_STRATEGY_GATE, mcts_sims=args.combat_mcts_sims,
        )

    if hidden_gate.get("status") == "PASS":
        fresh_seeds = _fresh_seeds(count=100, rng_seed=args.rng_seed+2, forbidden=forbidden)
        forbidden.update(fresh_seeds)
        parent = _evaluate(
            seeds=fresh_seeds, module_dir=args.module_dir, armg_root=args.armg_root,
            base_weight=args.armg_base_weight, card_prior=card_prior,
            card_strength=args.card_strength, path_prior=path_prior, path_strength=0.0,
            mcts_sims=args.combat_mcts_sims, workers=args.eval_workers,
        )
        candidate = _evaluate(
            seeds=fresh_seeds, module_dir=args.module_dir, armg_root=args.armg_root,
            base_weight=args.armg_base_weight, card_prior=card_prior,
            card_strength=args.card_strength, path_prior=path_prior, path_strength=path_strength,
            path_min_prior_spread=path_min_prior_spread,
            path_max_armg_margin=path_max_armg_margin,
            mcts_sims=args.combat_mcts_sims, workers=args.eval_workers,
        )
        fresh_gate = _paired_gate(
            parent=parent, candidate=candidate,
            policy=FRESH_STRATEGY_GATE, mcts_sims=args.combat_mcts_sims,
        )

    if _should_confirm_500(fresh_gate):
        confirm_seeds = _fresh_seeds(count=500, rng_seed=args.rng_seed+3, forbidden=forbidden)
        parent = _evaluate(
            seeds=confirm_seeds, module_dir=args.module_dir, armg_root=args.armg_root,
            base_weight=args.armg_base_weight, card_prior=card_prior,
            card_strength=args.card_strength, path_prior=path_prior, path_strength=0.0,
            mcts_sims=args.combat_mcts_sims, workers=args.eval_workers,
        )
        candidate = _evaluate(
            seeds=confirm_seeds, module_dir=args.module_dir, armg_root=args.armg_root,
            base_weight=args.armg_base_weight, card_prior=card_prior,
            card_strength=args.card_strength, path_prior=path_prior, path_strength=path_strength,
            path_min_prior_spread=path_min_prior_spread,
            path_max_armg_margin=path_max_armg_margin,
            mcts_sims=args.combat_mcts_sims, workers=args.eval_workers,
        )
        confirm_gate = _paired_gate(
            parent=parent, candidate=candidate,
            policy=CONFIRM_500_STRATEGY_GATE, mcts_sims=args.combat_mcts_sims,
        )

    report = {
        "schema_version": REPORT_SCHEMA,
        "card_strength": args.card_strength,
        "selected_path_strength": path_strength,
        "selected_path_min_prior_spread": path_min_prior_spread,
        "selected_path_max_armg_margin": path_max_armg_margin,
        "path_strength_candidates": [
            {
                "path_strength": row["path_strength"],
                "path_min_prior_spread": row["path_min_prior_spread"],
                "path_max_armg_margin": row["path_max_armg_margin"],
                "dev": _summary(row["gate"]),
                "path_diagnostics": row["path_diagnostics"],
            }
            for row in candidates
        ],
        "card_dataset": {
            "accepted_runs": card_data["accepted_runs"],
            "card_choice_examples": card_data["card_choice_examples"],
            "holdout_top1": card_data["holdout_top1"],
        },
        "path_dataset": {
            "accepted_runs": path_data["accepted_runs"],
            "example_count": path_data["example_count"],
            "contexts": len(path_data["context_log_probs"]),
        },
        "dev30": _summary(dev_gate),
        "dev_confirm20": (
            _summary(dev_confirm_gate)
            if dev_confirm_gate.get("status") != "SKIPPED"
            else dev_confirm_gate
        ),
        "selected_dev_path_diagnostics": dict(selected["path_diagnostics"]),
        "hidden50": _summary(hidden_gate) if hidden_gate.get("status") != "SKIPPED" else hidden_gate,
        "fresh100": _summary(fresh_gate) if fresh_gate.get("status") != "SKIPPED" else fresh_gate,
        "confirm500": _summary(confirm_gate) if confirm_gate.get("status") != "SKIPPED" else confirm_gate,
        "production_champion_replaced": False,
        "combat_policy": f"mcts_{args.combat_mcts_sims}",
    }
    _write_json(args.output_dir / "v5-path-benchmark.json", report)
    print("HUMAN_EXPERT_V5_RESULT", json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
