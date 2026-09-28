#!/usr/bin/env python3
"""ArmG Strategy v1.5 gate: Dev-30 -> fresh Hidden-50 -> fresh Final-100.

Combat is frozen to one pure-MCTS policy. This gate rejects any run where a
Student or hybrid Student vote participated in combat.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import sts1_armg_ppo_v13_auto_gate as rt


SCHEMA = "sts1-armg-strategy-v15-gate-v1"


def _read_seed_file(path: Path) -> list[int]:
    values = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != 50 or len(set(values)) != 50:
        raise RuntimeError(f"{path} must contain exactly 50 unique seeds")
    return values


def _fresh_seeds(*, count: int, rng_seed: int, forbidden: set[int]) -> list[int]:
    rng = random.Random(rng_seed)
    values: list[int] = []
    seen = set(forbidden)
    while len(values) < count:
        value = rng.randrange(1, 2**31 - 1)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _safe(row: dict[str, Any]) -> bool:
    return all(
        int(row.get(key, 0) or 0) == 0
        for key in (
            "illegal_action_count",
            "timeout_count",
            "crash_count",
            "remote_error_count",
        )
    )


def _pure_mcts(row: dict[str, Any], expected: str) -> bool:
    return (
        row.get("combat_policy") == expected
        and int(row.get("student_action_count", 0) or 0) == 0
        and int(row.get("hybrid_student_vote_count", 0) or 0) == 0
        and int(row.get("hybrid_student_tiebreak_count", 0) or 0) == 0
    )


def _complete(row: dict[str, Any]) -> bool:
    return (
        row.get("result") == "PASS_SIMULATOR_COMPLETE_RUN"
        and row.get("outcome") in {"victory", "defeat"}
    )


def _one_sided_sign_p(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(better, n + 1)) / float(2**n)


def _paired_summary(
    parent: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    expected_policy: str,
) -> dict[str, Any]:
    if len(parent) != len(candidate):
        raise RuntimeError("paired evaluation length mismatch")
    p = {int(row["seed"]): row for row in parent}
    c = {int(row["seed"]): row for row in candidate}
    if set(p) != set(c) or len(p) != len(parent):
        raise RuntimeError("paired evaluation seed mismatch")

    complete = all(_complete(row) for row in parent + candidate)
    safe = all(_safe(row) for row in candidate)
    combat_locked = all(
        _pure_mcts(row, expected_policy) for row in parent + candidate
    )

    parent_wins = sum(row.get("outcome") == "victory" for row in parent)
    candidate_wins = sum(row.get("outcome") == "victory" for row in candidate)
    floor_diffs: list[float] = []
    better = worse = ties = 0

    for seed in sorted(p):
        pr = p[seed]
        cr = c[seed]
        pf = float(pr.get("final_floor") or 0)
        cf = float(cr.get("final_floor") or 0)
        floor_diffs.append(cf - pf)

        pw = pr.get("outcome") == "victory"
        cw = cr.get("outcome") == "victory"
        if pw != cw:
            if cw:
                better += 1
            else:
                worse += 1
        elif cf > pf:
            better += 1
        elif cf < pf:
            worse += 1
        else:
            ties += 1

    return {
        "seeds": len(parent),
        "complete": complete,
        "candidate_safe": safe,
        "combat_locked_pure_mcts": combat_locked,
        "parent_wins": parent_wins,
        "candidate_wins": candidate_wins,
        "win_delta": candidate_wins - parent_wins,
        "mean_paired_floor_delta": float(statistics.mean(floor_diffs)),
        "median_paired_floor_delta": float(statistics.median(floor_diffs)),
        "candidate_better": better,
        "candidate_worse": worse,
        "ties": ties,
        "one_sided_sign_p": _one_sided_sign_p(better, worse),
    }


def _gate_dev(summary: dict[str, Any]) -> dict[str, Any]:
    material = (
        summary["win_delta"] >= 1
        or (
            summary["win_delta"] >= 0
            and summary["mean_paired_floor_delta"] >= 0.75
            and summary["median_paired_floor_delta"] >= 0.0
        )
    )
    passed = (
        summary["complete"]
        and summary["candidate_safe"]
        and summary["combat_locked_pure_mcts"]
        and summary["win_delta"] >= 0
        and material
    )
    return {
        "gate": "dev-30",
        "status": "PASS" if passed else "HOLD",
        "summary": summary,
    }


def _gate_hidden(summary: dict[str, Any]) -> dict[str, Any]:
    material = (
        summary["win_delta"] >= 1
        or (
            summary["win_delta"] >= 0
            and summary["mean_paired_floor_delta"] >= 0.50
            and summary["median_paired_floor_delta"] >= 0.0
        )
    )
    passed = (
        summary["complete"]
        and summary["candidate_safe"]
        and summary["combat_locked_pure_mcts"]
        and summary["win_delta"] >= 0
        and material
    )
    return {
        "gate": "hidden-50-fresh",
        "status": "PASS" if passed else "HOLD",
        "summary": summary,
    }


def _gate_fresh(summary: dict[str, Any]) -> dict[str, Any]:
    passed = (
        summary["complete"]
        and summary["candidate_safe"]
        and summary["combat_locked_pure_mcts"]
        and summary["win_delta"] >= 1
        and summary["mean_paired_floor_delta"] >= -0.50
        and summary["candidate_better"] > summary["candidate_worse"]
        and summary["one_sided_sign_p"] <= 0.10
    )
    return {
        "gate": "fresh-100",
        "status": "PASS" if passed else "HOLD",
        "summary": summary,
    }


def _evaluate_pair(
    *,
    label: str,
    parent_weight: Path,
    candidate_weight: Path,
    seeds: list[int],
    module_dir: Path,
    armg_root: Path,
    output_dir: Path,
    mcts_sims: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests = (
        [(f"{label}-parent", parent_weight, seed) for seed in seeds]
        + [(f"{label}-candidate", candidate_weight, seed) for seed in seeds]
    )
    results = rt._evaluate_parallel(
        requests=requests,
        module_dir=module_dir,
        armg_root=armg_root,
        output_dir=output_dir / label,
        mcts_sims=mcts_sims,
        all_formal_seeds=seeds,
        workers=workers,
    )
    parent = [
        results[(f"{label}-parent", seed)]
        for seed in seeds
    ]
    candidate = [
        results[(f"{label}-candidate", seed)]
        for seed in seeds
    ]
    return parent, candidate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module-dir", type=Path, required=True)
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--parent-weight", type=Path, required=True)
    p.add_argument("--candidate-weight", type=Path, required=True)
    p.add_argument("--dev-seed-file", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mcts-sims", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rng-seed", type=int, default=20260928)
    args = p.parse_args()

    if args.mcts_sims < 1 or args.workers < 1:
        raise RuntimeError("mcts/workers must be positive")

    state = json.loads(args.state.read_text(encoding="utf-8"))
    round_index = int(state.get("round_index", 0)) + 1
    dev_all = _read_seed_file(args.dev_seed_file)
    dev30 = dev_all[:30]

    historical = {
        int(x) for x in state.get("strategy_v15_used_eval_seeds", [])
    }
    forbidden = set(dev_all) | historical
    expected_policy = f"mcts_{args.mcts_sims}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parent_dev, candidate_dev = _evaluate_pair(
        label="dev30",
        parent_weight=args.parent_weight,
        candidate_weight=args.candidate_weight,
        seeds=dev30,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        output_dir=args.output_dir,
        mcts_sims=args.mcts_sims,
        workers=args.workers,
    )
    dev = _gate_dev(
        _paired_summary(
            parent_dev,
            candidate_dev,
            expected_policy=expected_policy,
        )
    )

    hidden: dict[str, Any] = {
        "gate": "hidden-50-fresh",
        "status": "SKIPPED",
        "reason": "dev_gate_not_passed",
    }
    final: dict[str, Any] = {
        "gate": "fresh-100",
        "status": "SKIPPED",
        "reason": "hidden_gate_not_passed",
    }
    hidden_seeds: list[int] = []
    final_seeds: list[int] = []

    if dev["status"] == "PASS":
        hidden_seeds = _fresh_seeds(
            count=50,
            rng_seed=args.rng_seed + round_index * 1009 + 1,
            forbidden=forbidden,
        )
        forbidden.update(hidden_seeds)
        parent_h, candidate_h = _evaluate_pair(
            label="hidden50",
            parent_weight=args.parent_weight,
            candidate_weight=args.candidate_weight,
            seeds=hidden_seeds,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            output_dir=args.output_dir,
            mcts_sims=args.mcts_sims,
            workers=args.workers,
        )
        hidden = _gate_hidden(
            _paired_summary(
                parent_h,
                candidate_h,
                expected_policy=expected_policy,
            )
        )

    if hidden.get("status") == "PASS":
        final_seeds = _fresh_seeds(
            count=100,
            rng_seed=args.rng_seed + round_index * 1009 + 2,
            forbidden=forbidden,
        )
        forbidden.update(final_seeds)
        parent_f, candidate_f = _evaluate_pair(
            label="fresh100",
            parent_weight=args.parent_weight,
            candidate_weight=args.candidate_weight,
            seeds=final_seeds,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            output_dir=args.output_dir,
            mcts_sims=args.mcts_sims,
            workers=args.workers,
        )
        final = _gate_fresh(
            _paired_summary(
                parent_f,
                candidate_f,
                expected_policy=expected_policy,
            )
        )

    ready = (
        dev["status"] == "PASS"
        and hidden.get("status") == "PASS"
        and final.get("status") == "PASS"
    )
    consumed = hidden_seeds + final_seeds
    state["strategy_v15_used_eval_seeds"] = (
        list(state.get("strategy_v15_used_eval_seeds", [])) + consumed
    )
    state["strategy_v15_last_gate"] = {
        "round_index": round_index,
        "dev": dev["status"],
        "hidden": hidden.get("status"),
        "fresh": final.get("status"),
        "ready": ready,
    }
    args.state.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": SCHEMA,
        "decision": "ADOPT_PARENT" if ready else "HOLD_PARENT",
        "final_readiness": "READY_FOR_REAL_GAME" if ready else "NOT_READY",
        "production_champion_replaced": False,
        "combat_training_enabled": False,
        "combat_policy": expected_policy,
        "dev_gate": dev,
        "hidden_gate": hidden,
        "fresh_gate": final,
        "hidden_seed_count": len(hidden_seeds),
        "fresh_seed_count": len(final_seeds),
        "historical_eval_seed_count_before": len(historical),
        "historical_eval_seed_count_after": len(historical) + len(consumed),
    }
    (args.output_dir / "gate-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("STRATEGY_V15_GATE", json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
