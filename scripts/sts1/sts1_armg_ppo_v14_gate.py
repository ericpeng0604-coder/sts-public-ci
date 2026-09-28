#!/usr/bin/env python3
"""PPO v1.4 two-level gate: cumulative Dev Parent + strict Final promotion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.champion_gate import (
    FAST_GATE_POLICY,
    FORMAL_GATE_POLICY,
    GatePolicy,
    evaluate_fixed_seed_gate,
)

_RUNTIME_PATH = Path(__file__).with_name("sts1_armg_ppo_v13_auto_gate.py")
_SPEC = importlib.util.spec_from_file_location("sts1_v13_gate_runtime", _RUNTIME_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load v1.3 gate runtime")
rt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rt)

CACHE_SCHEMA = "sts1-armg-ppo-v14-eval-cache-v1"
DEV_POLICY = GatePolicy("dev-parent-30", 30, 0)


def _read_seeds(path: Path) -> list[int]:
    seeds = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise RuntimeError(f"{path} must contain exactly 50 unique seeds")
    return seeds


def _cache_identity(
    *,
    weight_sha: str,
    seeds: list[int],
    mcts_sims: int,
    simulator_id: str,
) -> dict[str, Any]:
    return {
        "weight_sha256": weight_sha,
        "seeds": list(seeds),
        "mcts_sims": int(mcts_sims),
        "simulator_id": simulator_id,
    }


def _load_cache(
    state: dict[str, Any],
    *,
    key: str,
    weight_sha: str,
    seeds: list[int],
    mcts_sims: int,
    simulator_id: str,
) -> dict[int, dict[str, Any]]:
    payload = state.get("v14_eval_caches", {}).get(key)
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA:
        return {}
    expected = _cache_identity(
        weight_sha=weight_sha,
        seeds=seeds,
        mcts_sims=mcts_sims,
        simulator_id=simulator_id,
    )
    if any(payload.get(k) != v for k, v in expected.items()):
        return {}
    out: dict[int, dict[str, Any]] = {}
    allowed = set(seeds)
    for row in payload.get("runs", []):
        seed = int(row["seed"])
        if seed not in allowed or seed in out:
            raise RuntimeError(f"invalid cached seed {seed}")
        if row.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
            raise RuntimeError("cache contains incomplete run")
        out[seed] = dict(row)
    print(f"PPO_V14_CACHE_HIT key={key} runs={len(out)}", flush=True)
    return out


def _save_cache(
    state: dict[str, Any],
    *,
    key: str,
    weight_sha: str,
    seeds: list[int],
    mcts_sims: int,
    simulator_id: str,
    runs: dict[int, dict[str, Any]],
) -> None:
    ordered = [runs[s] for s in seeds if s in runs]
    state.setdefault("v14_eval_caches", {})[key] = {
        "schema_version": CACHE_SCHEMA,
        **_cache_identity(
            weight_sha=weight_sha,
            seeds=seeds,
            mcts_sims=mcts_sims,
            simulator_id=simulator_id,
        ),
        "runs": ordered,
    }
    print(f"PPO_V14_CACHE_WRITE key={key} runs={len(ordered)}", flush=True)


def _run_missing(
    *,
    label: str,
    weight: Path,
    seeds: list[int],
    cache: dict[int, dict[str, Any]],
    module_dir: Path,
    armg_root: Path,
    output_dir: Path,
    mcts_sims: int,
    all_seeds: list[int],
    workers: int,
) -> dict[int, dict[str, Any]]:
    requests = [(label, weight, seed) for seed in seeds if seed not in cache]
    fresh = rt._evaluate_parallel(
        requests=requests,
        module_dir=module_dir,
        armg_root=armg_root,
        output_dir=output_dir,
        mcts_sims=mcts_sims,
        all_formal_seeds=all_seeds,
        workers=workers,
    )
    for (_, seed), result in fresh.items():
        cache[int(seed)] = result
    return cache


def _ordered(runs: dict[int, dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    missing = [seed for seed in seeds if seed not in runs]
    if missing:
        raise RuntimeError(f"missing seeds: {missing}")
    return [runs[seed] for seed in seeds]


def _dev_gate(parent_runs: list[dict[str, Any]], candidate_runs: list[dict[str, Any]]) -> dict[str, Any]:
    base = evaluate_fixed_seed_gate(parent_runs, candidate_runs, policy=DEV_POLICY)
    by_parent = {int(r["seed"]): r for r in parent_runs}
    by_candidate = {int(r["seed"]): r for r in candidate_runs}

    diffs = []
    reach50_parent = 0
    reach50_candidate = 0
    for seed in sorted(by_parent):
        p = by_parent[seed]
        c = by_candidate[seed]
        pf = float(p.get("final_floor") or 0)
        cf = float(c.get("final_floor") or 0)
        diffs.append(cf - pf)
        reach50_parent += int(pf >= 50)
        reach50_candidate += int(cf >= 50)

    mean_delta = float(statistics.mean(diffs))
    median_delta = float(statistics.median(diffs))
    win_delta = int(base["win_delta"])
    safe = all(v == 0 for v in base["candidate"]["safety"].values())
    complete = (
        base["candidate"]["complete_runs"] == 30
        and base["champion"]["complete_runs"] == 30
    )
    wins_not_worse = win_delta >= 0
    reach50_delta = reach50_candidate - reach50_parent

    win_improvement = win_delta >= 1
    floor_improvement = (
        mean_delta >= 0.75
        and median_delta >= 0.0
        and reach50_delta >= 0
    )
    adopt = complete and safe and wins_not_worse and (
        win_improvement or floor_improvement
    )

    reasons: list[str] = []
    if not complete:
        reasons.append("incomplete_dev_runs")
    if not safe:
        reasons.append("candidate_safety_failure")
    if not wins_not_worse:
        reasons.append("dev_wins_regressed")
    if not (win_improvement or floor_improvement):
        reasons.append("no_material_dev_improvement")

    return {
        "schema_version": "sts1-armg-ppo-v14-dev-gate-v1",
        "gate": "dev-parent-30",
        "status": "PASS" if adopt else "HOLD",
        "decision": "ADOPT_PARENT" if adopt else "HOLD_PARENT",
        "seeds": 30,
        "win_delta": win_delta,
        "mean_paired_floor_delta": mean_delta,
        "median_paired_floor_delta": median_delta,
        "reach50_delta": reach50_delta,
        "parent": base["champion"],
        "candidate": base["candidate"],
        "criteria": {
            "wins_not_worse": True,
            "win_improvement": "win_delta >= 1",
            "floor_improvement": (
                "mean_paired_floor_delta >= 0.75 and "
                "median_paired_floor_delta >= 0 and reach50_delta >= 0"
            ),
            "candidate_safety_zero": True,
        },
        "reasons": reasons,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module-dir", type=Path, required=True)
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--parent-weight", type=Path, required=True)
    p.add_argument("--baseline-weight", type=Path, required=True)
    p.add_argument("--candidate-weight", type=Path, required=True)
    p.add_argument("--dev-seed-file", type=Path, required=True)
    p.add_argument("--final-seed-file", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mcts-sims", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    dev_seeds = _read_seeds(args.dev_seed_file)
    final_seeds = _read_seeds(args.final_seed_file)
    overlap = sorted(set(dev_seeds) & set(final_seeds))
    if overlap:
        raise RuntimeError(f"dev/final seed leakage: {overlap}")

    state = json.loads(args.state.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulator_id = rt._automatic_simulator_id(args.module_dir)
    parent_sha = rt._sha256(args.parent_weight)
    baseline_sha = rt._sha256(args.baseline_weight)
    candidate_sha = rt._sha256(args.candidate_weight)

    dev_parent = _load_cache(
        state,
        key="dev_parent",
        weight_sha=parent_sha,
        seeds=dev_seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
    )
    dev_parent = _run_missing(
        label="dev-parent",
        weight=args.parent_weight,
        seeds=dev_seeds[:30],
        cache=dev_parent,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        output_dir=args.output_dir,
        mcts_sims=args.mcts_sims,
        all_seeds=dev_seeds,
        workers=args.workers,
    )
    dev_candidate: dict[int, dict[str, Any]] = {}
    dev_candidate = _run_missing(
        label="dev-candidate",
        weight=args.candidate_weight,
        seeds=dev_seeds[:30],
        cache=dev_candidate,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        output_dir=args.output_dir,
        mcts_sims=args.mcts_sims,
        all_seeds=dev_seeds,
        workers=args.workers,
    )
    _save_cache(
        state,
        key="dev_parent",
        weight_sha=parent_sha,
        seeds=dev_seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
        runs=dev_parent,
    )
    dev_gate = _dev_gate(
        _ordered(dev_parent, dev_seeds[:30]),
        _ordered(dev_candidate, dev_seeds[:30]),
    )

    final30: dict[str, Any]
    final50: dict[str, Any]
    if dev_gate["status"] == "PASS":
        final_base = _load_cache(
            state,
            key="final_baseline",
            weight_sha=baseline_sha,
            seeds=final_seeds,
            mcts_sims=args.mcts_sims,
            simulator_id=simulator_id,
        )
        final_base = _run_missing(
            label="final-baseline",
            weight=args.baseline_weight,
            seeds=final_seeds[:30],
            cache=final_base,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            output_dir=args.output_dir,
            mcts_sims=args.mcts_sims,
            all_seeds=final_seeds,
            workers=args.workers,
        )
        final_candidate: dict[int, dict[str, Any]] = {}
        final_candidate = _run_missing(
            label="final-candidate",
            weight=args.candidate_weight,
            seeds=final_seeds[:30],
            cache=final_candidate,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            output_dir=args.output_dir,
            mcts_sims=args.mcts_sims,
            all_seeds=final_seeds,
            workers=args.workers,
        )
        final30 = evaluate_fixed_seed_gate(
            _ordered(final_base, final_seeds[:30]),
            _ordered(final_candidate, final_seeds[:30]),
            policy=FAST_GATE_POLICY,
        )

        if final30["status"] == "PASS":
            final_base = _run_missing(
                label="final-baseline",
                weight=args.baseline_weight,
                seeds=final_seeds[30:],
                cache=final_base,
                module_dir=args.module_dir,
                armg_root=args.armg_root,
                output_dir=args.output_dir,
                mcts_sims=args.mcts_sims,
                all_seeds=final_seeds,
                workers=args.workers,
            )
            final_candidate = _run_missing(
                label="final-candidate",
                weight=args.candidate_weight,
                seeds=final_seeds[30:],
                cache=final_candidate,
                module_dir=args.module_dir,
                armg_root=args.armg_root,
                output_dir=args.output_dir,
                mcts_sims=args.mcts_sims,
                all_seeds=final_seeds,
                workers=args.workers,
            )
            final50 = evaluate_fixed_seed_gate(
                _ordered(final_base, final_seeds),
                _ordered(final_candidate, final_seeds),
                policy=FORMAL_GATE_POLICY,
            )
        else:
            final50 = {
                "schema_version": "sts1-champion-gate-v1",
                "gate": "fixed-seed-50",
                "status": "SKIPPED",
                "reasons": ["final_30_seed_gate_did_not_pass"],
            }
        _save_cache(
            state,
            key="final_baseline",
            weight_sha=baseline_sha,
            seeds=final_seeds,
            mcts_sims=args.mcts_sims,
            simulator_id=simulator_id,
            runs=final_base,
        )
    else:
        final30 = {
            "schema_version": "sts1-champion-gate-v1",
            "gate": "fixed-seed-30",
            "status": "SKIPPED",
            "reasons": ["dev_parent_gate_did_not_pass"],
        }
        final50 = {
            "schema_version": "sts1-champion-gate-v1",
            "gate": "fixed-seed-50",
            "status": "SKIPPED",
            "reasons": ["dev_parent_gate_did_not_pass"],
        }

    ready = final30["status"] == "PASS" and final50["status"] == "PASS"
    report = {
        "schema_version": "sts1-armg-ppo-v14-two-level-gate-v1",
        "decision": dev_gate["decision"],
        "final_readiness": "READY_FOR_REAL_GAME" if ready else "NOT_READY",
        "production_champion_replaced": False,
        "real_game_gate_required_for_production": True,
        "parent_weight_sha256": parent_sha,
        "baseline_weight_sha256": baseline_sha,
        "candidate_weight_sha256": candidate_sha,
        "simulator_id": simulator_id,
        "mcts_sims": args.mcts_sims,
        "dev_gate": dev_gate,
        "final_gate_30": final30,
        "final_gate_50": final50,
    }

    state["v14_last_gate_preview"] = {
        "decision": report["decision"],
        "final_readiness": report["final_readiness"],
        "candidate_weight_sha256": candidate_sha,
    }
    args.state.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "gate-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PPO_V14_GATE_RESULT", json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
