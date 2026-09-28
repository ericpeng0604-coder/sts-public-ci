#!/usr/bin/env python3
"""Fast paired fixed-seed gate for the STS1 ArmG PPO v1.3 auto loop.

Safety invariants:
- Champion and Candidate use the exact same pinned held-out seeds.
- The 30-seed fast gate is unchanged; only PASS reaches the formal 50-seed gate.
- Champion results may be reused only when weight SHA, seed list, MCTS budget,
  and pinned simulator identity all match exactly.
- Missing evaluations run in parallel processes; result ordering is restored to
  the pinned seed order before the existing gate implementation is called.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from roguelike_ai.sts1_phase3.champion_gate import (
    FAST_GATE_POLICY,
    FORMAL_GATE_POLICY,
    evaluate_fixed_seed_gate,
)
from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)

CACHE_SCHEMA = "sts1-armg-ppo-v13-champion-eval-cache-v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _automatic_simulator_id(module_dir: Path) -> str:
    source = module_dir.parent
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    project_root = Path(__file__).resolve().parents[2]
    simulator_py = project_root / "src" / "roguelike_ai" / "sts1_phase3" / "simulator.py"
    gate_py = project_root / "src" / "roguelike_ai" / "sts1_phase3" / "champion_gate.py"
    return f"{head}:{_sha256(simulator_py)}:{_sha256(gate_py)}"


def _read_seeds(path: Path) -> list[int]:
    seeds = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise RuntimeError("formal seed file must contain exactly 50 unique seeds")
    return seeds


def _run_one(
    *,
    label: str,
    weight: Path,
    seed: int,
    module_dir: Path,
    armg_root: Path,
    evidence_root: Path,
    mcts_sims: int,
    heldout_seeds: list[int],
) -> dict[str, Any]:
    sts = _load_sts(module_dir)
    policy = ArmGNoncombatPolicy(root=armg_root, weight_path=weight)
    out = evidence_root / label
    out.mkdir(parents=True, exist_ok=True)
    result = run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=out / f"seed-{seed}.ndjson",
        armg_policy=policy,
        combat_mcts_sims=mcts_sims,
        heldout_seeds=heldout_seeds,
    )
    if result.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
        raise RuntimeError(f"{label} seed {seed} incomplete: {result}")
    return dict(result)


def _task(task: tuple[str, str, int, str, str, str, int, tuple[int, ...]]) -> tuple[str, int, dict[str, Any]]:
    label, weight, seed, module_dir, armg_root, evidence_root, mcts_sims, heldout = task
    result = _run_one(
        label=label,
        weight=Path(weight),
        seed=seed,
        module_dir=Path(module_dir),
        armg_root=Path(armg_root),
        evidence_root=Path(evidence_root),
        mcts_sims=mcts_sims,
        heldout_seeds=list(heldout),
    )
    return label, seed, result


def _evaluate_parallel(
    *,
    requests: list[tuple[str, Path, int]],
    module_dir: Path,
    armg_root: Path,
    output_dir: Path,
    mcts_sims: int,
    all_formal_seeds: list[int],
    workers: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    if not requests:
        return {}
    if workers < 1:
        raise RuntimeError("workers must be positive")
    tasks = [
        (
            label,
            str(weight),
            int(seed),
            str(module_dir),
            str(armg_root),
            str(output_dir),
            int(mcts_sims),
            tuple(all_formal_seeds),
        )
        for label, weight, seed in requests
    ]
    results: dict[tuple[str, int], dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        for label, seed, result in pool.map(_task, tasks):
            results[(label, int(seed))] = result
            print(
                "PPO_V13_GATE_GAME",
                json.dumps(
                    {
                        "label": label,
                        "seed": seed,
                        "outcome": result.get("outcome"),
                        "final_floor": result.get("final_floor"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return results


def _cache_identity(
    *,
    champion_sha: str,
    seeds: list[int],
    mcts_sims: int,
    simulator_id: str,
) -> dict[str, Any]:
    return {
        "champion_sha256": champion_sha,
        "formal_seeds": list(seeds),
        "mcts_sims": int(mcts_sims),
        "simulator_id": simulator_id,
    }


def _load_champion_cache(
    path: Path | None,
    *,
    state_path: Path | None,
    champion_sha: str,
    seeds: list[int],
    mcts_sims: int,
    simulator_id: str,
) -> dict[int, dict[str, Any]]:
    payload = None
    if path is not None and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif state_path is not None and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        payload = state.get("champion_eval_cache")
    if not isinstance(payload, dict):
        return {}
    if payload.get("schema_version") != CACHE_SCHEMA:
        print("PPO_V13_CHAMPION_CACHE_MISS schema", flush=True)
        return {}
    expected = _cache_identity(
        champion_sha=champion_sha,
        seeds=seeds,
        mcts_sims=mcts_sims,
        simulator_id=simulator_id,
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            print(f"PPO_V13_CHAMPION_CACHE_MISS {key}", flush=True)
            return {}

    cached: dict[int, dict[str, Any]] = {}
    allowed = set(seeds)
    for run in payload.get("runs", []):
        try:
            seed = int(run["seed"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("invalid champion cache seed")
        if seed not in allowed or seed in cached:
            raise RuntimeError("invalid/duplicate champion cache seed")
        if run.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
            raise RuntimeError("champion cache contains incomplete run")
        safety = (
            int(run.get("illegal_action_count", 0) or 0)
            + int(run.get("crash_count", 0) or 0)
            + int(run.get("timeout_count", 0) or 0)
        )
        if safety:
            raise RuntimeError("champion cache contains unsafe run")
        cached[seed] = dict(run)
    print(f"PPO_V13_CHAMPION_CACHE_HIT {len(cached)}/50", flush=True)
    return cached


def _write_champion_cache(
    path: Path | None,
    *,
    state_path: Path | None,
    champion_sha: str,
    seeds: list[int],
    mcts_sims: int,
    simulator_id: str,
    runs_by_seed: dict[int, dict[str, Any]],
) -> None:
    ordered = [runs_by_seed[s] for s in seeds if s in runs_by_seed]
    payload = {
        "schema_version": CACHE_SCHEMA,
        **_cache_identity(
            champion_sha=champion_sha,
            seeds=seeds,
            mcts_sims=mcts_sims,
            simulator_id=simulator_id,
        ),
        "runs": ordered,
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif state_path is not None and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["champion_eval_cache"] = payload
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"PPO_V13_CHAMPION_CACHE_WRITE {len(ordered)}/50", flush=True)


def _ordered(runs: dict[int, dict[str, Any]], seeds: Iterable[int]) -> list[dict[str, Any]]:
    values = []
    for seed in seeds:
        if seed not in runs:
            raise RuntimeError(f"missing run for seed {seed}")
        values.append(runs[seed])
    return values


def _write_runs(path: Path, runs: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--current-weight", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=Path, required=True)
    parser.add_argument("--formal-seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mcts-sims", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--current-cache", type=Path)
    parser.add_argument("--simulator-id", default="auto")
    args = parser.parse_args()

    seeds = _read_seeds(args.formal_seed_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_sha = _sha256(args.current_weight)
    candidate_sha = _sha256(args.candidate_weight)
    simulator_id = (
        _automatic_simulator_id(args.module_dir)
        if args.simulator_id == "auto"
        else args.simulator_id
    )
    state_path = args.current_weight.parent / "state.json"

    current_by_seed = _load_champion_cache(
        args.current_cache,
        state_path=state_path,
        champion_sha=current_sha,
        seeds=seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
    )
    candidate_by_seed: dict[int, dict[str, Any]] = {}

    first30 = seeds[:30]
    requests: list[tuple[str, Path, int]] = []
    requests.extend(
        ("current", args.current_weight, seed)
        for seed in first30
        if seed not in current_by_seed
    )
    requests.extend(("candidate", args.candidate_weight, seed) for seed in first30)
    fresh = _evaluate_parallel(
        requests=requests,
        module_dir=args.module_dir,
        armg_root=args.armg_root,
        output_dir=args.output_dir,
        mcts_sims=args.mcts_sims,
        all_formal_seeds=seeds,
        workers=args.workers,
    )
    for (label, seed), result in fresh.items():
        if label == "current":
            current_by_seed[seed] = result
        else:
            candidate_by_seed[seed] = result

    _write_champion_cache(
        args.current_cache,
        state_path=state_path,
        champion_sha=current_sha,
        seeds=seeds,
        mcts_sims=args.mcts_sims,
        simulator_id=simulator_id,
        runs_by_seed=current_by_seed,
    )

    current30 = _ordered(current_by_seed, first30)
    candidate30 = _ordered(candidate_by_seed, first30)
    gate30 = evaluate_fixed_seed_gate(current30, candidate30, policy=FAST_GATE_POLICY)

    if gate30["status"] == "PASS":
        tail = seeds[30:]
        requests = []
        requests.extend(
            ("current", args.current_weight, seed)
            for seed in tail
            if seed not in current_by_seed
        )
        requests.extend(("candidate", args.candidate_weight, seed) for seed in tail)
        fresh = _evaluate_parallel(
            requests=requests,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            output_dir=args.output_dir,
            mcts_sims=args.mcts_sims,
            all_formal_seeds=seeds,
            workers=args.workers,
        )
        for (label, seed), result in fresh.items():
            if label == "current":
                current_by_seed[seed] = result
            else:
                candidate_by_seed[seed] = result

        _write_champion_cache(
            args.current_cache,
            champion_sha=current_sha,
            seeds=seeds,
            mcts_sims=args.mcts_sims,
            simulator_id=args.simulator_id,
            runs_by_seed=current_by_seed,
        )
        current50 = _ordered(current_by_seed, seeds)
        candidate50 = _ordered(candidate_by_seed, seeds)
        gate50 = evaluate_fixed_seed_gate(current50, candidate50, policy=FORMAL_GATE_POLICY)
    else:
        gate50 = {
            "schema_version": "sts1-champion-gate-v1",
            "gate": "fixed-seed-50",
            "status": "SKIPPED",
            "reasons": ["fast_30_seed_gate_did_not_pass"],
        }

    decision = (
        "PROMOTE_OFFLINE"
        if gate30["status"] == "PASS" and gate50["status"] == "PASS"
        else "HOLD"
    )
    report = {
        "schema_version": "sts1-armg-ppo-v13-auto-gate-v2",
        "decision": decision,
        "production_champion_replaced": False,
        "real_game_gate_required_for_production": True,
        "current_weight_sha256": current_sha,
        "candidate_weight_sha256": candidate_sha,
        "mcts_sims": args.mcts_sims,
        "simulator_id": simulator_id,
        "parallel_workers": args.workers,
        "champion_cache_runs": len(current_by_seed),
        "gate_30": gate30,
        "gate_50": gate50,
    }

    _write_runs(args.output_dir / "current-runs.json", _ordered(current_by_seed, first30 if gate50["status"] == "SKIPPED" else seeds))
    _write_runs(args.output_dir / "candidate-runs.json", _ordered(candidate_by_seed, first30 if gate50["status"] == "SKIPPED" else seeds))
    (args.output_dir / "gate-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PPO_V13_AUTO_GATE_RESULT", json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
