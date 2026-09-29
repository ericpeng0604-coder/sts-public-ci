#!/usr/bin/env python3
"""Resumable STS1 Human Expert context v2 loop.

This loop is deliberately isolated from PPO and Strategy-only Champions.
Human data changes only non-combat card-reward preferences; combat stays frozen
pure MCTS. Promotion requires the same victory-first Dev-30 -> Hidden-50 ->
Fresh-100 gates used by the Strategy loop.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

from roguelike_ai.sts1_phase3.armg_strategy_evolve import (
    DEV_STRATEGY_GATE,
    FRESH_STRATEGY_GATE,
    HIDDEN_STRATEGY_GATE,
    evaluate_strategy_gate,
    strategy_promotion_decision,
)
from roguelike_ai.sts1_phase3.human_expert_context import (
    HumanExpertContextPolicy,
    build_context_prior,
)
from roguelike_ai.sts1_phase3.simulator import _load_sts, run_simulator_game


STATE_SCHEMA_VERSION = "sts1-human-expert-context-loop-state-v2"
REPORT_SCHEMA_VERSION = "sts1-human-expert-context-round-v2"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_seeds(path: Path) -> tuple[int, ...]:
    seeds = tuple(
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not seeds:
        raise RuntimeError(f"seed file is empty: {path}")
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"seed file contains duplicates: {path}")
    return seeds


def _fresh_seeds(
    *,
    count: int,
    rng_seed: int,
    forbidden: set[int],
) -> tuple[int, ...]:
    if count < 1:
        raise RuntimeError("fresh seed count must be positive")
    rng = random.Random(int(rng_seed))
    result: list[int] = []
    used = set(int(v) for v in forbidden)
    while len(result) < count:
        seed = rng.randint(1, 10**9)
        if seed in used:
            continue
        used.add(seed)
        result.append(seed)
    return tuple(result)


def _candidate_strengths(
    current: float,
    *,
    tried: Sequence[float] = (),
) -> tuple[float, ...]:
    """Return up to three untried strengths; never repeat a failed sweep."""
    current = float(current)
    if current < 0:
        raise RuntimeError("expert strength cannot be negative")
    tried_set = {round(float(value), 6) for value in tried}
    grid = [0.05, 0.10, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    if current == 0.0 and not tried_set:
        return (0.25, 0.50, 0.75)
    around = (
        max(0.05, current * 0.70) if current else 0.10,
        min(2.0, current * 1.25) if current else 1.00,
        min(2.0, current * 1.60) if current else 1.25,
    )
    pool = [round(float(value), 6) for value in (*around, *grid)]
    unique: list[float] = []
    for value in pool:
        if abs(value - current) < 1e-9 or value in tried_set or value in unique:
            continue
        unique.append(value)
    unique.sort(key=lambda value: (abs(value - current), value))
    return tuple(unique[:3])


def _load_or_init_state(state_dir: Path) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "human-expert-context-state.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise RuntimeError("human expert state schema mismatch")
        return payload
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "generation": 0,
        "accepted_rounds": 0,
        "rejected_rounds": 0,
        "current_strength": 0.0,
        "used_evaluation_seeds": [],
        "tried_strengths": [],
        "search_exhausted": False,
        "last_decision": None,
    }
    _write_json(path, payload)
    return payload


def _round_number(state: Mapping[str, Any]) -> int:
    return (
        int(state.get("accepted_rounds", 0))
        + int(state.get("rejected_rounds", 0))
        + 1
    )


def _evaluate_worker(
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
    module_dir = Path(module_dir_text)
    armg_root = Path(armg_root_text)
    sts = _load_sts(module_dir)
    policy = HumanExpertContextPolicy(
        root=armg_root,
        weight_path=Path(base_weight_text),
        expert_prior_path=Path(prior_path_text),
        expert_strength=float(strength),
    )
    result = run_simulator_game(
        student=None,
        sts=sts,
        seed=int(seed),
        armg_policy=policy,
        combat_mcts_sims=int(mcts_sims),
        heldout_seeds=all_seeds,
        collect_ppo=False,
        collect_teacher=False,
    )
    return dict(result)


def _evaluate_policy(
    *,
    seeds: Sequence[int],
    module_dir: Path,
    armg_root: Path,
    base_weight: Path,
    prior_path: Path,
    strength: float,
    mcts_sims: int,
    workers: int,
) -> dict[str, Any]:
    ordered = tuple(int(seed) for seed in seeds)
    if not ordered:
        raise RuntimeError("evaluation seed set is empty")
    if workers < 1:
        raise RuntimeError("eval workers must be positive")
    payloads = [
        (
            str(module_dir),
            str(armg_root),
            str(base_weight),
            str(prior_path),
            float(strength),
            seed,
            ordered,
            int(mcts_sims),
        )
        for seed in ordered
    ]
    if workers == 1:
        runs = [_evaluate_worker(payload) for payload in payloads]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(payloads)),
            mp_context=context,
        ) as pool:
            runs = list(pool.map(_evaluate_worker, payloads))
    return {"runs": runs}


def _gate_rank(gate: Mapping[str, Any]) -> tuple[float, ...]:
    paired = gate.get("paired_wins") or gate.get("paired") or {}
    return (
        1.0 if gate.get("status") == "PASS" else 0.0,
        float(gate.get("win_delta", -10**9) or 0.0),
        float(paired.get("candidate_better", 0) or 0)
        - float(paired.get("candidate_worse", 0) or 0),
        -float(paired.get("one_sided_sign_p", 1.0) or 1.0),
    )


def _eval_and_gate(
    *,
    current_prior: Path,
    candidate_prior: Path,
    current_strength: float,
    candidate_strength: float,
    seeds: Sequence[int],
    module_dir: Path,
    armg_root: Path,
    base_weight: Path,
    mcts_sims: int,
    workers: int,
    policy: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    current_eval = _evaluate_policy(
        seeds=seeds,
        module_dir=module_dir,
        armg_root=armg_root,
        base_weight=base_weight,
        prior_path=current_prior,
        strength=current_strength,
        mcts_sims=mcts_sims,
        workers=workers,
    )
    candidate_eval = _evaluate_policy(
        seeds=seeds,
        module_dir=module_dir,
        armg_root=armg_root,
        base_weight=base_weight,
        prior_path=candidate_prior,
        strength=candidate_strength,
        mcts_sims=mcts_sims,
        workers=workers,
    )
    gate = evaluate_strategy_gate(
        current_eval,
        candidate_eval,
        policy=policy,
        expected_combat_policy=f"mcts_{mcts_sims}",
    )
    return current_eval, candidate_eval, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--armg-base-weight", type=Path, required=True)
    parser.add_argument("--dev-seed-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--combat-mcts-sims", type=int, default=2000)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--rng-seed", type=int, default=2026092901)
    parser.add_argument("--min-ascension", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--min-context-support", type=float, default=6.0)
    args = parser.parse_args()

    if args.rounds < 1:
        raise RuntimeError("rounds must be positive")
    if args.combat_mcts_sims < 1 or args.eval_workers < 1:
        raise RuntimeError("MCTS and worker counts must be positive")
    if not args.armg_base_weight.is_file():
        raise RuntimeError(f"ArmG base weight missing: {args.armg_base_weight}")

    dev_all = _read_seeds(args.dev_seed_file)
    if len(dev_all) != 50:
        raise RuntimeError("human expert gate requires exactly 50 frozen dev seeds")
    dev30 = dev_all[:30]

    state = _load_or_init_state(args.state_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    for _ in range(args.rounds):
        round_no = _round_number(state)
        round_dir = args.output_dir / f"round-{round_no:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        candidate_prior = round_dir / "candidate-expert-prior.json"
        dataset_report = build_context_prior(
            args.data_root,
            candidate_prior,
            min_ascension=args.min_ascension,
            alpha=args.alpha,
            min_context_support=args.min_context_support,
        )
        if float(dataset_report.get("context_test_delta", -1.0)) < 0.0:
            raise RuntimeError(
                "context expert prior underperformed the non-context baseline "
                f"on held-out human decisions: delta={dataset_report.get('context_test_delta')}"
            )

        current_prior = args.state_dir / "current-expert-context-prior.json"
        if not current_prior.is_file():
            # Generation zero is pure ArmG because current_strength=0.  Reusing
            # the candidate JSON only supplies a valid schema to the loader.
            current_prior = candidate_prior

        current_strength = float(state.get("current_strength", 0.0) or 0.0)
        current_dev = _evaluate_policy(
            seeds=dev30,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            base_weight=args.armg_base_weight,
            prior_path=current_prior,
            strength=current_strength,
            mcts_sims=args.combat_mcts_sims,
            workers=args.eval_workers,
        )
        _write_json(round_dir / "eval-current-dev30.json", current_dev)

        strengths = _candidate_strengths(
            current_strength,
            tried=state.get("tried_strengths", []),
        )
        if not strengths:
            state["search_exhausted"] = True
            state["last_decision"] = {
                "round": round_no,
                "decision": "PAUSE_SEARCH_EXHAUSTED",
                "candidate_strength": None,
                "dev": "SKIPPED",
                "hidden": "SKIPPED",
                "fresh": "SKIPPED",
            }
            _write_json(args.state_dir / "human-expert-context-state.json", state)
            report = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "round": round_no,
                "generation_before": int(state.get("generation", 0)),
                "generation_after": int(state.get("generation", 0)),
                "dataset": dataset_report,
                "current_strength_before": current_strength,
                "candidate_strengths": [],
                "selected_strength": None,
                "dev_gate": {"status": "SKIPPED", "reasons": ["strength_search_exhausted"]},
                "hidden_gate": {"status": "SKIPPED", "reasons": ["strength_search_exhausted"]},
                "fresh_gate": {"status": "SKIPPED", "reasons": ["strength_search_exhausted"]},
                "promotion": {"decision": "PAUSE_SEARCH_EXHAUSTED"},
                "production_champion_replaced": False,
                "combat_training_enabled": False,
                "combat_policy": f"mcts_{args.combat_mcts_sims}",
                "hidden_seed_count": 0,
                "fresh_seed_count": 0,
            }
            _write_json(round_dir / "human-expert-context-report.json", report)
            _write_json(args.output_dir / "latest-report.json", report)
            reports.append(report)
            break

        candidates: list[dict[str, Any]] = []
        for strength in strengths:
            candidate_eval = _evaluate_policy(
                seeds=dev30,
                module_dir=args.module_dir,
                armg_root=args.armg_root,
                base_weight=args.armg_base_weight,
                prior_path=candidate_prior,
                strength=strength,
                mcts_sims=args.combat_mcts_sims,
                workers=args.eval_workers,
            )
            gate = evaluate_strategy_gate(
                current_dev,
                candidate_eval,
                policy=DEV_STRATEGY_GATE,
                expected_combat_policy=f"mcts_{args.combat_mcts_sims}",
            )
            label = str(strength).replace(".", "p")
            _write_json(round_dir / f"eval-candidate-dev30-{label}.json", candidate_eval)
            candidates.append(
                {
                    "strength": float(strength),
                    "dev_gate": gate,
                    "candidate_eval": candidate_eval,
                }
            )

        selected = max(candidates, key=lambda row: _gate_rank(row["dev_gate"]))
        candidate_strength = float(selected["strength"])
        dev_gate = dict(selected["dev_gate"])

        historical = {int(v) for v in state.get("used_evaluation_seeds", [])}
        forbidden = set(dev_all) | historical
        hidden_seeds: tuple[int, ...] = ()
        fresh_seeds: tuple[int, ...] = ()
        hidden_gate: dict[str, Any] = {
            "status": "SKIPPED",
            "reasons": ["dev_30_gate_did_not_pass"],
        }
        fresh_gate: dict[str, Any] = {
            "status": "SKIPPED",
            "reasons": ["hidden_50_gate_not_passed"],
        }

        if dev_gate.get("status") == "PASS":
            hidden_seeds = _fresh_seeds(
                count=50,
                rng_seed=args.rng_seed + round_no * 1009 + 1,
                forbidden=forbidden,
            )
            forbidden.update(hidden_seeds)
            current_hidden, candidate_hidden, hidden_gate = _eval_and_gate(
                current_prior=current_prior,
                candidate_prior=candidate_prior,
                current_strength=current_strength,
                candidate_strength=candidate_strength,
                seeds=hidden_seeds,
                module_dir=args.module_dir,
                armg_root=args.armg_root,
                base_weight=args.armg_base_weight,
                mcts_sims=args.combat_mcts_sims,
                workers=args.eval_workers,
                policy=HIDDEN_STRATEGY_GATE,
            )
            _write_json(round_dir / "eval-current-hidden50.json", current_hidden)
            _write_json(round_dir / "eval-candidate-hidden50.json", candidate_hidden)

        if hidden_gate.get("status") == "PASS":
            fresh_seeds = _fresh_seeds(
                count=100,
                rng_seed=args.rng_seed + round_no * 1009 + 2,
                forbidden=forbidden,
            )
            forbidden.update(fresh_seeds)
            current_fresh, candidate_fresh, fresh_gate = _eval_and_gate(
                current_prior=current_prior,
                candidate_prior=candidate_prior,
                current_strength=current_strength,
                candidate_strength=candidate_strength,
                seeds=fresh_seeds,
                module_dir=args.module_dir,
                armg_root=args.armg_root,
                base_weight=args.armg_base_weight,
                mcts_sims=args.combat_mcts_sims,
                workers=args.eval_workers,
                policy=FRESH_STRATEGY_GATE,
            )
            _write_json(round_dir / "eval-current-fresh100.json", current_fresh)
            _write_json(round_dir / "eval-candidate-fresh100.json", candidate_fresh)

        promotion = strategy_promotion_decision(dev_gate, hidden_gate, fresh_gate)
        promoted = promotion.get("decision") == "PROMOTE_STRATEGY"
        state["used_evaluation_seeds"] = list(
            dict.fromkeys(
                list(state.get("used_evaluation_seeds", []))
                + list(hidden_seeds)
                + list(fresh_seeds)
            )
        )
        tried = list(state.get("tried_strengths", []))
        tried.extend(float(row["strength"]) for row in candidates)
        if promoted:
            shutil.copy2(candidate_prior, args.state_dir / "current-expert-context-prior.json")
            state["generation"] = int(state.get("generation", 0)) + 1
            state["accepted_rounds"] = int(state.get("accepted_rounds", 0)) + 1
            state["current_strength"] = candidate_strength
            state["current_corpus_sha256"] = dataset_report["corpus_sha256"]
            state["tried_strengths"] = []
            state["search_exhausted"] = False
        else:
            state["rejected_rounds"] = int(state.get("rejected_rounds", 0)) + 1
            state["tried_strengths"] = list(dict.fromkeys(round(v, 6) for v in tried))
            state["search_exhausted"] = not bool(
                _candidate_strengths(
                    current_strength,
                    tried=state["tried_strengths"],
                )
            )

        state["last_decision"] = {
            "round": round_no,
            "decision": promotion.get("decision"),
            "candidate_strength": candidate_strength,
            "dev": dev_gate.get("status"),
            "hidden": hidden_gate.get("status"),
            "fresh": fresh_gate.get("status"),
        }
        _write_json(args.state_dir / "human-expert-context-state.json", state)

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "round": round_no,
            "generation_before": int(state.get("generation", 0)) - int(promoted),
            "generation_after": int(state.get("generation", 0)),
            "dataset": dataset_report,
            "current_strength_before": current_strength,
            "candidate_strengths": [
                {
                    "strength": row["strength"],
                    "dev_gate": row["dev_gate"],
                    "selected": row is selected,
                }
                for row in candidates
            ],
            "selected_strength": candidate_strength,
            "dev_gate": dev_gate,
            "hidden_gate": hidden_gate,
            "fresh_gate": fresh_gate,
            "promotion": promotion,
            "production_champion_replaced": False,
            "combat_training_enabled": False,
            "expert_context_version": 2,
            "combat_policy": f"mcts_{args.combat_mcts_sims}",
            "hidden_seed_count": len(hidden_seeds),
            "fresh_seed_count": len(fresh_seeds),
        }
        _write_json(round_dir / "human-expert-context-report.json", report)
        _write_json(args.output_dir / "latest-report.json", report)
        reports.append(report)

    print(json.dumps(reports[-1], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
