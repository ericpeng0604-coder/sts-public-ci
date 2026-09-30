#!/usr/bin/env python3
"""Resumable STS1 Strategy-only self-improvement loop.

The only trainable component is ArmG's non-combat scorer. Every battle is
resolved by one frozen pure-MCTS budget, so combat learning cannot contaminate
Strategy comparisons.

Each round:
1. collect fresh non-combat branch-rollout labels;
2. merge them into a prioritized, kind-balanced replay buffer;
3. train a Strategy candidate from the current Strategy Champion;
4. paired Dev-30 gate on the stable development seeds;
5. paired Hidden-50 gate on never-before-used seeds;
6. paired Fresh-100 gate on another never-before-used seed set;
7. replace only the offline Strategy Champion if every gate passes.

This script never replaces the production/real-game Champion.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import math
import multiprocessing as mp
import json
import os
import time
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

import torch

from roguelike_ai.sts1_phase3.armg_strategy_evolve import (
    DEV_STRATEGY_GATE,
    FRESH_STRATEGY_GATE,
    HIDDEN_STRATEGY_GATE,
    STRATEGY_DATASET_SCHEMA_VERSION,
    branch_quality,
    evaluate_strategy_gate,
    soft_branch_targets,
    strategy_example_priority,
    strategy_promotion_decision,
)
from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    SimulatorRunError,
    _load_sts,
    _set_pauses,
    run_simulator_game,
)


STATE_SCHEMA_VERSION = "sts1-armg-strategy-loop-state-v1"
ROUND_SCHEMA_VERSION = "sts1-armg-strategy-round-v1"
EVAL_SCHEMA_VERSION = "sts1-armg-strategy-eval-v1"
DATA_EFFICIENCY_VERSION = 5
ELITE_POOL_SCHEMA_VERSION = "sts1-strategy-elite-pool-v1"
CANDIDATE_RECIPES = (
    {"name": "balanced", "min_teacher_confidence": 0.0, "step_scale": 0.25},
    {"name": "confident", "min_teacher_confidence": 0.50, "step_scale": 0.50},
    {"name": "strict", "min_teacher_confidence": 0.75, "step_scale": 1.00},
)
RESCUE_CANDIDATE_RECIPES = (
    {"name": "rescue_micro", "min_teacher_confidence": 0.0, "step_scale": 0.10},
    {"name": "rescue_confident", "min_teacher_confidence": 0.50, "step_scale": 0.20},
    {"name": "rescue_strict", "min_teacher_confidence": 0.75, "step_scale": 0.35},
)


def _candidate_recipes_for_stagnation(stagnation_count: int) -> tuple[dict[str, Any], ...]:
    if stagnation_count < 0:
        raise RuntimeError("stagnation count cannot be negative")
    return RESCUE_CANDIDATE_RECIPES if stagnation_count >= 12 else CANDIDATE_RECIPES
AMBIGUITY_PARENT_DISTILL_COEF = 1.0
PARALLEL_CPU_VERSION = 1

_TEACHER_PARALLEL_SELECTED: Sequence[Mapping[str, Any]] | None = None
_TEACHER_PARALLEL_STS: Any = None
_TEACHER_PARALLEL_ARMG: Any = None
_TEACHER_PARALLEL_CONFIG: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


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
    rng = random.Random(rng_seed)
    result: list[int] = []
    used = set(forbidden)
    while len(result) < count:
        value = rng.randint(1, 10**9)
        if value in used:
            continue
        used.add(value)
        result.append(value)
    return tuple(result)


def _desc_rows(descs: Sequence[Any]) -> list[list[float]]:
    result: list[list[float]] = []
    for desc in descs:
        try:
            row = [float(value) for value in desc]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"ArmG descriptor is not numeric: {exc}") from exc
        if not row:
            raise RuntimeError("ArmG emitted an empty descriptor")
        result.append(row)
    return result


def _choice_snapshot(gc: Any, armg: ArmGNoncombatPolicy) -> dict[str, Any]:
    kind, descs, _ = armg.choices(gc)
    return {
        "seed": int(getattr(gc, "seed", 0) or 0),
        "floor": int(getattr(gc, "floor_num", 0) or 0),
        "act": int(getattr(gc, "act", 0) or 0),
        "hp": int(getattr(gc, "cur_hp", 0) or 0),
        "max_hp": int(getattr(gc, "max_hp", 0) or 0),
        "gold": int(getattr(gc, "gold", 0) or 0),
        "screen": str(getattr(gc, "screen_state", "")),
        "obs": [float(value) for value in armg.module.obs_vec(gc)],
        "kind": str(kind),
        "descs": _desc_rows(descs),
    }


def _choice_uncertainty(scores: Any) -> dict[str, float]:
    values = [float(v) for v in scores.tolist()]
    if len(values) < 2:
        return {
            "entropy": 0.0,
            "top_probability": 1.0,
            "probability_margin": 1.0,
            "uncertainty": 0.0,
        }
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    total = sum(weights)
    probs = [w / total for w in weights]
    ordered = sorted(probs, reverse=True)
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    entropy /= math.log(len(probs))
    probability_margin = ordered[0] - ordered[1]
    uncertainty = 0.65 * entropy + 0.35 * (1.0 - ordered[0])
    return {
        "entropy": float(entropy),
        "top_probability": float(ordered[0]),
        "probability_margin": float(probability_margin),
        "uncertainty": float(uncertainty),
    }


def _state_bucket(snapshot: Mapping[str, Any], *, choices: int) -> str:
    max_hp = max(1, int(snapshot.get("max_hp", 1) or 1))
    hp = max(0, int(snapshot.get("hp", 0) or 0))
    hp_bucket = min(4, int(5 * hp / max_hp))
    choice_bucket = "4+" if choices >= 4 else str(choices)
    return (
        f"{snapshot.get('kind','unknown')}|act{int(snapshot.get('act',0) or 0)}"
        f"|f{int(snapshot.get('floor',0) or 0)//10}|hp{hp_bucket}|c{choice_bucket}"
    )


def _prelabel_priority(
    snapshot: Mapping[str, Any],
    uncertainty: Mapping[str, float],
    *,
    choices: int,
) -> float:
    floor = max(0, int(snapshot.get("floor", 0) or 0))
    max_hp = max(1, int(snapshot.get("max_hp", 1) or 1))
    hp = max(0, int(snapshot.get("hp", 0) or 0))
    late_bonus = min(1.0, floor / 50.0)
    danger_bonus = 1.0 - min(1.0, hp / max_hp)
    complexity_bonus = min(1.0, max(0, choices - 2) / 4.0)
    return float(
        2.0 * float(uncertainty.get("uncertainty", 0.0))
        + 0.45 * late_bonus
        + 0.35 * danger_bonus
        + 0.20 * complexity_bonus
    )


def _teacher_confidence_weight(margin: float) -> float:
    """Down-weight ambiguous Teacher labels without weakening evaluation gates."""
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0:
        raise RuntimeError("teacher margin must be finite and non-negative")
    if margin <= 0.25:
        return 0.10
    if margin <= 1.0:
        return 0.25
    if margin <= 3.0:
        return 0.50
    if margin <= 8.0:
        return 0.75
    return 1.00


def _teacher_mix_weight(confidence: float, *, min_teacher_confidence: float) -> float:
    confidence = float(confidence)
    min_teacher_confidence = float(min_teacher_confidence)
    if not 0.0 <= confidence <= 1.0:
        raise RuntimeError("teacher confidence must be in [0, 1]")
    if not 0.0 <= min_teacher_confidence <= 1.0:
        raise RuntimeError("minimum teacher confidence must be in [0, 1]")
    return confidence if confidence >= min_teacher_confidence else 0.0


def _select_teacher_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    min_per_kind: int,
) -> list[Mapping[str, Any]]:
    if budget < 1:
        raise RuntimeError("teacher label budget must be positive")
    if min_per_kind < 0:
        raise RuntimeError("min labels per kind must be non-negative")

    dedup: dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        identity = str(row["identity"])
        old = dedup.get(identity)
        if old is None or float(row["prelabel_priority"]) > float(old["prelabel_priority"]):
            dedup[identity] = row

    pool = list(dedup.values())
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool:
        groups[str(row["kind"])].append(row)
    for rows in groups.values():
        rows.sort(
            key=lambda row: (
                float(row["prelabel_priority"]),
                float((row.get("uncertainty") or {}).get("uncertainty", 0.0)),
                int(row.get("floor", 0) or 0),
            ),
            reverse=True,
        )

    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    selected_buckets: set[str] = set()

    # First guarantee kind coverage, preferring new state buckets inside each kind.
    for _ in range(min_per_kind):
        for kind in sorted(groups):
            if len(selected) >= budget:
                break
            rows = groups[kind]
            choice = next(
                (
                    row
                    for row in rows
                    if row["identity"] not in selected_ids
                    and row["state_bucket"] not in selected_buckets
                ),
                None,
            )
            if choice is None:
                choice = next(
                    (row for row in rows if row["identity"] not in selected_ids),
                    None,
                )
            if choice is None:
                continue
            selected.append(choice)
            selected_ids.add(str(choice["identity"]))
            selected_buckets.add(str(choice["state_bucket"]))

    # Fill remaining budget globally. Novel buckets get a small diversity boost.
    while len(selected) < budget:
        remaining = [row for row in pool if str(row["identity"]) not in selected_ids]
        if not remaining:
            break
        choice = max(
            remaining,
            key=lambda row: (
                float(row["prelabel_priority"])
                + (0.30 if str(row["state_bucket"]) not in selected_buckets else 0.0),
                float((row.get("uncertainty") or {}).get("uncertainty", 0.0)),
                int(row.get("floor", 0) or 0),
            ),
        )
        selected.append(choice)
        selected_ids.add(str(choice["identity"]))
        selected_buckets.add(str(choice["state_bucket"]))

    return selected


def _elite_candidate_decision(
    fresh_gate: Mapping[str, Any],
    *,
    promoted: bool,
    max_sign_p: float,
    min_win_delta: int,
    min_floor_delta: float,
) -> dict[str, Any]:
    """Classify a near-miss Candidate without weakening the Champion gate."""
    if promoted:
        return {"eligible": False, "reason": "already_promoted"}
    if fresh_gate.get("status") == "SKIPPED":
        return {"eligible": False, "reason": "fresh_gate_not_reached"}

    current = fresh_gate.get("current") or {}
    candidate = fresh_gate.get("candidate") or {}
    paired = fresh_gate.get("paired") or {}
    reasons = set(str(v) for v in (fresh_gate.get("reasons") or []))
    forbidden_reasons = {
        "incomplete_eval",
        "current_safety_failure",
        "candidate_safety_failure",
        "combat_policy_not_frozen_pure_mcts",
    }
    if reasons & forbidden_reasons:
        return {
            "eligible": False,
            "reason": "unsafe_or_incomplete",
            "blocking_reasons": sorted(reasons & forbidden_reasons),
        }

    win_delta = int(fresh_gate.get("win_delta", -10**9))
    floor_delta_raw = fresh_gate.get("floor_delta")
    floor_delta = (
        float(floor_delta_raw)
        if isinstance(floor_delta_raw, (int, float)) and not isinstance(floor_delta_raw, bool)
        else None
    )
    better = int(paired.get("candidate_better", 0) or 0)
    worse = int(paired.get("candidate_worse", 0) or 0)
    sign_p = float(paired.get("one_sided_sign_p", 1.0) or 1.0)
    expected = int(fresh_gate.get("expected_seed_count", 0) or 0)
    complete = (
        int(current.get("complete_runs", 0) or 0) == expected
        and int(candidate.get("complete_runs", 0) or 0) == expected
        and expected > 0
    )
    eligible = (
        complete
        and win_delta >= min_win_delta
        and floor_delta is not None
        and floor_delta >= min_floor_delta
        and better > worse
        and sign_p <= max_sign_p
    )
    return {
        "eligible": bool(eligible),
        "reason": "near_miss_elite" if eligible else "not_strong_enough",
        "win_delta": win_delta,
        "floor_delta": floor_delta,
        "candidate_better": better,
        "candidate_worse": worse,
        "one_sided_sign_p": sign_p,
        "max_sign_p": float(max_sign_p),
        "min_win_delta": int(min_win_delta),
        "min_floor_delta": float(min_floor_delta),
    }


def _elite_score(entry: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = entry.get("metrics") or {}
    win_delta = float(metrics.get("win_delta", 0.0) or 0.0)
    floor_delta = float(metrics.get("floor_delta", 0.0) or 0.0)
    better = float(metrics.get("candidate_better", 0.0) or 0.0)
    worse = float(metrics.get("candidate_worse", 0.0) or 0.0)
    sign_p = float(metrics.get("one_sided_sign_p", 1.0) or 1.0)
    return (
        10.0 * win_delta + floor_delta + 0.1 * (better - worse) - sign_p,
        win_delta,
        -sign_p,
    )


def _load_elite_pool(state_dir: Path) -> list[dict[str, Any]]:
    pool_path = state_dir / "elite-pool.json"
    if not pool_path.is_file():
        _write_json(
            pool_path,
            {"schema_version": ELITE_POOL_SCHEMA_VERSION, "entries": []},
        )
        return []
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ELITE_POOL_SCHEMA_VERSION:
        raise RuntimeError("Strategy Elite pool schema mismatch")
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise RuntimeError("Strategy Elite pool entries must be a list")
    valid: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Strategy Elite pool entry must be an object")
        entry = dict(raw)
        rel = str(entry.get("weight_file", ""))
        sha = str(entry.get("sha256", ""))
        weight = state_dir / rel
        if not rel or not weight.is_file() or len(sha) != 64 or _sha256(weight) != sha:
            raise RuntimeError(f"Strategy Elite pool checkpoint invalid: {rel!r}")
        valid.append(entry)
    return valid


def _write_elite_pool(state_dir: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    _write_json(
        state_dir / "elite-pool.json",
        {
            "schema_version": ELITE_POOL_SCHEMA_VERSION,
            "entries": [dict(row) for row in entries],
        },
    )


def _save_elite_candidate(
    *,
    state_dir: Path,
    candidate_weight: Path,
    round_no: int,
    parent_champion_sha: str,
    metrics: Mapping[str, Any],
    max_entries: int,
) -> dict[str, Any]:
    if max_entries < 1:
        raise RuntimeError("elite max entries must be positive")
    entries = _load_elite_pool(state_dir)
    elite_dir = state_dir / "elite-candidates"
    elite_dir.mkdir(parents=True, exist_ok=True)
    candidate_sha = _sha256(candidate_weight)
    rel = f"elite-candidates/elite-round-{round_no:04d}-{candidate_sha[:12]}.pt"
    target = state_dir / rel
    shutil.copy2(candidate_weight, target)
    entry = {
        "round": int(round_no),
        "sha256": candidate_sha,
        "weight_file": rel,
        "parent_champion_sha256": str(parent_champion_sha),
        "metrics": dict(metrics),
    }
    entries = [row for row in entries if row.get("sha256") != candidate_sha]
    entries.append(entry)
    entries.sort(key=_elite_score, reverse=True)
    kept = entries[:max_entries]
    keep_files = {str(row["weight_file"]) for row in kept}
    for path in elite_dir.glob("*.pt"):
        relpath = str(path.relative_to(state_dir))
        if relpath not in keep_files:
            path.unlink()
    _write_elite_pool(state_dir, kept)
    return {
        "saved": True,
        "entry": entry,
        "pool_size": len(kept),
        "best_sha256": kept[0]["sha256"] if kept else None,
    }


def _clear_elite_pool(state_dir: Path) -> None:
    elite_dir = state_dir / "elite-candidates"
    if elite_dir.is_dir():
        shutil.rmtree(elite_dir)
    _write_elite_pool(state_dir, [])


def _elite_verified_example(
    example: Mapping[str, Any],
    *,
    elite_index: int,
    elite_sha256: str,
    min_quality_margin: float,
) -> dict[str, Any] | None:
    current_index = int(example["current_armg_index"])
    teacher_index = int(example["teacher_best_index"])
    values = [float(v) for v in example["branch_quality"]]
    if not (0 <= elite_index < len(values)):
        raise RuntimeError("elite index outside branch values")
    if elite_index == current_index or teacher_index != elite_index:
        return None
    margin = values[elite_index] - values[current_index]
    if margin < min_quality_margin:
        return None
    row = dict(example)
    row["source"] = "elite_candidate_verified_disagreement"
    row["elite_candidate_sha256"] = str(elite_sha256)
    row["elite_candidate_index"] = int(elite_index)
    row["elite_quality_margin_over_champion"] = float(margin)
    row["priority"] = min(
        5.0,
        max(float(row.get("priority", 1.0)), 3.5 + min(1.0, margin / 15.0)),
    )
    return row


def _drive_pure_mcts_battle(
    gc: Any,
    *,
    sts: Any,
    mcts_sims: int,
    max_battle_steps: int,
) -> None:
    """Resolve one battle with no Student/model vote of any kind."""
    battle = sts.BattleContext()
    battle.init(gc)
    steps = 0
    while battle.outcome == sts.Outcome.UNDECIDED and steps < max_battle_steps:
        steps += 1
        legal = list(sts.get_legal_actions(battle))
        if not legal:
            raise RuntimeError("pure-MCTS branch battle exposed no legal action")
        if len(legal) == 1:
            chosen = legal[0]
        else:
            chosen = sts.mcts_recommend(battle, int(mcts_sims))
            if chosen is None:
                raise RuntimeError("pure-MCTS branch battle returned no recommendation")
        chosen.execute(battle)

    if battle.outcome == sts.Outcome.UNDECIDED:
        raise RuntimeError("pure-MCTS branch battle step bound reached")
    battle.exit_battle(gc)


def _play_from_state(
    gc: Any,
    *,
    sts: Any,
    armg: ArmGNoncombatPolicy,
    mcts_sims: int,
    max_game_steps: int,
    max_battle_steps: int,
) -> dict[str, Any]:
    """Finish a cloned game under current Strategy + frozen pure MCTS."""
    agent = sts.Agent()
    _set_pauses(agent)
    steps = 0
    while gc.outcome == sts.GameOutcome.UNDECIDED and steps < max_game_steps:
        steps += 1
        agent.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        if gc.screen_state == sts.ScreenState.BATTLE:
            _drive_pure_mcts_battle(
                gc,
                sts=sts,
                mcts_sims=mcts_sims,
                max_battle_steps=max_battle_steps,
            )
        else:
            armg.step(gc, sts)

    if gc.outcome == sts.GameOutcome.UNDECIDED:
        raise RuntimeError("strategy branch game step bound reached")
    return {
        "outcome": (
            "victory"
            if gc.outcome == sts.GameOutcome.PLAYER_VICTORY
            else "defeat"
        ),
        "final_floor": int(getattr(gc, "floor_num", 0) or 0),
        "final_hp": int(getattr(gc, "cur_hp", 0) or 0),
        "max_hp": int(getattr(gc, "max_hp", 1) or 1),
    }


def _branch_example(
    gc: Any,
    *,
    sts: Any,
    armg: ArmGNoncombatPolicy,
    mcts_sims: int,
    max_game_steps: int,
    max_battle_steps: int,
    temperature: float,
) -> dict[str, Any]:
    """Evaluate every legal ArmG choice by cloning and playing to terminal."""
    if not callable(getattr(gc, "clone", None)):
        raise RuntimeError("pinned simulator does not expose GameContext.clone()")

    before = _choice_snapshot(gc, armg)
    kind, descs, _ = armg.choices(gc)
    desc_rows = _desc_rows(descs)
    if len(desc_rows) < 2:
        raise RuntimeError("strategy branch example requires at least two choices")

    probe = gc.clone()
    if _choice_snapshot(probe, armg) != before:
        raise RuntimeError("GameContext.clone non-combat parity mismatch")

    _, _, scores = armg.score_choices(gc)
    current_index = int(torch.argmax(scores).item())
    branches: list[dict[str, Any]] = []
    values: list[float] = []

    for index in range(len(desc_rows)):
        branch = gc.clone()
        branch_kind, branch_descs, branch_execs = armg.choices(branch)
        branch_desc_rows = _desc_rows(branch_descs)
        if str(branch_kind) != str(kind) or branch_desc_rows != desc_rows:
            raise RuntimeError(
                f"cloned ArmG choice identity drift at index={index}: "
                f"kind={branch_kind!r} expected={kind!r}"
            )
        if len(branch_execs) != len(desc_rows):
            raise RuntimeError("cloned ArmG descriptor/executor length mismatch")

        branch_execs[index](branch)
        terminal = _play_from_state(
            branch,
            sts=sts,
            armg=armg,
            mcts_sims=mcts_sims,
            max_game_steps=max_game_steps,
            max_battle_steps=max_battle_steps,
        )
        value = branch_quality(
            outcome=terminal["outcome"],
            final_floor=terminal["final_floor"],
            final_hp=terminal["final_hp"],
            max_hp=terminal["max_hp"],
        )
        values.append(value)
        branches.append(
            {
                "index": index,
                **terminal,
                "quality": value,
            }
        )

    after = _choice_snapshot(gc, armg)
    if after != before:
        raise RuntimeError("strategy branch rollout mutated original GameContext")

    targets = soft_branch_targets(values, temperature=temperature)
    best_index = max(range(len(values)), key=values.__getitem__)
    priority = strategy_example_priority(
        current_index=current_index,
        teacher_best_index=best_index,
        branch_values=values,
    )
    ordered_values = sorted(values, reverse=True)
    teacher_margin = ordered_values[0] - ordered_values[1]
    confidence_weight = _teacher_confidence_weight(teacher_margin)
    return {
        "schema_version": STRATEGY_DATASET_SCHEMA_VERSION,
        "type": "armg_strategy_branch_example",
        "seed": int(getattr(gc, "seed", 0) or 0),
        "floor": int(getattr(gc, "floor_num", 0) or 0),
        "act": int(getattr(gc, "act", 0) or 0),
        "kind": str(kind),
        "obs": before["obs"],
        "descs": desc_rows,
        "current_armg_index": current_index,
        "teacher_best_index": best_index,
        "branch_quality": values,
        "target_probs": list(targets),
        "priority": priority,
        "teacher_margin": teacher_margin,
        "confidence_weight": confidence_weight,
        "branches": branches,
        "combat_policy": f"mcts_{mcts_sims}",
    }



def _teacher_parallel_process(
    worker_id: int,
    indices: Sequence[int],
    result_queue: Any,
) -> None:
    selected = _TEACHER_PARALLEL_SELECTED
    config = _TEACHER_PARALLEL_CONFIG
    worker_started = time.perf_counter()
    worker_cpu_started = time.process_time()
    try:
        if (
            selected is None
            or config is None
            or _TEACHER_PARALLEL_STS is None
            or _TEACHER_PARALLEL_ARMG is None
        ):
            raise RuntimeError("parallel Teacher worker context is not initialized")
        torch.set_num_threads(1)
        for raw_index in indices:
            index = int(raw_index)
            candidate = selected[index]
            row = _branch_example(
                candidate["gc"],
                sts=_TEACHER_PARALLEL_STS,
                armg=_TEACHER_PARALLEL_ARMG,
                mcts_sims=int(config["mcts_sims"]),
                max_game_steps=int(config["max_game_steps"]),
                max_battle_steps=int(config["max_battle_steps"]),
                temperature=float(config["temperature"]),
            )
            result_queue.put(("result", worker_id, index, row))
        result_queue.put(
            (
                "done",
                worker_id,
                None,
                {
                    "busy_seconds": time.perf_counter() - worker_started,
                    "cpu_seconds": time.process_time() - worker_cpu_started,
                },
            )
        )
    except BaseException as exc:
        result_queue.put(
            ("error", worker_id, None, f"{type(exc).__name__}:{exc}")
        )


def _label_selected_candidates(
    selected: Sequence[Mapping[str, Any]],
    *,
    sts: Any,
    armg: ArmGNoncombatPolicy,
    mcts_sims: int,
    max_game_steps: int,
    max_battle_steps: int,
    temperature: float,
    collection_workers: int,
    parallel_timeout_seconds: int,
    parallel_parity_probes: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label independent Teacher states with rolling forked processes."""
    if collection_workers < 1:
        raise RuntimeError("collection-workers must be positive")
    if parallel_timeout_seconds < 1:
        raise RuntimeError("parallel Teacher timeout must be positive")
    if parallel_parity_probes < 0:
        raise RuntimeError("parallel Teacher parity probes must be non-negative")

    effective = min(
        int(collection_workers),
        len(selected),
        max(1, os.cpu_count() or 1),
    )
    started = time.perf_counter()

    def sequential(
        reason: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sequential_started = time.perf_counter()
        rows = [
            _branch_example(
                candidate["gc"],
                sts=sts,
                armg=armg,
                mcts_sims=mcts_sims,
                max_game_steps=max_game_steps,
                max_battle_steps=max_battle_steps,
                temperature=temperature,
            )
            for candidate in selected
        ]
        sequential_elapsed = time.perf_counter() - sequential_started
        return rows, {
            "configured_workers": int(collection_workers),
            "effective_workers": 1,
            "mode": "sequential" if reason is None else "sequential_fallback",
            "fallback_reason": reason,
            "available_cpu": int(os.cpu_count() or 1),
            "elapsed_seconds": time.perf_counter() - started,
            "parallel_elapsed_seconds": 0.0,
            "sequential_elapsed_seconds": sequential_elapsed,
            "parity_elapsed_seconds": 0.0,
            "parity_parallel_busy_max_seconds": None,
            "parity_probe_speedup": None,
            "worker_busy_seconds": None,
            "worker_cpu_seconds": None,
            "observed_parallelism": 1.0,
            "observed_cpu_cores": 1.0,
            "parity_probes": 0,
            "parity_status": "not_applicable",
        }

    if effective <= 1 or len(selected) <= 1:
        return sequential()
    if "fork" not in mp.get_all_start_methods():
        return sequential("fork_start_method_unavailable")

    global _TEACHER_PARALLEL_SELECTED
    global _TEACHER_PARALLEL_STS
    global _TEACHER_PARALLEL_ARMG
    global _TEACHER_PARALLEL_CONFIG
    _TEACHER_PARALLEL_SELECTED = selected
    _TEACHER_PARALLEL_STS = sts
    _TEACHER_PARALLEL_ARMG = armg
    _TEACHER_PARALLEL_CONFIG = {
        "mcts_sims": int(mcts_sims),
        "max_game_steps": int(max_game_steps),
        "max_battle_steps": int(max_battle_steps),
        "temperature": float(temperature),
    }

    context = mp.get_context("fork")
    result_queue = context.Queue()
    processes: list[Any] = []
    active: dict[int, Any] = {}

    def launch(index: int) -> None:
        process = context.Process(
            target=_teacher_parallel_process,
            args=(index, (index,), result_queue),
        )
        processes.append(process)
        active[index] = process
        process.start()

    try:
        results: dict[int, dict[str, Any]] = {}
        worker_busy: dict[int, float] = {}
        worker_cpu: dict[int, float] = {}
        deadline = time.monotonic() + parallel_timeout_seconds
        error: str | None = None
        next_index = 0

        # Keep a rolling window full. Each Teacher state still gets a fresh
        # child process, but a fast state no longer waits for the slowest state
        # in a fixed batch before the next state can start.
        while next_index < min(effective, len(selected)):
            launch(next_index)
            next_index += 1

        parallel_started = time.perf_counter()
        while active and error is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = (
                    "TimeoutError:parallel Teacher exceeded "
                    f"{parallel_timeout_seconds}s"
                )
                break

            try:
                kind, worker_id, index, payload = result_queue.get(
                    timeout=min(1.0, remaining)
                )
            except Exception:
                dead_bad = [
                    (worker_id, process.exitcode)
                    for worker_id, process in active.items()
                    if not process.is_alive()
                    and process.exitcode not in (None, 0)
                ]
                if dead_bad:
                    error = (
                        "RuntimeError:parallel Teacher child exited abnormally "
                        f"{dead_bad}"
                    )
                continue

            worker_id = int(worker_id)
            if kind == "result":
                results[int(index)] = dict(payload)
            elif kind == "done":
                if isinstance(payload, Mapping):
                    worker_busy[worker_id] = float(
                        payload.get("busy_seconds", 0.0) or 0.0
                    )
                    worker_cpu[worker_id] = float(
                        payload.get("cpu_seconds", 0.0) or 0.0
                    )
                process = active.pop(worker_id, None)
                if process is None:
                    error = (
                        "RuntimeError:parallel Teacher completed unknown worker "
                        f"{worker_id}"
                    )
                    break
                process.join()
                if process.exitcode != 0:
                    error = (
                        "RuntimeError:parallel Teacher nonzero child exitcode="
                        f"{process.exitcode} worker={worker_id}"
                    )
                    break
                if next_index < len(selected):
                    launch(next_index)
                    next_index += 1
            elif kind == "error":
                error = str(payload)
            else:
                error = (
                    "RuntimeError:unknown parallel Teacher message "
                    f"{kind!r}"
                )

        parallel_elapsed = time.perf_counter() - parallel_started

        if error is not None:
            for process in active.values():
                if process.is_alive():
                    process.terminate()
            for process in active.values():
                process.join()
            return sequential(error)

        if len(results) != len(selected):
            return sequential(
                "RuntimeError:parallel Teacher lost rows "
                f"expected={len(selected)} got={len(results)}"
            )

        rows = [results[index] for index in range(len(selected))]
        drifts = []
        for index, (candidate, row) in enumerate(zip(selected, rows, strict=True)):
            expected_index = int(candidate["current_armg_index"])
            actual_index = int(row.get("current_armg_index", -1))
            if expected_index != actual_index:
                drifts.append(
                    {
                        "index": index,
                        "seed": int(candidate.get("seed", 0) or 0),
                        "kind": str(candidate.get("kind", "unknown")),
                        "expected": expected_index,
                        "actual": actual_index,
                    }
                )
        if drifts:
            first = drifts[0]
            return sequential(
                "RuntimeError:parallel Teacher current-choice drift "
                f"index={first['index']} seed={first['seed']} "
                f"kind={first['kind']} expected={first['expected']} "
                f"actual={first['actual']} count={len(drifts)}"
            )

        parity_count = min(
            int(parallel_parity_probes),
            len(selected),
        )
        parity_started = time.perf_counter()
        for index in range(parity_count):
            reference = _branch_example(
                selected[index]["gc"],
                sts=sts,
                armg=armg,
                mcts_sims=mcts_sims,
                max_game_steps=max_game_steps,
                max_battle_steps=max_battle_steps,
                temperature=temperature,
            )
            parallel = rows[index]
            stable_fields = (
                "kind",
                "current_armg_index",
                "teacher_best_index",
                "branch_quality",
                "target_probs",
            )
            mismatch = [
                field
                for field in stable_fields
                if parallel.get(field) != reference.get(field)
            ]
            if mismatch:
                return sequential(
                    "RuntimeError:parallel Teacher parity mismatch "
                    f"index={index} fields={','.join(mismatch)}"
                )
        parity_elapsed = time.perf_counter() - parity_started
        parity_parallel_busy_max = (
            max(
                (worker_busy.get(index, 0.0) for index in range(parity_count)),
                default=0.0,
            )
            if parity_count > 0
            else 0.0
        )
        parity_probe_speedup = (
            parity_elapsed / parity_parallel_busy_max
            if parity_parallel_busy_max > 0
            else None
        )

        elapsed = time.perf_counter() - started
        busy = sum(worker_busy.values())
        cpu = sum(worker_cpu.values())
        return rows, {
            "configured_workers": int(collection_workers),
            "effective_workers": effective,
            "mode": "fork_rolling",
            "fallback_reason": None,
            "available_cpu": int(os.cpu_count() or 1),
            "elapsed_seconds": elapsed,
            "parallel_elapsed_seconds": parallel_elapsed,
            "sequential_elapsed_seconds": None,
            "parity_elapsed_seconds": parity_elapsed,
            "parity_parallel_busy_max_seconds": parity_parallel_busy_max,
            "parity_probe_speedup": parity_probe_speedup,
            "worker_busy_seconds": busy,
            "worker_cpu_seconds": cpu,
            "observed_parallelism": (
                busy / parallel_elapsed if parallel_elapsed > 0 else 0.0
            ),
            "observed_cpu_cores": (
                cpu / parallel_elapsed if parallel_elapsed > 0 else 0.0
            ),
            "parity_probes": parity_count,
            "parity_status": "PASS",
        }
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
        result_queue.close()
        result_queue.join_thread()
        _TEACHER_PARALLEL_SELECTED = None
        _TEACHER_PARALLEL_STS = None
        _TEACHER_PARALLEL_ARMG = None
        _TEACHER_PARALLEL_CONFIG = None


def _collect_dataset(
    *,
    seeds: Sequence[int],
    sts: Any,
    armg: ArmGNoncombatPolicy,
    output: Path,
    mcts_sims: int,
    max_branch_points_per_game: int,
    max_branch_points_per_kind_per_game: int,
    max_choice_branches: int,
    max_game_steps: int,
    max_battle_steps: int,
    temperature: float,
    teacher_label_budget: int,
    min_labels_per_kind: int,
    collection_workers: int,
    parallel_timeout_seconds: int,
    parallel_parity_probes: int,
) -> dict[str, Any]:
    """Scout cheaply first, then spend MCTS only on high-value Strategy states."""
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    skipped_wide = 0
    skipped_single = 0

    # Phase A: scout trajectories under the current Strategy. This is cheap
    # relative to terminal branch labeling and visits states the learner
    # actually reaches.
    for seed in seeds:
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, int(seed), 0)
        agent = sts.Agent()
        _set_pauses(agent)
        steps = 0
        candidate_count = 0
        kind_seen: Counter[str] = Counter()

        while gc.outcome == sts.GameOutcome.UNDECIDED and steps < max_game_steps:
            steps += 1
            agent.playout(gc)
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                break

            if gc.screen_state == sts.ScreenState.BATTLE:
                _drive_pure_mcts_battle(
                    gc,
                    sts=sts,
                    mcts_sims=mcts_sims,
                    max_battle_steps=max_battle_steps,
                )
                continue

            kind, descs, execs = armg.choices(gc)
            choices = len(descs)
            if choices == 0:
                armg.step(gc, sts)
                continue
            if choices == 1:
                skipped_single += 1
                execs[0](gc)
                continue
            if choices > max_choice_branches:
                skipped_wide += 1
                armg.step(gc, sts)
                continue

            # Keep scouting bounded per game/kind so a long card-heavy run
            # cannot dominate memory or the candidate pool.
            if (
                candidate_count < max_branch_points_per_game
                and kind_seen[str(kind)] < max_branch_points_per_kind_per_game
            ):
                snapshot = _choice_snapshot(gc, armg)
                _, _, scores = armg.score_choices(gc)
                uncertainty = _choice_uncertainty(scores)
                current_index = int(torch.argmax(scores).item())
                bucket = _state_bucket(snapshot, choices=choices)
                identity = hashlib.sha256(
                    json.dumps(
                        {
                            "kind": snapshot["kind"],
                            "obs": snapshot["obs"],
                            "descs": snapshot["descs"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                candidates.append(
                    {
                        "identity": identity,
                        "seed": int(seed),
                        "floor": int(snapshot["floor"]),
                        "act": int(snapshot["act"]),
                        "kind": str(kind),
                        "choices": choices,
                        "snapshot": snapshot,
                        "state_bucket": bucket,
                        "uncertainty": uncertainty,
                        "prelabel_priority": _prelabel_priority(
                            snapshot,
                            uncertainty,
                            choices=choices,
                        ),
                        "current_armg_index": current_index,
                        "gc": gc.clone(),
                    }
                )
                candidate_count += 1
                kind_seen[str(kind)] += 1

            # Scout trajectory follows current Strategy. Teacher labels are
            # generated later and therefore cannot bias which states are found.
            armg.step(gc, sts)

        if gc.outcome == sts.GameOutcome.UNDECIDED:
            raise RuntimeError(f"strategy scout game step bound reached seed={seed}")
        games.append(
            {
                "seed": int(seed),
                "outcome": (
                    "victory"
                    if gc.outcome == sts.GameOutcome.PLAYER_VICTORY
                    else "defeat"
                ),
                "final_floor": int(getattr(gc, "floor_num", 0) or 0),
                "candidate_states": candidate_count,
                "candidate_kinds": dict(kind_seen),
            }
        )

    if not candidates:
        raise RuntimeError("strategy scout produced no candidate states")

    selected = _select_teacher_candidates(
        candidates,
        budget=teacher_label_budget,
        min_per_kind=min_labels_per_kind,
    )
    if not selected:
        raise RuntimeError("strategy candidate selector returned no Teacher states")

    selected_by_seed: Counter[int] = Counter()
    selected_by_kind: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    # Phase B: only now pay the expensive terminal branch-rollout cost.
    raw_examples, parallel_teacher = _label_selected_candidates(
        selected,
        sts=sts,
        armg=armg,
        mcts_sims=mcts_sims,
        max_game_steps=max_game_steps,
        max_battle_steps=max_battle_steps,
        temperature=temperature,
        collection_workers=collection_workers,
        parallel_timeout_seconds=parallel_timeout_seconds,
        parallel_parity_probes=parallel_parity_probes,
    )
    with output.open("w", encoding="utf-8") as handle:
        for candidate, example in zip(selected, raw_examples, strict=True):
            if int(example["current_armg_index"]) != int(candidate["current_armg_index"]):
                raise RuntimeError("Strategy scout/Teacher current-choice drift")
            example["prelabel_priority"] = float(candidate["prelabel_priority"])
            example["prelabel_uncertainty"] = dict(candidate["uncertainty"])
            example["state_bucket"] = str(candidate["state_bucket"])
            example["priority"] = min(
                4.0,
                float(example["priority"])
                + 0.25 * float(candidate["prelabel_priority"]),
            )
            examples.append(example)
            selected_by_seed[int(candidate["seed"])] += 1
            selected_by_kind[str(candidate["kind"])] += 1
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    by_kind = Counter(str(row["kind"]) for row in examples)
    candidate_by_kind = Counter(str(row["kind"]) for row in candidates)
    agreement = sum(
        int(row["current_armg_index"] == row["teacher_best_index"])
        for row in examples
    ) / len(examples)
    mistake_examples = sum(
        int(row["current_armg_index"] != row["teacher_best_index"])
        for row in examples
    )
    high_uncertainty = sum(
        float((row.get("prelabel_uncertainty") or {}).get("uncertainty", 0.0)) >= 0.55
        for row in examples
    )

    for game in games:
        game["selected_teacher_states"] = selected_by_seed[int(game["seed"])]

    return {
        "schema_version": STRATEGY_DATASET_SCHEMA_VERSION,
        "data_efficiency_version": DATA_EFFICIENCY_VERSION,
        "combat_policy": f"mcts_{mcts_sims}",
        "training_seed_count": len(seeds),
        "candidate_pool_count": len(candidates),
        "candidate_pool_by_kind": dict(sorted(candidate_by_kind.items())),
        "teacher_label_budget": teacher_label_budget,
        "selected_teacher_count": len(examples),
        "selected_teacher_by_kind": dict(sorted(selected_by_kind.items())),
        "parallel_teacher": parallel_teacher,
        "example_count": len(examples),
        "examples_by_kind": dict(sorted(by_kind.items())),
        "teacher_agreement": agreement,
        "mistake_examples": mistake_examples,
        "high_uncertainty_examples": high_uncertainty,
        "mean_prelabel_priority": (
            sum(float(row.get("prelabel_priority", 0.0)) for row in examples)
            / len(examples)
        ),
        "unique_state_buckets": len(
            {str(row.get("state_bucket", "unknown")) for row in examples}
        ),
        "skipped_wide_decisions": skipped_wide,
        "skipped_single_choice_decisions": skipped_single,
        "games": games,
    }


def _mine_elite_replay(
    *,
    seeds: Sequence[int],
    sts: Any,
    armg_root: Path,
    champion_weight: Path,
    elite_weight: Path,
    output: Path,
    mcts_sims: int,
    max_examples_per_seed: int,
    max_choice_branches: int,
    max_game_steps: int,
    max_battle_steps: int,
    temperature: float,
    min_quality_margin: float,
) -> dict[str, Any]:
    """Mine only Teacher-verified Elite disagreements on training-only seeds."""
    champion = ArmGNoncombatPolicy(root=armg_root, weight_path=champion_weight)
    elite = ArmGNoncombatPolicy(root=armg_root, weight_path=elite_weight)
    elite_sha = _sha256(elite_weight)
    examples: list[dict[str, Any]] = []
    disagreements = 0
    checked = 0
    teacher_checks = 0
    by_kind: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            gc = sts.GameContext(sts.CharacterClass.IRONCLAD, int(seed), 0)
            agent = sts.Agent()
            _set_pauses(agent)
            steps = 0
            mined_this_seed = 0
            teacher_checks_this_seed = 0
            while gc.outcome == sts.GameOutcome.UNDECIDED and steps < max_game_steps:
                steps += 1
                agent.playout(gc)
                if gc.outcome != sts.GameOutcome.UNDECIDED:
                    break
                if gc.screen_state == sts.ScreenState.BATTLE:
                    _drive_pure_mcts_battle(
                        gc,
                        sts=sts,
                        mcts_sims=mcts_sims,
                        max_battle_steps=max_battle_steps,
                    )
                    continue

                kind, descs, execs = champion.choices(gc)
                choices = len(descs)
                if choices == 0:
                    champion.step(gc, sts)
                    continue
                if choices == 1:
                    execs[0](gc)
                    continue

                _, elite_descs, elite_scores = elite.score_choices(gc)
                if _desc_rows(elite_descs) != _desc_rows(descs):
                    raise RuntimeError("Elite/Champion descriptor drift on identical state")
                _, _, champion_scores = champion.score_choices(gc)
                champion_index = int(torch.argmax(champion_scores).item())
                elite_index = int(torch.argmax(elite_scores).item())
                checked += 1

                if elite_index != champion_index:
                    disagreements += 1
                    if (
                        teacher_checks_this_seed < max_examples_per_seed
                        and choices <= max_choice_branches
                    ):
                        teacher_checks_this_seed += 1
                        teacher_checks += 1
                        teacher = _branch_example(
                            gc,
                            sts=sts,
                            armg=champion,
                            mcts_sims=mcts_sims,
                            max_game_steps=max_game_steps,
                            max_battle_steps=max_battle_steps,
                            temperature=temperature,
                        )
                        row = _elite_verified_example(
                            teacher,
                            elite_index=elite_index,
                            elite_sha256=elite_sha,
                            min_quality_margin=min_quality_margin,
                        )
                        if row is not None:
                            row["elite_mining_seed"] = int(seed)
                            examples.append(row)
                            by_kind[str(row.get("kind", "unknown"))] += 1
                            mined_this_seed += 1
                            handle.write(
                                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                            handle.flush()

                # Stay on the Champion path. The Elite only proposes alternatives.
                kind_now, descs_now, execs_now = champion.choices(gc)
                if str(kind_now) != str(kind) or _desc_rows(descs_now) != _desc_rows(descs):
                    raise RuntimeError("Champion choice identity changed during Elite mining")
                execs_now[champion_index](gc)

            if gc.outcome == sts.GameOutcome.UNDECIDED:
                raise RuntimeError(f"Elite mining game step bound reached seed={seed}")

    return {
        "elite_candidate_sha256": elite_sha,
        "training_seed_count": len(seeds),
        "checked_multichoice_states": checked,
        "disagreements": disagreements,
        "teacher_checks": teacher_checks,
        "verified_examples": len(examples),
        "examples_by_kind": dict(sorted(by_kind.items())),
        "min_quality_margin": float(min_quality_margin),
        "combat_policy": f"mcts_{mcts_sims}",
    }


def _example_identity(row: Mapping[str, Any]) -> str:
    payload = {
        "kind": row.get("kind"),
        "obs": row.get("obs"),
        "descs": row.get("descs"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _focus_identity_set(paths: Sequence[Path]) -> set[str]:
    identities: set[str] = set()
    for path in paths:
        for row in _read_examples(path):
            identities.add(_example_identity(row))
    return identities


def _example_focus_weight(
    row: Mapping[str, Any],
    *,
    focus_ids: set[str],
    focus_weight: float,
) -> float:
    if not math.isfinite(float(focus_weight)) or float(focus_weight) < 1.0:
        raise RuntimeError("focus weight must be finite and >= 1")
    return float(focus_weight) if _example_identity(row) in focus_ids else 1.0


def _read_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != STRATEGY_DATASET_SCHEMA_VERSION:
            raise RuntimeError("strategy dataset schema mismatch")
        if row.get("combat_policy") is None:
            raise RuntimeError("strategy dataset is missing combat-policy identity")
        rows.append(row)
    return rows


def _merge_replay(
    replay_path: Path,
    new_dataset_path: Path,
    *,
    max_examples: int,
    shop_max_fraction: float = 0.25,
) -> dict[str, Any]:
    """Deduplicate, confidence-rank, and prevent noisy Shop replay domination."""
    if max_examples < 1:
        raise RuntimeError("max replay examples must be positive")
    if not 0.0 < shop_max_fraction <= 1.0:
        raise RuntimeError("shop max fraction must be in (0, 1]")
    rows = _read_examples(replay_path) + _read_examples(new_dataset_path)

    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _example_identity(row)
        old = dedup.get(key)
        row_margin = float(row.get("teacher_margin", 0.0))
        row_confidence = float(
            row.get("confidence_weight", _teacher_confidence_weight(row_margin))
        )
        row["confidence_weight"] = row_confidence
        if old is None:
            dedup[key] = row
            continue
        old_margin = float(old.get("teacher_margin", 0.0))
        old_confidence = float(
            old.get("confidence_weight", _teacher_confidence_weight(old_margin))
        )
        new_rank = (
            row_confidence,
            float(row.get("priority", 1.0)),
            row_margin,
        )
        old_rank = (
            old_confidence,
            float(old.get("priority", 1.0)),
            old_margin,
        )
        if new_rank >= old_rank:
            dedup[key] = row

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dedup.values():
        groups[str(row.get("kind", "unknown"))].append(row)
    for values in groups.values():
        values.sort(
            key=lambda row: (
                float(row.get("confidence_weight", 0.10)),
                float(row.get("priority", 1.0)),
                float(row.get("teacher_margin", 0.0)),
            ),
            reverse=True,
        )

    dropped_shop_examples = 0
    shop_cap = len(groups.get("shop", []))
    nonshop_count = sum(
        len(values) for kind, values in groups.items() if kind != "shop"
    )
    if "shop" in groups and shop_max_fraction < 1.0 and nonshop_count > 0:
        ratio_cap = int(
            math.floor(
                nonshop_count * shop_max_fraction / (1.0 - shop_max_fraction)
            )
        )
        shop_cap = max(1, min(len(groups["shop"]), ratio_cap, max_examples))
        dropped_shop_examples = max(0, len(groups["shop"]) - shop_cap)
        groups["shop"] = groups["shop"][:shop_cap]

    kept: list[dict[str, Any]] = []
    kind_names = sorted(kind for kind, values in groups.items() if values)
    cursor = 0
    while len(kept) < max_examples and kind_names:
        kind = kind_names[cursor % len(kind_names)]
        bucket = groups[kind]
        if bucket:
            kept.append(bucket.pop(0))
        if not bucket:
            kind_names.remove(kind)
            cursor = 0
        else:
            cursor += 1

    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in kept
        ),
        encoding="utf-8",
    )
    by_kind = Counter(str(row.get("kind", "unknown")) for row in kept)
    confidences = [float(row.get("confidence_weight", 0.10)) for row in kept]
    return {
        "examples": len(kept),
        "examples_by_kind": dict(sorted(by_kind.items())),
        "mistake_examples": sum(
            int(row.get("current_armg_index") != row.get("teacher_best_index"))
            for row in kept
        ),
        "shop_max_fraction": float(shop_max_fraction),
        "shop_cap": int(shop_cap),
        "dropped_shop_examples": int(dropped_shop_examples),
        "mean_confidence_weight": (
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
        "low_confidence_examples": sum(value <= 0.25 for value in confidences),
        "high_confidence_examples": sum(value >= 0.75 for value in confidences),
    }

def _teacher_accuracy(
    net: Any,
    examples: Sequence[Mapping[str, Any]],
) -> tuple[float, dict[str, float]]:
    if not examples:
        raise RuntimeError("strategy accuracy requires examples")
    net.eval()
    correct = 0
    kind_total: Counter[str] = Counter()
    kind_correct: Counter[str] = Counter()
    with torch.no_grad():
        for row in examples:
            obs = torch.tensor(row["obs"], dtype=torch.float32)
            scores = net.score(obs, row["descs"])
            pred = int(torch.argmax(scores).item())
            target = int(row["teacher_best_index"])
            kind = str(row.get("kind", "unknown"))
            correct += int(pred == target)
            kind_total[kind] += 1
            kind_correct[kind] += int(pred == target)
    return (
        correct / len(examples),
        {
            kind: kind_correct[kind] / kind_total[kind]
            for kind in sorted(kind_total)
        },
    )


def _train_candidate(
    *,
    armg: ArmGNoncombatPolicy,
    source_weight: Path,
    replay_path: Path,
    output_weight: Path,
    epochs: int,
    learning_rate: float,
    anchor_coef: float,
    rng_seed: int,
    min_teacher_confidence: float = 0.0,
    parent_distill_coef: float = AMBIGUITY_PARENT_DISTILL_COEF,
    focus_paths: Sequence[Path] = (),
    focus_weight: float = 1.0,
) -> dict[str, Any]:
    examples = _read_examples(replay_path)
    if not examples:
        raise RuntimeError("strategy replay is empty")
    focus_ids = _focus_identity_set(focus_paths)
    if not math.isfinite(float(focus_weight)) or float(focus_weight) < 1.0:
        raise RuntimeError("focus weight must be finite and >= 1")

    module = armg.module
    source_state = torch.load(source_weight, weights_only=True, map_location="cpu")
    candidate = module.Scorer((128, 128))
    candidate.load_state_dict(source_state)
    candidate.train()
    parent = module.Scorer((128, 128))
    parent.load_state_dict(source_state)
    parent.eval()
    source_params = {
        name: value.detach().clone()
        for name, value in candidate.named_parameters()
    }
    optimizer = torch.optim.Adam(candidate.parameters(), lr=learning_rate)

    before, before_by_kind = _teacher_accuracy(candidate, examples)
    kinds = Counter(str(row.get("kind", "unknown")) for row in examples)
    total = len(examples)
    kind_count = max(1, len(kinds))
    kind_weight = {
        kind: min(3.0, total / float(kind_count * count))
        for kind, count in kinds.items()
    }

    rng = random.Random(rng_seed)
    losses: list[float] = []
    ce_losses: list[float] = []
    parent_distill_losses: list[float] = []
    anchor_losses: list[float] = []
    teacher_mix_values: list[float] = []
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for index in order:
            row = examples[index]
            obs = torch.tensor(row["obs"], dtype=torch.float32)
            scores = candidate.score(obs, row["descs"])
            target = torch.tensor(row["target_probs"], dtype=torch.float32)
            ce = -(target * torch.log_softmax(scores, dim=0)).sum()
            with torch.no_grad():
                parent_scores = parent.score(obs, row["descs"])
                parent_probs = torch.softmax(parent_scores, dim=0)
            parent_distill = -(
                parent_probs * torch.log_softmax(scores, dim=0)
            ).sum()

            anchor_terms = [
                (param - source_params[name]).pow(2).mean()
                for name, param in candidate.named_parameters()
            ]
            anchor = torch.stack(anchor_terms).mean()

            priority = float(row.get("priority", 1.0))
            balance = kind_weight[str(row.get("kind", "unknown"))]
            margin = float(row.get("teacher_margin", 0.0))
            confidence = float(
                row.get("confidence_weight", _teacher_confidence_weight(margin))
            )
            teacher_mix = _teacher_mix_weight(
                confidence,
                min_teacher_confidence=min_teacher_confidence,
            )
            strategy_loss = (
                teacher_mix * ce
                + parent_distill_coef * (1.0 - teacher_mix) * parent_distill
            )
            focus = _example_focus_weight(
                row,
                focus_ids=focus_ids,
                focus_weight=focus_weight,
            )
            loss = priority * balance * focus * strategy_loss + anchor_coef * anchor

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            ce_losses.append(float(ce.item()))
            parent_distill_losses.append(float(parent_distill.item()))
            anchor_losses.append(float(anchor.item()))
            teacher_mix_values.append(float(teacher_mix))

    after, after_by_kind = _teacher_accuracy(candidate, examples)
    candidate.eval()
    output_weight.parent.mkdir(parents=True, exist_ok=True)
    torch.save(candidate.state_dict(), output_weight)
    return {
        "examples": len(examples),
        "examples_by_kind": dict(sorted(kinds.items())),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "anchor_coef": anchor_coef,
        "min_teacher_confidence": min_teacher_confidence,
        "parent_distill_coef": parent_distill_coef,
        "focus_weight": float(focus_weight),
        "focus_example_count": sum(
            _example_identity(row) in focus_ids for row in examples
        ),
        "mean_teacher_mix": sum(teacher_mix_values) / len(teacher_mix_values),
        "teacher_driven_updates": sum(value > 0.0 for value in teacher_mix_values),
        "parent_preservation_updates": sum(value < 1.0 for value in teacher_mix_values),
        "mean_confidence_weight": sum(
            float(
                row.get(
                    "confidence_weight",
                    _teacher_confidence_weight(float(row.get("teacher_margin", 0.0))),
                )
            )
            for row in examples
        ) / len(examples),
        "before_teacher_top1": before,
        "after_teacher_top1": after,
        "before_teacher_top1_by_kind": before_by_kind,
        "after_teacher_top1_by_kind": after_by_kind,
        "mean_loss": sum(losses) / len(losses),
        "mean_cross_entropy": sum(ce_losses) / len(ce_losses),
        "mean_parent_distill_loss": (
            sum(parent_distill_losses) / len(parent_distill_losses)
        ),
        "mean_anchor_loss": sum(anchor_losses) / len(anchor_losses),
        "source_sha256": _sha256(source_weight),
        "candidate_sha256": _sha256(output_weight),
    }


def _interpolate_checkpoint(
    *,
    source_weight: Path,
    trained_weight: Path,
    output_weight: Path,
    alpha: float,
) -> None:
    """Create a Parent-protected Candidate by scaling one learned update."""
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise RuntimeError("candidate step alpha must be in (0, 1]")
    source = torch.load(source_weight, weights_only=True, map_location="cpu")
    trained = torch.load(trained_weight, weights_only=True, map_location="cpu")
    if set(source) != set(trained):
        raise RuntimeError("candidate checkpoint keys drifted from Parent")
    interpolated: dict[str, Any] = {}
    for name in source:
        parent_value = source[name]
        trained_value = trained[name]
        if getattr(parent_value, "shape", None) != getattr(trained_value, "shape", None):
            raise RuntimeError(f"candidate checkpoint shape drift: {name}")
        if torch.is_floating_point(parent_value):
            interpolated[name] = parent_value + alpha * (trained_value - parent_value)
        else:
            if not torch.equal(parent_value, trained_value):
                raise RuntimeError(f"non-floating checkpoint value drift: {name}")
            interpolated[name] = parent_value
    output_weight.parent.mkdir(parents=True, exist_ok=True)
    torch.save(interpolated, output_weight)


def _dev_gate_rank(gate: Mapping[str, Any]) -> tuple[float, ...]:
    """Victory-first Candidate ordering; floor depth is not a selection target."""
    paired_wins = gate.get("paired_wins") or {}
    return (
        1.0 if gate.get("status") == "PASS" else 0.0,
        float(gate.get("win_delta", -10**9) or 0.0),
        float(paired_wins.get("candidate_better", 0) or 0)
        - float(paired_wins.get("candidate_worse", 0) or 0),
        -float(paired_wins.get("one_sided_sign_p", 1.0) or 1.0),
    )


def _select_dev_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not candidates:
        raise RuntimeError("candidate pool is empty")
    return max(candidates, key=lambda row: _dev_gate_rank(row["dev_gate"]))


def _evaluate_seed_worker(
    payload: tuple[str, str, str, int, tuple[int, ...], int],
) -> dict[str, Any]:
    module_dir_text, armg_root_text, weight_text, seed, all_seeds, mcts_sims = payload
    module_dir = Path(module_dir_text)
    armg_root = Path(armg_root_text)
    strategy_weight = Path(weight_text)
    sts = _load_sts(module_dir)
    armg = ArmGNoncombatPolicy(
        root=armg_root,
        weight_path=strategy_weight,
    )
    summary = run_simulator_game(
        student=None,
        sts=sts,
        seed=int(seed),
        armg_policy=armg,
        combat_mcts_sims=mcts_sims,
        heldout_seeds=all_seeds,
        collect_ppo=False,
        collect_teacher=False,
    )
    return dict(summary)


def _evaluate_weight(
    *,
    seeds: Sequence[int],
    module_dir: Path,
    armg_root: Path,
    strategy_weight: Path,
    mcts_sims: int,
    workers: int,
) -> dict[str, Any]:
    ordered_seeds = tuple(int(seed) for seed in seeds)
    if not ordered_seeds:
        raise RuntimeError("strategy evaluation seed set is empty")
    if workers < 1:
        raise RuntimeError("eval workers must be positive")

    payloads = [
        (
            str(module_dir),
            str(armg_root),
            str(strategy_weight),
            seed,
            ordered_seeds,
            int(mcts_sims),
        )
        for seed in ordered_seeds
    ]

    if workers == 1:
        runs = [_evaluate_seed_worker(payload) for payload in payloads]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(payloads)),
            mp_context=context,
        ) as pool:
            runs = list(pool.map(_evaluate_seed_worker, payloads))

    by_seed = {int(row["seed"]): row for row in runs}
    if set(by_seed) != set(ordered_seeds):
        raise RuntimeError("parallel Strategy evaluation lost or duplicated seeds")
    ordered_runs = [by_seed[seed] for seed in ordered_seeds]
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "strategy_weight_sha256": _sha256(strategy_weight),
        "combat_policy": f"mcts_{mcts_sims}",
        "seed_count": len(ordered_seeds),
        "workers": min(workers, len(ordered_seeds)),
        "runs": ordered_runs,
    }


def _load_or_init_state(
    *,
    state_dir: Path,
    base_weight: Path,
    combat_mcts_sims: int,
) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "strategy-state.json"
    current = state_dir / "current-strategy.pt"
    if state_path.is_file():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise RuntimeError("strategy loop state schema mismatch")
        if not current.is_file():
            raise RuntimeError("strategy loop state is missing current-strategy.pt")
        if payload.get("current_strategy_sha256") != _sha256(current):
            raise RuntimeError("strategy current checkpoint SHA drift")
        if payload.get("base_strategy_sha256") != _sha256(base_weight):
            raise RuntimeError(
                "ArmG base checkpoint changed inside the existing Strategy lineage"
            )
        training_history = {int(v) for v in payload.get("used_training_seeds", [])}
        evaluation_history = {int(v) for v in payload.get("used_evaluation_seeds", [])}
        if training_history & evaluation_history:
            raise RuntimeError("Strategy train/evaluation seed history overlap")
        if int(payload.get("combat_mcts_sims", -1)) != combat_mcts_sims:
            raise RuntimeError(
                "combat MCTS budget is frozen for one Strategy lineage; "
                "start a new state-dir to change it"
            )
        if int(payload.get("data_efficiency_version", 1)) < DATA_EFFICIENCY_VERSION:
            payload["data_efficiency_version"] = DATA_EFFICIENCY_VERSION
            payload["stagnation_count"] = 0
            payload["data_efficiency_upgrade_round"] = (
                int(payload.get("accepted_rounds", 0))
                + int(payload.get("rejected_rounds", 0))
            )
        if int(payload.get("parallel_cpu_version", 0)) < PARALLEL_CPU_VERSION:
            payload["parallel_cpu_version"] = PARALLEL_CPU_VERSION
        _write_json(state_path, payload)
        return payload

    shutil.copy2(base_weight, current)
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "generation": 0,
        "accepted_rounds": 0,
        "rejected_rounds": 0,
        "stagnation_count": 0,
        "used_training_seeds": [],
        "used_evaluation_seeds": [],
        "current_strategy_sha256": _sha256(current),
        "base_strategy_sha256": _sha256(base_weight),
        "combat_mcts_sims": int(combat_mcts_sims),
        "data_efficiency_version": DATA_EFFICIENCY_VERSION,
        "parallel_cpu_version": PARALLEL_CPU_VERSION,
    }
    _write_json(state_path, payload)
    return payload


def _stagnation_limit_reached(*, count: int, max_stagnation: int) -> bool:
    if count < 0 or max_stagnation < 0:
        raise RuntimeError("stagnation values must be non-negative")
    return max_stagnation > 0 and count >= max_stagnation


def _round_number(state: Mapping[str, Any]) -> int:
    return int(state.get("accepted_rounds", 0)) + int(
        state.get("rejected_rounds", 0)
    ) + 1


def _eval_and_gate(
    *,
    current_weight: Path,
    candidate_weight: Path,
    seeds: Sequence[int],
    module_dir: Path,
    armg_root: Path,
    mcts_sims: int,
    workers: int,
    policy: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    current_eval = _evaluate_weight(
        seeds=seeds,
        module_dir=module_dir,
        armg_root=armg_root,
        strategy_weight=current_weight,
        mcts_sims=mcts_sims,
        workers=workers,
    )
    candidate_eval = _evaluate_weight(
        seeds=seeds,
        module_dir=module_dir,
        armg_root=armg_root,
        strategy_weight=candidate_weight,
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
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--armg-base-weight", type=Path, required=True)
    parser.add_argument("--dev-seed-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--training-seeds", type=int, default=4)
    parser.add_argument("--combat-mcts-sims", type=int, default=2000)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--rng-seed", type=int, default=20260928)
    parser.add_argument("--max-branch-points-per-game", type=int, default=8)
    parser.add_argument(
        "--max-branch-points-per-kind-per-game", type=int, default=1
    )
    parser.add_argument("--max-choice-branches", type=int, default=20)
    parser.add_argument("--teacher-label-budget", type=int, default=24)
    parser.add_argument("--min-labels-per-kind", type=int, default=3)
    parser.add_argument("--collection-workers", type=int, default=1)
    parser.add_argument("--parallel-teacher-timeout-seconds", type=int, default=2400)
    parser.add_argument("--parallel-parity-probes", type=int, default=0)
    parser.add_argument("--max-game-steps", type=int, default=600)
    parser.add_argument("--max-battle-steps", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-coef", type=float, default=0.01)
    parser.add_argument("--max-replay-examples", type=int, default=10000)
    parser.add_argument("--shop-max-fraction", type=float, default=0.25)
    parser.add_argument("--max-stagnation", type=int, default=5)
    parser.add_argument("--elite-mining-seeds", type=int, default=2)
    parser.add_argument("--elite-examples-per-seed", type=int, default=2)
    parser.add_argument("--elite-min-quality-margin", type=float, default=1.0)
    parser.add_argument("--elite-max-sign-p", type=float, default=0.25)
    parser.add_argument("--elite-min-win-delta", type=int, default=1)
    parser.add_argument("--elite-min-floor-delta", type=float, default=-0.5)
    parser.add_argument("--elite-max-pool", type=int, default=3)
    args = parser.parse_args()

    if args.rounds < 1 or args.training_seeds < 1:
        raise RuntimeError("rounds and training-seeds must be positive")
    if args.combat_mcts_sims < 1:
        raise RuntimeError("combat-mcts-sims must be positive")
    if args.eval_workers < 1:
        raise RuntimeError("eval-workers must be positive")
    if (
        args.max_branch_points_per_game < 1
        or args.max_branch_points_per_kind_per_game < 1
        or args.max_choice_branches < 2
    ):
        raise RuntimeError("branch limits are invalid")
    if args.epochs < 1 or args.max_stagnation < 0:
        raise RuntimeError("epochs must be positive and max-stagnation non-negative")
    if not 0.0 < args.shop_max_fraction <= 1.0:
        raise RuntimeError("shop-max-fraction must be in (0, 1]")
    if args.teacher_label_budget < 1 or args.min_labels_per_kind < 0:
        raise RuntimeError("Teacher selection limits are invalid")
    if (
        args.collection_workers < 1
        or args.parallel_teacher_timeout_seconds < 1
        or args.parallel_parity_probes < 0
    ):
        raise RuntimeError("Parallel Teacher settings are invalid")
    if args.elite_mining_seeds < 0 or args.elite_examples_per_seed < 1:
        raise RuntimeError("Elite mining limits are invalid")
    if args.elite_min_quality_margin < 0 or not 0 < args.elite_max_sign_p <= 1:
        raise RuntimeError("Elite thresholds are invalid")
    if args.elite_min_win_delta < 1 or args.elite_max_pool < 1:
        raise RuntimeError("Elite promotion-side thresholds are invalid")

    dev_all = _read_seeds(args.dev_seed_file)
    if len(dev_all) != 50:
        raise RuntimeError(
            "dev seed file must contain exactly 50 frozen held-out seeds; "
            "the first 30 are the reusable Dev gate and all 50 stay train-excluded"
        )
    dev30 = dev_all[:30]

    if not args.armg_base_weight.is_file():
        raise RuntimeError(f"ArmG base weight missing: {args.armg_base_weight}")
    state = _load_or_init_state(
        state_dir=args.state_dir,
        base_weight=args.armg_base_weight,
        combat_mcts_sims=args.combat_mcts_sims,
    )
    elite_pool = _load_elite_pool(args.state_dir)

    sts = _load_sts(args.module_dir)
    clone_probe = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)
    if not callable(getattr(clone_probe, "clone", None)):
        raise RuntimeError("GameContext.clone API unavailable; Strategy teacher cannot run")

    current_weight = args.state_dir / "current-strategy.pt"
    replay_path = args.state_dir / "strategy-replay.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    for _ in range(args.rounds):
        if _stagnation_limit_reached(
            count=int(state.get("stagnation_count", 0)),
            max_stagnation=args.max_stagnation,
        ):
            break

        round_no = _round_number(state)
        round_dir = args.output_dir / f"round-{round_no:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        forbidden = (
            set(dev_all)
            | {int(v) for v in state.get("used_training_seeds", [])}
            | {int(v) for v in state.get("used_evaluation_seeds", [])}
        )
        train_seeds = _fresh_seeds(
            count=args.training_seeds,
            rng_seed=args.rng_seed + round_no * 1000 + 1,
            forbidden=forbidden,
        )
        forbidden.update(train_seeds)

        armg = ArmGNoncombatPolicy(
            root=args.armg_root,
            weight_path=current_weight,
        )
        dataset_path = round_dir / "strategy-branch-dataset.jsonl"
        dataset_report = _collect_dataset(
            seeds=train_seeds,
            sts=sts,
            armg=armg,
            output=dataset_path,
            mcts_sims=args.combat_mcts_sims,
            max_branch_points_per_game=args.max_branch_points_per_game,
            max_branch_points_per_kind_per_game=(
                args.max_branch_points_per_kind_per_game
            ),
            max_choice_branches=args.max_choice_branches,
            max_game_steps=args.max_game_steps,
            max_battle_steps=args.max_battle_steps,
            temperature=args.temperature,
            teacher_label_budget=args.teacher_label_budget,
            min_labels_per_kind=args.min_labels_per_kind,
            collection_workers=args.collection_workers,
            parallel_timeout_seconds=args.parallel_teacher_timeout_seconds,
            parallel_parity_probes=args.parallel_parity_probes,
        )
        replay_report = _merge_replay(
            replay_path,
            dataset_path,
            max_examples=args.max_replay_examples,
            shop_max_fraction=args.shop_max_fraction,
        )
        focus_paths: list[Path] = [dataset_path]

        elite_mining_seeds: tuple[int, ...] = ()
        elite_mining_report: dict[str, Any] = {
            "status": "SKIPPED",
            "reason": "elite_pool_empty",
            "verified_examples": 0,
        }
        if elite_pool and args.elite_mining_seeds > 0:
            # Elite mining gets dedicated training-only fresh seeds. Gate seeds are
            # never recycled into Replay, preserving evaluation integrity.
            elite_mining_seeds = _fresh_seeds(
                count=args.elite_mining_seeds,
                rng_seed=args.rng_seed + round_no * 1000 + 11,
                forbidden=forbidden,
            )
            forbidden.update(elite_mining_seeds)
            best_elite = sorted(elite_pool, key=_elite_score, reverse=True)[0]
            elite_weight = args.state_dir / str(best_elite["weight_file"])
            elite_dataset = round_dir / "elite-verified-replay.jsonl"
            elite_mining_report = _mine_elite_replay(
                seeds=elite_mining_seeds,
                sts=sts,
                armg_root=args.armg_root,
                champion_weight=current_weight,
                elite_weight=elite_weight,
                output=elite_dataset,
                mcts_sims=args.combat_mcts_sims,
                max_examples_per_seed=args.elite_examples_per_seed,
                max_choice_branches=args.max_choice_branches,
                max_game_steps=args.max_game_steps,
                max_battle_steps=args.max_battle_steps,
                temperature=args.temperature,
                min_quality_margin=args.elite_min_quality_margin,
            )
            elite_mining_report["status"] = "COMPLETE"
            if int(elite_mining_report.get("verified_examples", 0)) > 0:
                replay_report = _merge_replay(
                    replay_path,
                    elite_dataset,
                    max_examples=args.max_replay_examples,
                    shop_max_fraction=args.shop_max_fraction,
                )
                focus_paths.append(elite_dataset)

        current_dev = _evaluate_weight(
            seeds=dev30,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            strategy_weight=current_weight,
            mcts_sims=args.combat_mcts_sims,
            workers=args.eval_workers,
        )
        _write_json(round_dir / "eval-current-dev30.json", current_dev)

        candidate_records: list[dict[str, Any]] = []
        training_variants: list[dict[str, Any]] = []
        candidate_recipes = _candidate_recipes_for_stagnation(
            int(state.get("stagnation_count", 0))
        )
        for candidate_index, recipe in enumerate(candidate_recipes):
            recipe_name = str(recipe["name"])
            min_teacher_confidence = float(recipe["min_teacher_confidence"])
            step_scale = float(recipe["step_scale"])
            trained_weight = round_dir / f"candidate-trained-{recipe_name}.pt"
            rescue_mode = int(state.get("stagnation_count", 0)) >= 12
            variant_train_report = _train_candidate(
                armg=armg,
                source_weight=current_weight,
                replay_path=replay_path,
                output_weight=trained_weight,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                anchor_coef=args.anchor_coef,
                rng_seed=args.rng_seed + round_no * 1000 + 2 + candidate_index,
                min_teacher_confidence=min_teacher_confidence,
                focus_paths=focus_paths if rescue_mode else (),
                focus_weight=6.0 if rescue_mode else 1.0,
            )
            label = f"{recipe_name}-step-{int(round(step_scale * 100)):03d}"
            variant_weight = round_dir / f"candidate-{label}.pt"
            _interpolate_checkpoint(
                source_weight=current_weight,
                trained_weight=trained_weight,
                output_weight=variant_weight,
                alpha=step_scale,
            )
            candidate_eval = _evaluate_weight(
                seeds=dev30,
                module_dir=args.module_dir,
                armg_root=args.armg_root,
                strategy_weight=variant_weight,
                mcts_sims=args.combat_mcts_sims,
                workers=args.eval_workers,
            )
            gate = evaluate_strategy_gate(
                current_dev,
                candidate_eval,
                policy=DEV_STRATEGY_GATE,
                expected_combat_policy=f"mcts_{args.combat_mcts_sims}",
            )
            _write_json(round_dir / f"eval-candidate-dev30-{label}.json", candidate_eval)
            training_variants.append(
                {
                    **variant_train_report,
                    "recipe": recipe_name,
                    "step_scale": step_scale,
                }
            )
            candidate_records.append(
                {
                    "index": candidate_index,
                    "name": label,
                    "recipe": recipe_name,
                    "step_scale": step_scale,
                    "min_teacher_confidence": min_teacher_confidence,
                    "weight_path": str(variant_weight),
                    "candidate_strategy_sha256": _sha256(variant_weight),
                    "candidate_eval": candidate_eval,
                    "dev_gate": gate,
                    "training": variant_train_report,
                }
            )

        chosen_candidate = _select_dev_candidate(candidate_records)
        candidate_weight = round_dir / "candidate-strategy.pt"
        shutil.copy2(Path(str(chosen_candidate["weight_path"])), candidate_weight)
        candidate_dev = dict(chosen_candidate["candidate_eval"])
        dev_gate = dict(chosen_candidate["dev_gate"])
        _write_json(round_dir / "eval-candidate-dev30.json", candidate_dev)
        candidate_pool_report = [
            {
                "index": int(row["index"]),
                "name": str(row["name"]),
                "recipe": str(row["recipe"]),
                "step_scale": float(row["step_scale"]),
                "min_teacher_confidence": float(row["min_teacher_confidence"]),
                "candidate_strategy_sha256": str(row["candidate_strategy_sha256"]),
                "dev_gate": row["dev_gate"],
                "selected": row is chosen_candidate,
            }
            for row in candidate_records
        ]
        train_report = {
            **dict(chosen_candidate["training"]),
            "candidate_recipes": [
                {
                    "name": str(recipe["name"]),
                    "min_teacher_confidence": float(recipe["min_teacher_confidence"]),
                    "step_scale": float(recipe["step_scale"]),
                }
                for recipe in candidate_recipes
            ],
            "stagnation_rescue": int(state.get("stagnation_count", 0)) >= 12,
            "training_variants": training_variants,
            "selected_recipe": str(chosen_candidate["recipe"]),
            "selected_step_scale": float(chosen_candidate["step_scale"]),
            "selected_candidate_name": str(chosen_candidate["name"]),
            "selected_candidate_sha256": _sha256(candidate_weight),
        }

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

        if dev_gate["status"] == "PASS":
            hidden_seeds = _fresh_seeds(
                count=50,
                rng_seed=args.rng_seed + round_no * 1000 + 3,
                forbidden=forbidden,
            )
            forbidden.update(hidden_seeds)
            current_hidden, candidate_hidden, hidden_gate = _eval_and_gate(
                current_weight=current_weight,
                candidate_weight=candidate_weight,
                seeds=hidden_seeds,
                module_dir=args.module_dir,
                armg_root=args.armg_root,
                mcts_sims=args.combat_mcts_sims,
                workers=args.eval_workers,
                policy=HIDDEN_STRATEGY_GATE,
            )
            _write_json(round_dir / "eval-current-hidden50.json", current_hidden)
            _write_json(round_dir / "eval-candidate-hidden50.json", candidate_hidden)

        if hidden_gate.get("status") == "PASS":
            fresh_seeds = _fresh_seeds(
                count=100,
                rng_seed=args.rng_seed + round_no * 1000 + 4,
                forbidden=forbidden,
            )
            forbidden.update(fresh_seeds)
            current_fresh, candidate_fresh, fresh_gate = _eval_and_gate(
                current_weight=current_weight,
                candidate_weight=candidate_weight,
                seeds=fresh_seeds,
                module_dir=args.module_dir,
                armg_root=args.armg_root,
                mcts_sims=args.combat_mcts_sims,
                workers=args.eval_workers,
                policy=FRESH_STRATEGY_GATE,
            )
            _write_json(round_dir / "eval-current-fresh100.json", current_fresh)
            _write_json(round_dir / "eval-candidate-fresh100.json", candidate_fresh)

        promotion = strategy_promotion_decision(
            dev_gate,
            hidden_gate,
            fresh_gate,
        )
        promoted = promotion["decision"] == "PROMOTE_STRATEGY"
        before_sha = _sha256(current_weight)
        elite_decision = _elite_candidate_decision(
            fresh_gate,
            promoted=promoted,
            max_sign_p=args.elite_max_sign_p,
            min_win_delta=args.elite_min_win_delta,
            min_floor_delta=args.elite_min_floor_delta,
        )
        elite_pool_update: dict[str, Any] = {
            "saved": False,
            "pool_size": len(elite_pool),
        }
        if promoted:
            shutil.copy2(candidate_weight, current_weight)
            if _sha256(current_weight) != _sha256(candidate_weight):
                raise RuntimeError("Strategy promotion checkpoint checksum mismatch")
            # Old near-miss Candidates were measured against the previous Champion.
            _clear_elite_pool(args.state_dir)
            elite_pool = []
            elite_pool_update = {
                "saved": False,
                "cleared_after_promotion": True,
                "pool_size": 0,
            }
        elif elite_decision.get("eligible") is True:
            elite_pool_update = _save_elite_candidate(
                state_dir=args.state_dir,
                candidate_weight=candidate_weight,
                round_no=round_no,
                parent_champion_sha=before_sha,
                metrics=elite_decision,
                max_entries=args.elite_max_pool,
            )
            elite_pool = _load_elite_pool(args.state_dir)

        consumed_eval = list(hidden_seeds) + list(fresh_seeds)
        state = {
            **state,
            "generation": int(state.get("generation", 0)) + int(promoted),
            "accepted_rounds": int(state.get("accepted_rounds", 0)) + int(promoted),
            "rejected_rounds": int(state.get("rejected_rounds", 0)) + int(not promoted),
            "stagnation_count": 0 if promoted else int(state.get("stagnation_count", 0)) + 1,
            "used_training_seeds": list(state.get("used_training_seeds", []))
            + list(train_seeds)
            + list(elite_mining_seeds),
            "used_evaluation_seeds": list(state.get("used_evaluation_seeds", []))
            + consumed_eval,
            "current_strategy_sha256": _sha256(current_weight),
            "data_efficiency_version": DATA_EFFICIENCY_VERSION,
            "parallel_cpu_version": PARALLEL_CPU_VERSION,
            "elite_pool_count": len(elite_pool),
        }
        _write_json(args.state_dir / "strategy-state.json", state)

        report = {
            "schema_version": ROUND_SCHEMA_VERSION,
            "round": round_no,
            "combat_training_enabled": False,
            "combat_policy": f"mcts_{args.combat_mcts_sims}",
            "strategy_scope": "all_armg_noncombat_choice_kinds",
            "data_efficiency_version": DATA_EFFICIENCY_VERSION,
            "parallel_cpu_version": PARALLEL_CPU_VERSION,
            "collection_workers": int(args.collection_workers),
            "current_strategy_sha_before": before_sha,
            "current_strategy_sha_after": _sha256(current_weight),
            "candidate_strategy_sha256": _sha256(candidate_weight),
            "training_seeds": list(train_seeds),
            "elite_mining_seeds": list(elite_mining_seeds),
            "hidden_eval_seeds": list(hidden_seeds),
            "fresh_eval_seeds": list(fresh_seeds),
            "dataset": dataset_report,
            "replay": replay_report,
            "elite_mining": elite_mining_report,
            "elite_decision": elite_decision,
            "elite_pool": {
                "entries": len(elite_pool),
                "update": elite_pool_update,
            },
            "training": train_report,
            "candidate_pool": candidate_pool_report,
            "dev_gate": dev_gate,
            "hidden_gate": hidden_gate,
            "fresh_gate": fresh_gate,
            "promotion": promotion,
            "generation": int(state["generation"]),
            "stagnation_count": int(state["stagnation_count"]),
            "production_champion_replaced": False,
        }
        _write_json(round_dir / "round-report.json", report)
        reports.append(report)

    final = {
        "schema_version": STATE_SCHEMA_VERSION,
        "result": "PASS_STRATEGY_LOOP_COMPLETE",
        "rounds_requested": args.rounds,
        "rounds_completed": len(reports),
        "combat_training_enabled": False,
        "combat_policy": f"mcts_{args.combat_mcts_sims}",
        "data_efficiency_version": DATA_EFFICIENCY_VERSION,
        "parallel_cpu_version": PARALLEL_CPU_VERSION,
        "collection_workers": int(args.collection_workers),
        "generation": int(state.get("generation", 0)),
        "accepted_rounds": int(state.get("accepted_rounds", 0)),
        "rejected_rounds": int(state.get("rejected_rounds", 0)),
        "stagnation_count": int(state.get("stagnation_count", 0)),
        "paused_for_stagnation": _stagnation_limit_reached(
            count=int(state.get("stagnation_count", 0)),
            max_stagnation=args.max_stagnation,
        ),
        "current_strategy_sha256": _sha256(current_weight),
        "elite_pool_count": len(_load_elite_pool(args.state_dir)),
        "replay": (
            {
                "examples": len(_read_examples(replay_path)),
                "sha256": _sha256(replay_path),
            }
            if replay_path.is_file()
            else {"examples": 0}
        ),
        "production_champion_replaced": False,
    }
    _write_json(args.output_dir / "loop-summary.json", final)
    print("STS1_STRATEGY_LOOP", json.dumps(final, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
