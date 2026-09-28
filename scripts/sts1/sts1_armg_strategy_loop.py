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
import multiprocessing as mp
import json
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
        "teacher_margin": (
            sorted(values, reverse=True)[0] - sorted(values, reverse=True)[1]
        ),
        "branches": branches,
        "combat_policy": f"mcts_{mcts_sims}",
    }


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
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    skipped_wide = 0
    skipped_single = 0

    with output.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            gc = sts.GameContext(sts.CharacterClass.IRONCLAD, int(seed), 0)
            agent = sts.Agent()
            _set_pauses(agent)
            steps = 0
            branch_points = 0
            kind_counts: Counter[str] = Counter()

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

                can_branch = (
                    branch_points < max_branch_points_per_game
                    and kind_counts[str(kind)] < max_branch_points_per_kind_per_game
                    and choices <= max_choice_branches
                )
                if can_branch:
                    example = _branch_example(
                        gc,
                        sts=sts,
                        armg=armg,
                        mcts_sims=mcts_sims,
                        max_game_steps=max_game_steps,
                        max_battle_steps=max_battle_steps,
                        temperature=temperature,
                    )
                    examples.append(example)
                    handle.write(
                        json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    handle.flush()
                    branch_points += 1
                    kind_counts[str(kind)] += 1

                    best_index = int(example["teacher_best_index"])
                    # Execute the teacher choice on the untouched original state.
                    kind_now, descs_now, execs_now = armg.choices(gc)
                    if (
                        str(kind_now) != str(kind)
                        or _desc_rows(descs_now) != example["descs"]
                        or len(execs_now) != len(example["descs"])
                    ):
                        raise RuntimeError("original ArmG choice identity changed after branches")
                    execs_now[best_index](gc)
                else:
                    if choices > max_choice_branches:
                        skipped_wide += 1
                    armg.step(gc, sts)

            if gc.outcome == sts.GameOutcome.UNDECIDED:
                raise RuntimeError(f"strategy dataset game step bound reached seed={seed}")
            games.append(
                {
                    "seed": int(seed),
                    "outcome": (
                        "victory"
                        if gc.outcome == sts.GameOutcome.PLAYER_VICTORY
                        else "defeat"
                    ),
                    "final_floor": int(getattr(gc, "floor_num", 0) or 0),
                    "branch_points": branch_points,
                    "kinds": dict(kind_counts),
                }
            )

    if not examples:
        raise RuntimeError("strategy branch rollout produced no trainable examples")

    by_kind = Counter(str(row["kind"]) for row in examples)
    agreement = sum(
        int(row["current_armg_index"] == row["teacher_best_index"])
        for row in examples
    ) / len(examples)
    return {
        "schema_version": STRATEGY_DATASET_SCHEMA_VERSION,
        "combat_policy": f"mcts_{mcts_sims}",
        "training_seed_count": len(seeds),
        "example_count": len(examples),
        "examples_by_kind": dict(sorted(by_kind.items())),
        "teacher_agreement": agreement,
        "mistake_examples": sum(
            int(row["current_armg_index"] != row["teacher_best_index"])
            for row in examples
        ),
        "skipped_wide_decisions": skipped_wide,
        "skipped_single_choice_decisions": skipped_single,
        "games": games,
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
) -> dict[str, Any]:
    """Deduplicate, prioritize mistakes, and preserve decision-kind diversity."""
    if max_examples < 1:
        raise RuntimeError("max replay examples must be positive")
    rows = _read_examples(replay_path) + _read_examples(new_dataset_path)

    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _example_identity(row)
        old = dedup.get(key)
        if old is None or float(row.get("priority", 1.0)) >= float(old.get("priority", 1.0)):
            dedup[key] = row

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dedup.values():
        groups[str(row.get("kind", "unknown"))].append(row)
    for values in groups.values():
        values.sort(
            key=lambda row: (
                float(row.get("priority", 1.0)),
                float(row.get("teacher_margin", 0.0)),
            ),
            reverse=True,
        )

    # Round-robin kinds so map/card/shop/rest/event cannot silently starve.
    kept: list[dict[str, Any]] = []
    kind_names = sorted(groups)
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
    return {
        "examples": len(kept),
        "examples_by_kind": dict(sorted(by_kind.items())),
        "mistake_examples": sum(
            int(row.get("current_armg_index") != row.get("teacher_best_index"))
            for row in kept
        ),
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
) -> dict[str, Any]:
    examples = _read_examples(replay_path)
    if not examples:
        raise RuntimeError("strategy replay is empty")

    module = armg.module
    candidate = module.Scorer((128, 128))
    candidate.load_state_dict(
        torch.load(source_weight, weights_only=True, map_location="cpu")
    )
    candidate.train()
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
    anchor_losses: list[float] = []
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for index in order:
            row = examples[index]
            obs = torch.tensor(row["obs"], dtype=torch.float32)
            scores = candidate.score(obs, row["descs"])
            target = torch.tensor(row["target_probs"], dtype=torch.float32)
            ce = -(target * torch.log_softmax(scores, dim=0)).sum()

            anchor_terms = [
                (param - source_params[name]).pow(2).mean()
                for name, param in candidate.named_parameters()
            ]
            anchor = torch.stack(anchor_terms).mean()

            priority = float(row.get("priority", 1.0))
            balance = kind_weight[str(row.get("kind", "unknown"))]
            loss = priority * balance * ce + anchor_coef * anchor

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            ce_losses.append(float(ce.item()))
            anchor_losses.append(float(anchor.item()))

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
        "before_teacher_top1": before,
        "after_teacher_top1": after,
        "before_teacher_top1_by_kind": before_by_kind,
        "after_teacher_top1_by_kind": after_by_kind,
        "mean_loss": sum(losses) / len(losses),
        "mean_cross_entropy": sum(ce_losses) / len(ce_losses),
        "mean_anchor_loss": sum(anchor_losses) / len(anchor_losses),
        "source_sha256": _sha256(source_weight),
        "candidate_sha256": _sha256(output_weight),
    }


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
        if int(payload.get("combat_mcts_sims", -1)) != combat_mcts_sims:
            raise RuntimeError(
                "combat MCTS budget is frozen for one Strategy lineage; "
                "start a new state-dir to change it"
            )
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
    }
    _write_json(state_path, payload)
    return payload


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
    parser.add_argument("--max-game-steps", type=int, default=600)
    parser.add_argument("--max-battle-steps", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-coef", type=float, default=0.01)
    parser.add_argument("--max-replay-examples", type=int, default=10000)
    parser.add_argument("--max-stagnation", type=int, default=5)
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
    if args.epochs < 1 or args.max_stagnation < 1:
        raise RuntimeError("epochs/max-stagnation must be positive")

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

    sts = _load_sts(args.module_dir)
    clone_probe = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)
    if not callable(getattr(clone_probe, "clone", None)):
        raise RuntimeError("GameContext.clone API unavailable; Strategy teacher cannot run")

    current_weight = args.state_dir / "current-strategy.pt"
    replay_path = args.state_dir / "strategy-replay.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    for _ in range(args.rounds):
        if int(state.get("stagnation_count", 0)) >= args.max_stagnation:
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
        )
        replay_report = _merge_replay(
            replay_path,
            dataset_path,
            max_examples=args.max_replay_examples,
        )

        candidate_weight = round_dir / "candidate-strategy.pt"
        train_report = _train_candidate(
            armg=armg,
            source_weight=current_weight,
            replay_path=replay_path,
            output_weight=candidate_weight,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            anchor_coef=args.anchor_coef,
            rng_seed=args.rng_seed + round_no * 1000 + 2,
        )

        current_dev, candidate_dev, dev_gate = _eval_and_gate(
            current_weight=current_weight,
            candidate_weight=candidate_weight,
            seeds=dev30,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            mcts_sims=args.combat_mcts_sims,
            workers=args.eval_workers,
            policy=DEV_STRATEGY_GATE,
        )
        _write_json(round_dir / "eval-current-dev30.json", current_dev)
        _write_json(round_dir / "eval-candidate-dev30.json", candidate_dev)

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
        if promoted:
            shutil.copy2(candidate_weight, current_weight)
            if _sha256(current_weight) != _sha256(candidate_weight):
                raise RuntimeError("Strategy promotion checkpoint checksum mismatch")

        consumed_eval = list(hidden_seeds) + list(fresh_seeds)
        state = {
            **state,
            "generation": int(state.get("generation", 0)) + int(promoted),
            "accepted_rounds": int(state.get("accepted_rounds", 0)) + int(promoted),
            "rejected_rounds": int(state.get("rejected_rounds", 0)) + int(not promoted),
            "stagnation_count": 0 if promoted else int(state.get("stagnation_count", 0)) + 1,
            "used_training_seeds": list(state.get("used_training_seeds", []))
            + list(train_seeds),
            "used_evaluation_seeds": list(state.get("used_evaluation_seeds", []))
            + consumed_eval,
            "current_strategy_sha256": _sha256(current_weight),
        }
        _write_json(args.state_dir / "strategy-state.json", state)

        report = {
            "schema_version": ROUND_SCHEMA_VERSION,
            "round": round_no,
            "combat_training_enabled": False,
            "combat_policy": f"mcts_{args.combat_mcts_sims}",
            "strategy_scope": "all_armg_noncombat_choice_kinds",
            "current_strategy_sha_before": before_sha,
            "current_strategy_sha_after": _sha256(current_weight),
            "candidate_strategy_sha256": _sha256(candidate_weight),
            "training_seeds": list(train_seeds),
            "hidden_eval_seeds": list(hidden_seeds),
            "fresh_eval_seeds": list(fresh_seeds),
            "dataset": dataset_report,
            "replay": replay_report,
            "training": train_report,
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
        "generation": int(state.get("generation", 0)),
        "accepted_rounds": int(state.get("accepted_rounds", 0)),
        "rejected_rounds": int(state.get("rejected_rounds", 0)),
        "stagnation_count": int(state.get("stagnation_count", 0)),
        "paused_for_stagnation": (
            int(state.get("stagnation_count", 0)) >= args.max_stagnation
        ),
        "current_strategy_sha256": _sha256(current_weight),
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
