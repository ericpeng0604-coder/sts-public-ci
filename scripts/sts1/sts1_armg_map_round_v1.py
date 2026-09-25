#!/usr/bin/env python3
"""One resumable ArmG map self-improvement round.

The map component learns from exact GameContext branch rollouts. Non-map ArmG
behavior stays on the frozen upstream weight so map training cannot silently
damage card/shop/rest/event decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

import torch

from roguelike_ai.sts1_phase3.armg_map_evolve import (
    ARMG_MAP_DATASET_SCHEMA_VERSION,
    ArmGMapGatePolicy,
    FAST_MAP_GATE,
    FORMAL_MAP_GATE,
    armg_map_promotion_decision,
    branch_quality,
    evaluate_armg_map_gate,
    soft_branch_targets,
)
from roguelike_ai.sts1_phase3.frozen_student import FrozenStudentV0
from roguelike_ai.sts1_phase3.hybrid_model import DEFAULT_HYBRID_MCTS_BUDGETS
from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    SimulatorCombatAdapter,
    SimulatorRunError,
    _hybrid_vote_choice_bits,
    _load_sts,
    _project_legal_actions,
    _set_pauses,
    public_run_state,
    run_simulator_game,
)
from roguelike_ai.sts1_phase3.student_v1_ppo import StudentV1Config, StudentV1PPO


STATE_SCHEMA = "sts1-armg-map-loop-state-v1"
ROUND_SCHEMA = "sts1-armg-map-round-v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_seeds(path: Path) -> tuple[int, ...]:
    seeds = tuple(
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise RuntimeError(f"invalid seed file: {path}")
    return seeds


def _fresh_seeds(
    count: int,
    *,
    rng_seed: int,
    forbidden: set[int],
) -> tuple[int, ...]:
    rng = random.Random(rng_seed)
    values: list[int] = []
    seen = set(forbidden)
    while len(values) < count:
        value = rng.randint(1, 10**9)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)


def _load_or_init_state(state_dir: Path, base_weight: Path) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "armg-map-state.json"
    current = state_dir / "current-map.pt"
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != STATE_SCHEMA:
            raise RuntimeError("ArmG map state schema mismatch")
        if not current.is_file():
            raise RuntimeError("ArmG map state is missing current-map.pt")
        if payload.get("current_map_sha256") != _sha256(current):
            raise RuntimeError("ArmG current map checkpoint SHA drift")
        return payload

    shutil.copy2(base_weight, current)
    payload = {
        "schema_version": STATE_SCHEMA,
        "generation": 0,
        "accepted_rounds": 0,
        "rejected_rounds": 0,
        "used_training_seeds": [],
        "current_map_sha256": _sha256(current),
        "base_nonmap_sha256": _sha256(base_weight),
    }
    state_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_state(state_dir: Path, payload: Mapping[str, Any]) -> None:
    path = state_dir / "armg-map-state.json"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _drive_hybrid_battle(
    gc: Any,
    *,
    sts: Any,
    student: StudentV1PPO,
    budgets: Sequence[int],
    max_battle_steps: int,
) -> None:
    battle = sts.BattleContext()
    battle.init(gc)
    adapter = SimulatorCombatAdapter()
    steps = 0
    while battle.outcome == sts.Outcome.UNDECIDED and steps < max_battle_steps:
        steps += 1
        native_actions = list(sts.get_legal_actions(battle))
        if not native_actions:
            raise RuntimeError("branch rollout battle exposed no legal action")
        if len(native_actions) == 1:
            native_actions[0].execute(battle)
            continue

        hand_raw = list(getattr(battle, "hand", []))
        public_actions, native_index_map, _ = _project_legal_actions(native_actions, hand_raw)
        public_state = adapter.adapt(
            battle,
            legal_actions=native_actions,
            run_state=public_run_state(gc),
            projected_legal_actions=public_actions,
        )

        recommendations: list[tuple[int, Any, int]] = []
        for budget in budgets:
            rec = sts.mcts_recommend(battle, int(budget))
            if rec is None:
                raise RuntimeError(f"branch rollout MCTS returned no action for budget={budget}")
            bits = getattr(rec, "bits", None)
            if not isinstance(bits, int):
                raise RuntimeError("branch rollout MCTS action has no integer bits")
            recommendations.append((int(budget), rec, bits))

        decision = student.select_action(public_state, require_command=False)
        if not 0 <= int(decision.action_index) < len(native_index_map):
            raise RuntimeError("branch rollout Student selected illegal public action")
        student_native = native_actions[native_index_map[int(decision.action_index)]]
        student_bits = getattr(student_native, "bits", None)
        if not isinstance(student_bits, int):
            raise RuntimeError("branch rollout Student action has no integer bits")

        chosen_bits, _ = _hybrid_vote_choice_bits(
            [(budget, bits) for budget, _, bits in recommendations],
            student_bits=student_bits,
        )
        chosen = next(
            rec
            for _, rec, bits in reversed(recommendations)
            if bits == chosen_bits
        )
        chosen.execute(battle)

    if battle.outcome == sts.Outcome.UNDECIDED:
        raise RuntimeError("branch rollout battle step bound reached")
    battle.exit_battle(gc)


def _play_from_state(
    gc: Any,
    *,
    sts: Any,
    student: StudentV1PPO,
    armg: ArmGNoncombatPolicy,
    budgets: Sequence[int],
    max_game_steps: int,
    max_battle_steps: int,
) -> dict[str, Any]:
    agent = sts.Agent()
    _set_pauses(agent)
    steps = 0
    while gc.outcome == sts.GameOutcome.UNDECIDED and steps < max_game_steps:
        steps += 1
        agent.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        if gc.screen_state == sts.ScreenState.BATTLE:
            _drive_hybrid_battle(
                gc,
                sts=sts,
                student=student,
                budgets=budgets,
                max_battle_steps=max_battle_steps,
            )
        else:
            armg.step(gc, sts)

    if gc.outcome == sts.GameOutcome.UNDECIDED:
        raise RuntimeError("branch rollout game step bound reached")
    outcome = (
        "victory"
        if gc.outcome == sts.GameOutcome.PLAYER_VICTORY
        else "defeat"
    )
    return {
        "outcome": outcome,
        "final_floor": int(getattr(gc, "floor_num", 0) or 0),
        "final_hp": int(getattr(gc, "cur_hp", 0) or 0),
        "max_hp": int(getattr(gc, "max_hp", 1) or 1),
    }


def _clone_parity_signature(gc: Any, armg: ArmGNoncombatPolicy) -> dict[str, Any]:
    obs = [float(x) for x in armg.module.obs_vec(gc)]
    return {
        "seed": int(getattr(gc, "seed", 0)),
        "floor": int(getattr(gc, "floor_num", 0)),
        "act": int(getattr(gc, "act", 0)),
        "hp": int(getattr(gc, "cur_hp", 0)),
        "max_hp": int(getattr(gc, "max_hp", 0)),
        "gold": int(getattr(gc, "gold", 0)),
        "map_x": int(getattr(gc, "cur_map_node_x", -1)),
        "map_y": int(getattr(gc, "cur_map_node_y", -1)),
        "screen": str(getattr(gc, "screen_state", "")),
        "obs": obs,
    }


def _branch_example(
    gc: Any,
    *,
    sts: Any,
    student: StudentV1PPO,
    armg: ArmGNoncombatPolicy,
    budgets: Sequence[int],
    max_game_steps: int,
    max_battle_steps: int,
    temperature: float,
) -> dict[str, Any]:
    if not callable(getattr(gc, "clone", None)):
        raise RuntimeError("pinned simulator does not expose GameContext.clone()")

    kind, descs, scores = armg.score_choices(gc)
    if kind != "map" or len(descs) < 2:
        raise RuntimeError("branch example requires a real map decision")
    actions = list(sts.get_legal_game_actions(gc))
    if len(actions) != len(descs):
        raise RuntimeError("ArmG map descriptors and legal game actions diverged")

    before = _clone_parity_signature(gc, armg)
    clone_probe = gc.clone()
    if _clone_parity_signature(clone_probe, armg) != before:
        raise RuntimeError("GameContext.clone public-state parity mismatch")

    obs = [float(x) for x in armg.module.obs_vec(gc)]
    current_index = int(torch.argmax(scores).item())
    branch_rows: list[dict[str, Any]] = []
    values: list[float] = []

    for index, action in enumerate(actions):
        branch = gc.clone()
        branch_actions = list(sts.get_legal_game_actions(branch))
        bits = getattr(action, "bits", None)
        matching = [candidate for candidate in branch_actions if getattr(candidate, "bits", None) == bits]
        if len(matching) != 1:
            raise RuntimeError(f"branch action identity mismatch at index={index}")
        matching[0].execute(branch)
        result = _play_from_state(
            branch,
            sts=sts,
            student=student,
            armg=armg,
            budgets=budgets,
            max_game_steps=max_game_steps,
            max_battle_steps=max_battle_steps,
        )
        value = branch_quality(
            outcome=result["outcome"],
            final_floor=result["final_floor"],
            final_hp=result["final_hp"],
            max_hp=result["max_hp"],
        )
        values.append(value)
        branch_rows.append({
            "index": index,
            "action_bits": bits,
            **result,
            "quality": value,
        })

    after = _clone_parity_signature(gc, armg)
    if after != before:
        raise RuntimeError("branch rollout mutated original GameContext")

    targets = soft_branch_targets(values, temperature=temperature)
    best_index = max(range(len(values)), key=values.__getitem__)
    return {
        "schema_version": ARMG_MAP_DATASET_SCHEMA_VERSION,
        "type": "armg_map_branch_example",
        "seed": int(getattr(gc, "seed", 0)),
        "floor": int(getattr(gc, "floor_num", 0)),
        "act": int(getattr(gc, "act", 0)),
        "obs": obs,
        "descs": [[float(v) for v in desc] for desc in descs],
        "current_armg_index": current_index,
        "teacher_best_index": best_index,
        "branch_quality": values,
        "target_probs": list(targets),
        "branches": branch_rows,
    }


def _collect_dataset(
    *,
    seeds: Sequence[int],
    sts: Any,
    student: StudentV1PPO,
    armg: ArmGNoncombatPolicy,
    output: Path,
    budgets: Sequence[int],
    max_branch_points_per_game: int,
    max_game_steps: int,
    max_battle_steps: int,
    temperature: float,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    game_summaries: list[dict[str, Any]] = []

    with output.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            gc = sts.GameContext(sts.CharacterClass.IRONCLAD, int(seed), 0)
            agent = sts.Agent()
            _set_pauses(agent)
            branch_points = 0
            steps = 0
            while gc.outcome == sts.GameOutcome.UNDECIDED and steps < max_game_steps:
                steps += 1
                agent.playout(gc)
                if gc.outcome != sts.GameOutcome.UNDECIDED:
                    break

                if gc.screen_state == sts.ScreenState.BATTLE:
                    _drive_hybrid_battle(
                        gc,
                        sts=sts,
                        student=student,
                        budgets=budgets,
                        max_battle_steps=max_battle_steps,
                    )
                    continue

                kind, descs, _ = armg.choices(gc)
                if (
                    kind == "map"
                    and len(descs) > 1
                    and branch_points < max_branch_points_per_game
                ):
                    example = _branch_example(
                        gc,
                        sts=sts,
                        student=student,
                        armg=armg,
                        budgets=budgets,
                        max_game_steps=max_game_steps,
                        max_battle_steps=max_battle_steps,
                        temperature=temperature,
                    )
                    rows.append(example)
                    handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    branch_points += 1

                    # Follow the branch-rollout teacher's best path so later
                    # examples come from progressively stronger trajectories.
                    actions = list(sts.get_legal_game_actions(gc))
                    actions[int(example["teacher_best_index"])].execute(gc)
                else:
                    armg.step(gc, sts)

            if gc.outcome == sts.GameOutcome.UNDECIDED:
                raise RuntimeError(f"dataset collection game step bound reached for seed={seed}")
            game_summaries.append({
                "seed": int(seed),
                "outcome": (
                    "victory"
                    if gc.outcome == sts.GameOutcome.PLAYER_VICTORY
                    else "defeat"
                ),
                "final_floor": int(getattr(gc, "floor_num", 0) or 0),
                "branch_points": branch_points,
            })

    if not rows:
        raise RuntimeError("ArmG branch rollout produced no map examples")
    return {
        "schema_version": ARMG_MAP_DATASET_SCHEMA_VERSION,
        "training_seed_count": len(seeds),
        "example_count": len(rows),
        "teacher_agreement": (
            sum(int(row["current_armg_index"] == row["teacher_best_index"]) for row in rows)
            / len(rows)
        ),
        "games": game_summaries,
    }


def _read_examples(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != ARMG_MAP_DATASET_SCHEMA_VERSION:
            raise RuntimeError("ArmG map dataset schema mismatch")
        rows.append(row)
    if not rows:
        raise RuntimeError("ArmG map dataset is empty")
    return rows


def _train_map_candidate(
    *,
    armg: ArmGNoncombatPolicy,
    source_weight: Path,
    dataset_path: Path,
    output_weight: Path,
    epochs: int,
    learning_rate: float,
    anchor_coef: float,
    rng_seed: int,
) -> dict[str, Any]:
    examples = _read_examples(dataset_path)
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

    def accuracy(net: Any) -> float:
        net.eval()
        correct = 0
        with torch.no_grad():
            for row in examples:
                obs = torch.tensor(row["obs"], dtype=torch.float32)
                scores = net.score(obs, row["descs"])
                correct += int(int(torch.argmax(scores).item()) == int(row["teacher_best_index"]))
        net.train()
        return correct / len(examples)

    before_acc = accuracy(candidate)
    rng = random.Random(rng_seed)
    losses: list[float] = []
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for idx in order:
            row = examples[idx]
            obs = torch.tensor(row["obs"], dtype=torch.float32)
            scores = candidate.score(obs, row["descs"])
            target = torch.tensor(row["target_probs"], dtype=torch.float32)
            ce = -(target * torch.log_softmax(scores, dim=0)).sum()

            anchor_terms = []
            for name, param in candidate.named_parameters():
                anchor_terms.append((param - source_params[name]).pow(2).mean())
            anchor = torch.stack(anchor_terms).mean()
            loss = ce + anchor_coef * anchor

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

    after_acc = accuracy(candidate)
    candidate.eval()
    output_weight.parent.mkdir(parents=True, exist_ok=True)
    torch.save(candidate.state_dict(), output_weight)
    return {
        "examples": len(examples),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "anchor_coef": anchor_coef,
        "before_teacher_top1": before_acc,
        "after_teacher_top1": after_acc,
        "mean_loss": sum(losses) / len(losses),
        "source_sha256": _sha256(source_weight),
        "candidate_sha256": _sha256(output_weight),
    }


def _evaluate_weight(
    *,
    seeds: Sequence[int],
    sts: Any,
    student: StudentV1PPO,
    armg_root: Path,
    base_weight: Path,
    map_weight: Path,
) -> dict[str, Any]:
    armg = ArmGNoncombatPolicy(
        root=armg_root,
        weight_path=base_weight,
        map_weight_path=map_weight,
    )
    runs = []
    for seed in seeds:
        summary = run_simulator_game(
            student=student,
            sts=sts,
            seed=int(seed),
            armg_policy=armg,
            hybrid_mcts_budgets=DEFAULT_HYBRID_MCTS_BUDGETS,
            heldout_seeds=seeds,
            collect_ppo=False,
        )
        runs.append(summary)
    return {
        "schema_version": "sts1-armg-map-eval-v1",
        "map_weight_sha256": _sha256(map_weight),
        "seeds": len(seeds),
        "runs": runs,
    }


def _load_student(
    baseline_v0: Path,
    student_v1: Path | None,
) -> tuple[FrozenStudentV0, StudentV1PPO]:
    baseline = FrozenStudentV0.from_path(baseline_v0)
    if student_v1 is not None and student_v1.is_file():
        return baseline, StudentV1PPO.load(student_v1, baseline, device="cpu")
    torch.manual_seed(20260925)
    return baseline, StudentV1PPO(
        baseline,
        config=StudentV1Config(),
        generation=0,
        device="cpu",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-v0", type=Path, required=True)
    parser.add_argument("--student-v1", type=Path)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--armg-base-weight", type=Path, required=True)
    parser.add_argument("--eval-seed-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-seeds", type=int, default=8)
    parser.add_argument("--rng-seed", type=int, default=20261002)
    parser.add_argument("--max-branch-points-per-game", type=int, default=4)
    parser.add_argument("--max-game-steps", type=int, default=600)
    parser.add_argument("--max-battle-steps", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-coef", type=float, default=0.01)
    parser.add_argument("--smoke-eval-count", type=int, default=0)
    args = parser.parse_args()

    eval_seeds = _read_seeds(args.eval_seed_file)
    if len(eval_seeds) != 50:
        raise RuntimeError("ArmG map eval seed file must contain exactly 50 seeds")

    state = _load_or_init_state(args.state_dir, args.armg_base_weight)
    current_map = args.state_dir / "current-map.pt"
    _, student = _load_student(args.baseline_v0, args.student_v1)

    sts = _load_sts(args.module_dir)
    if not callable(getattr(sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0), "clone", None)):
        raise RuntimeError("GameContext.clone API is unavailable; branch rollout cannot proceed")

    armg = ArmGNoncombatPolicy(
        root=args.armg_root,
        weight_path=args.armg_base_weight,
        map_weight_path=current_map,
    )

    forbidden = set(eval_seeds) | {int(x) for x in state.get("used_training_seeds", [])}
    training_seeds = _fresh_seeds(
        args.training_seeds,
        rng_seed=args.rng_seed + int(state["generation"]) * 1000,
        forbidden=forbidden,
    )

    round_no = int(state["accepted_rounds"]) + int(state["rejected_rounds"]) + 1
    round_dir = args.output_dir / f"round-{round_no:04d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = round_dir / "map-branch-dataset.jsonl"

    dataset_report = _collect_dataset(
        seeds=training_seeds,
        sts=sts,
        student=student,
        armg=armg,
        output=dataset_path,
        budgets=DEFAULT_HYBRID_MCTS_BUDGETS,
        max_branch_points_per_game=args.max_branch_points_per_game,
        max_game_steps=args.max_game_steps,
        max_battle_steps=args.max_battle_steps,
        temperature=args.temperature,
    )

    candidate_path = round_dir / "candidate-map.pt"
    train_report = _train_map_candidate(
        armg=armg,
        source_weight=current_map,
        dataset_path=dataset_path,
        output_weight=candidate_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        anchor_coef=args.anchor_coef,
        rng_seed=args.rng_seed + round_no,
    )

    eval_count = args.smoke_eval_count or 50
    if eval_count < 1 or eval_count > 50:
        raise RuntimeError("smoke eval count must be 0 or within 1..50")

    if args.smoke_eval_count:
        current_eval = _evaluate_weight(
            seeds=eval_seeds[:eval_count],
            sts=sts,
            student=student,
            armg_root=args.armg_root,
            base_weight=args.armg_base_weight,
            map_weight=current_map,
        )
        candidate_eval = _evaluate_weight(
            seeds=eval_seeds[:eval_count],
            sts=sts,
            student=student,
            armg_root=args.armg_root,
            base_weight=args.armg_base_weight,
            map_weight=candidate_path,
        )
        gate = evaluate_armg_map_gate(
            current_eval,
            candidate_eval,
            policy=ArmGMapGatePolicy(
                f"armg-map-smoke-{eval_count}",
                eval_count,
                min_floor_delta=0.0,
            ),
        )
        promotion = {
            "schema_version": "sts1-armg-map-smoke-v1",
            "decision": "SMOKE_ONLY_NO_PROMOTION",
            "gate": gate,
        }
    else:
        current30 = _evaluate_weight(
            seeds=eval_seeds[:30],
            sts=sts,
            student=student,
            armg_root=args.armg_root,
            base_weight=args.armg_base_weight,
            map_weight=current_map,
        )
        candidate30 = _evaluate_weight(
            seeds=eval_seeds[:30],
            sts=sts,
            student=student,
            armg_root=args.armg_root,
            base_weight=args.armg_base_weight,
            map_weight=candidate_path,
        )
        gate30 = evaluate_armg_map_gate(
            current30,
            candidate30,
            policy=FAST_MAP_GATE,
        )
        if gate30["status"] == "PASS":
            current50 = {
                **current30,
                "seeds": 50,
                "runs": current30["runs"] + _evaluate_weight(
                    seeds=eval_seeds[30:],
                    sts=sts,
                    student=student,
                    armg_root=args.armg_root,
                    base_weight=args.armg_base_weight,
                    map_weight=current_map,
                )["runs"],
            }
            candidate50 = {
                **candidate30,
                "seeds": 50,
                "runs": candidate30["runs"] + _evaluate_weight(
                    seeds=eval_seeds[30:],
                    sts=sts,
                    student=student,
                    armg_root=args.armg_root,
                    base_weight=args.armg_base_weight,
                    map_weight=candidate_path,
                )["runs"],
            }
            gate50 = evaluate_armg_map_gate(
                current50,
                candidate50,
                policy=FORMAL_MAP_GATE,
            )
        else:
            gate50 = {
                "schema_version": "sts1-armg-map-gate-v1",
                "gate": "armg-map-fixed-seed-50",
                "status": "SKIPPED",
                "reasons": ["fast_map_gate_did_not_pass"],
            }
        promotion = armg_map_promotion_decision(gate30, gate50)

        if promotion["decision"] == "PROMOTE_MAP":
            shutil.copy2(candidate_path, current_map)
            state = {
                **state,
                "generation": int(state["generation"]) + 1,
                "accepted_rounds": int(state["accepted_rounds"]) + 1,
                "used_training_seeds": list(state.get("used_training_seeds", [])) + list(training_seeds),
                "current_map_sha256": _sha256(current_map),
            }
        else:
            state = {
                **state,
                "rejected_rounds": int(state["rejected_rounds"]) + 1,
                "used_training_seeds": list(state.get("used_training_seeds", [])) + list(training_seeds),
                "current_map_sha256": _sha256(current_map),
            }
        _write_state(args.state_dir, state)

    report = {
        "schema_version": ROUND_SCHEMA,
        "round": round_no,
        "map_generation_before": int(state.get("generation", 0)) if args.smoke_eval_count else (
            int(state["generation"]) - 1 if promotion["decision"] == "PROMOTE_MAP" else int(state["generation"])
        ),
        "map_generation_after": int(state.get("generation", 0)),
        "training_seeds": list(training_seeds),
        "dataset": dataset_report,
        "training": train_report,
        "promotion": promotion,
        "current_map_sha256": _sha256(current_map),
        "candidate_map_sha256": _sha256(candidate_path),
        "base_nonmap_sha256": _sha256(args.armg_base_weight),
        "student_checkpoint_used": (
            _sha256(args.student_v1)
            if args.student_v1 is not None and args.student_v1.is_file()
            else "student-v1-gen0-from-frozen-v0"
        ),
        "hybrid_mcts_budgets": list(DEFAULT_HYBRID_MCTS_BUDGETS),
    }
    (round_dir / "round-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ARMG_MAP_ROUND", json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
