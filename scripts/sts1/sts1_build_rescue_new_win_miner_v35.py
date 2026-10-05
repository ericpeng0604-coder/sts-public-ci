#!/usr/bin/env python3
"""Mine new STS1 wins plus verified failed alternatives for v3.5.

The miner replays the frozen G7 production ArmG policy from the original seed.
It first proves the loss is still Build-limited under Boss MCTS-50k, then scans
up to the last 20 non-combat decisions. One-step interventions are tried first;
if none rescue the target Boss, bounded two-step combinations are attempted.

Every accepted teacher must pass both Boss-10k and Boss-50k confirmation.
Combat policy outside the target Boss remains MCTS-2000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.simulator import (
    ArmGNoncombatPolicy,
    _load_sts,
    run_simulator_game,
)

SCHEMA = "sts1-armg-strategy-branch-dataset-v1"
SAFETY = ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count")
BOSS_FLOORS = (16, 33, 50)


class StaleForcedAction(RuntimeError):
    """The replay no longer presents the action state a counterfactual targets."""


def _read_seeds(path: Path) -> list[int]:
    rows = [
        int(x.strip())
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]
    if not rows or len(rows) != len(set(rows)):
        raise RuntimeError(f"seed file must be non-empty and unique: {path}")
    return rows


def _safe(result: dict[str, Any]) -> bool:
    return (
        result.get("result") == "PASS_SIMULATOR_COMPLETE_RUN"
        and all(int(result.get(k, 0) or 0) == 0 for k in SAFETY)
    )


def _floor(result: dict[str, Any]) -> int:
    return int(result.get("final_floor") or result.get("max_floor") or 0)


def _is_win(result: dict[str, Any]) -> bool:
    return str(result.get("outcome", "")).lower() == "victory"


def _target_boss(final_floor: int) -> int:
    if final_floor <= 16:
        return 16
    if final_floor <= 33:
        return 33
    return 50


def _passes_target(result: dict[str, Any], target: int) -> bool:
    return _is_win(result) or _floor(result) > int(target)


class ReplayProductionArmG(ArmGNoncombatPolicy):
    """Production deterministic ArmG with replay-from-seed forced choices."""

    def __init__(
        self,
        *,
        root: Path,
        weight_path: Path,
        forced: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(root=root, weight_path=weight_path)
        self.forced = dict(forced or {})
        self.branch_index = 0
        self.records: list[dict[str, Any]] = []

    def decide(self, gc: Any, sts: Any):
        kind, selected, descs, execs, scores = super().decide(gc, sts)
        if selected < 0 or len(descs) < 2:
            return kind, selected, descs, execs, scores

        idx = self.branch_index
        self.branch_index += 1
        snapshot = self.training_vector_snapshot(gc, descs)
        record = {
            "branch_index": int(idx),
            "kind": str(kind),
            "selected_index": int(selected),
            "scores": [float(v) for v in scores],
            "floor": int(getattr(gc, "floor_num", 0) or 0),
            "act": int(getattr(gc, "act", 0) or 0),
            "hp": int(getattr(gc, "cur_hp", 0) or 0),
            "max_hp": int(getattr(gc, "max_hp", 1) or 1),
            "obs": list(snapshot["obs_412"]),
            "descs": [list(row) for row in snapshot["candidate_desc_368"]],
            "semantics": [self.describe_choice(str(kind), d) for d in descs],
        }

        force = self.forced.get(idx)
        if force is not None:
            if str(force["kind"]) != str(kind):
                raise StaleForcedAction(
                    f"forced kind drift branch={idx}: {force['kind']} != {kind}"
                )
            if force["descs"] != record["descs"]:
                raise StaleForcedAction(f"forced candidate identity drift branch={idx}")
            forced_index = int(force["index"])
            if not 0 <= forced_index < len(descs):
                raise StaleForcedAction(
                    f"forced index {forced_index} outside {len(descs)} candidates"
                )
            record["unforced_index"] = int(selected)
            record["forced_index"] = forced_index
            selected = forced_index

        self.records.append(record)
        return kind, int(selected), descs, execs, scores


def _force_spec(record: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "kind": str(record["kind"]),
        "descs": record["descs"],
        "index": int(index),
    }


def _run_variant(
    *,
    seed: int,
    module_dir: Path,
    armg_root: Path,
    weight: Path,
    heldout: list[int],
    boss_sims: int | None,
    forced: dict[int, dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sts = _load_sts(module_dir)
    policy = ReplayProductionArmG(
        root=armg_root,
        weight_path=weight,
        forced=forced,
    )
    result = run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=None,
        armg_policy=policy,
        combat_mcts_sims=2000,
        combat_mcts_boss_sims=boss_sims,
        combat_mcts_boss_floors=BOSS_FLOORS,
        heldout_seeds=heldout,
    )
    if not _safe(result):
        raise RuntimeError(f"unsafe/incomplete seed {seed}: {result}")
    return dict(result), list(policy.records)


def _isolated_variant(
    *,
    seed: int,
    module_dir: Path,
    armg_root: Path,
    weight: Path,
    heldout: list[int],
    boss_sims: int | None,
    forced: dict[int, dict[str, Any]] | None,
    timeout_seconds: int,
    retries: int,
    deadline: float | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not hasattr(os, "fork"):
        raise RuntimeError("v2.4 miner requires os.fork on the GitHub runner")

    for attempt in range(retries + 1):
        if deadline is not None and time.monotonic() >= deadline:
            failures.append(
                {"attempt": attempt, "kind": "seed_deadline", "scope": "seed"}
            )
            break

        fd, name = tempfile.mkstemp(prefix="sts1-v24-", suffix=".json")
        os.close(fd)
        pid = os.fork()
        if pid == 0:
            try:
                result, records = _run_variant(
                    seed=seed,
                    module_dir=module_dir,
                    armg_root=armg_root,
                    weight=weight,
                    heldout=heldout,
                    boss_sims=boss_sims,
                    forced=forced,
                )
                Path(name).write_text(
                    json.dumps(
                        {"ok": True, "result": result, "records": records},
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os._exit(0)
            except BaseException as exc:
                try:
                    Path(name).write_text(
                        json.dumps(
                            {
                                "ok": False,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                finally:
                    os._exit(2)

        attempt_deadline = time.monotonic() + timeout_seconds
        if deadline is not None:
            attempt_deadline = min(attempt_deadline, deadline)
        status = None
        timed_out = False
        seed_deadline_hit = False
        while status is None:
            waited, raw = os.waitpid(pid, os.WNOHANG)
            if waited == pid:
                status = raw
                break
            now = time.monotonic()
            if now >= attempt_deadline:
                timed_out = True
                seed_deadline_hit = deadline is not None and now >= deadline
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status = os.waitpid(pid, 0)
                break
            time.sleep(0.05)

        payload = None
        try:
            p = Path(name)
            if p.exists() and p.stat().st_size:
                payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(
                {
                    "attempt": attempt,
                    "kind": "decode_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            try:
                Path(name).unlink()
            except FileNotFoundError:
                pass

        if (
            not timed_out
            and status is not None
            and os.WIFEXITED(status)
            and os.WEXITSTATUS(status) == 0
            and payload
            and payload.get("ok") is True
        ):
            return dict(payload["result"]), list(payload["records"]), failures

        if timed_out:
            failure = {
                "attempt": attempt,
                "kind": "timeout",
                "timeout_seconds": timeout_seconds,
                "scope": "seed_deadline" if seed_deadline_hit else "variant",
            }
        elif status is not None and os.WIFSIGNALED(status):
            sig = int(os.WTERMSIG(status))
            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = str(sig)
            failure = {
                "attempt": attempt,
                "kind": "native_signal",
                "signal": sig,
                "signal_name": sig_name,
            }
        elif payload and payload.get("ok") is False:
            error_type = payload.get("error_type")
            failure = {
                "attempt": attempt,
                "kind": "stale_forced_action" if error_type == "StaleForcedAction" else "python_exception",
                "error_type": error_type,
                "error": payload.get("error"),
            }
        else:
            failure = {
                "attempt": attempt,
                "kind": "child_exit",
                "exit_code": (
                    int(os.WEXITSTATUS(status))
                    if status is not None and os.WIFEXITED(status)
                    else None
                ),
            }
        failures.append(failure)

        # Deterministic Python/data-contract errors will not improve on retry.
        # Retry only transient child timeouts/signals/exits, and never past the
        # seed-wide deadline.
        if failure["kind"] in {"python_exception", "decode_error", "stale_forced_action"}:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break

    return None, [], failures

def _teacher(
    *,
    seed: int,
    record: dict[str, Any],
    alternative: int,
    target: int,
    boss10: dict[str, Any],
    boss50: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    probs = [0.0] * len(record["descs"])
    probs[int(alternative)] = 1.0
    full_win = _is_win(boss50)
    return {
        "schema_version": SCHEMA,
        "type": (
            "v24_new_win_full_victory"
            if full_win
            else "v24_new_win_boss_rescue"
        ),
        "source": "sts1-build-rescue-new-win-miner-v24",
        "seed": int(seed),
        "floor": int(record["floor"]),
        "act": int(record["act"]),
        "kind": str(record["kind"]),
        "obs": record["obs"],
        "descs": record["descs"],
        "current_armg_index": int(record["selected_index"]),
        "teacher_best_index": int(alternative),
        "target_probs": probs,
        "priority": 5.5 if full_win else 5.0,
        "confidence_weight": 1.0,
        "teacher_consensus_fraction": 1.0,
        "combat_policy": "mcts_2000",
        "confirmation_policy": "boss_mcts_10000_and_50000",
        "target_boss_floor": int(target),
        "intervention_step": int(step),
        "original_choice": record["semantics"][int(record["selected_index"])],
        "teacher_choice": record["semantics"][int(alternative)],
        "boss_10k": boss10,
        "boss_50k": boss50,
    }


def _negative_example(
    *,
    seed: int,
    record: dict[str, Any],
    alternative: int,
    target: int,
    stage: str,
    boss10: dict[str, Any],
    boss50: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "sts1-armg-negative-branch-v1",
        "type": "v35_failed_rescue_alternative",
        "source": "sts1-build-rescue-new-win-miner-v35",
        "seed": int(seed),
        "floor": int(record["floor"]),
        "act": int(record["act"]),
        "kind": str(record["kind"]),
        "obs": record["obs"],
        "descs": record["descs"],
        "current_armg_index": int(record["selected_index"]),
        "rejected_index": int(alternative),
        "target_boss_floor": int(target),
        "failure_stage": str(stage),
        "combat_policy": "mcts_2000",
        "confirmation_policy": (
            "boss_mcts_10000" if boss50 is None
            else "boss_mcts_10000_and_50000"
        ),
        "boss_10k": boss10,
        "boss_50k": boss50,
    }


def _rank_alternatives(record: dict[str, Any], limit: int) -> list[int]:
    selected = int(record["selected_index"])
    choices = [i for i in range(len(record["descs"])) if i != selected]
    scores = list(record.get("scores") or [])
    choices.sort(
        key=lambda i: float(scores[i]) if i < len(scores) else -1e30,
        reverse=True,
    )
    return choices[:limit]


def mine_seed(
    *,
    seed: int,
    module_dir: Path,
    armg_root: Path,
    weight: Path,
    heldout: list[int],
    max_states: int,
    max_alternatives: int,
    max_two_step_states: int,
    max_second_states: int,
    max_second_alternatives: int,
    timeout_seconds: int,
    retries: int,
    seed_timeout_seconds: int = 0,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    negative_examples: list[dict[str, Any]] = []
    deadline = time.monotonic() + seed_timeout_seconds if seed_timeout_seconds else None
    search_config = {
        "max_states": max_states,
        "max_alternatives": max_alternatives,
        "max_two_step_states": max_two_step_states,
        "max_second_states": max_second_states,
        "max_second_alternatives": max_second_alternatives,
    }
    weight_sha256 = hashlib.sha256(weight.read_bytes()).hexdigest()
    completed: dict[str, dict[str, Any]] = {}
    failed_variants: dict[str, dict[str, Any]] = {}
    checkpoint_schema = "sts1-v24-seed-checkpoint-v1"

    def save_checkpoint() -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": checkpoint_schema,
            "seed": int(seed),
            "weight_sha256": weight_sha256,
            "search_config": search_config,
            "completed_variants": completed,
            "failed_variants": failed_variants,
            "updated_at_epoch": time.time(),
        }
        temp_path = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_path, checkpoint)

    if checkpoint is not None and checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            saved.get("schema_version") != checkpoint_schema
            or int(saved.get("seed", -1)) != seed
            or saved.get("weight_sha256") != weight_sha256
            or saved.get("search_config") != search_config
        ):
            raise RuntimeError("checkpoint does not match this seed, G7 weight, or search configuration")
        completed = dict(saved.get("completed_variants") or {})
        failed_variants = dict(saved.get("failed_variants") or {})
        for item in failed_variants.values():
            failures.extend(list(item.get("errors") or []))

    budget_expired = False

    def variant_key(boss_sims: int | None, forced: dict[int, dict[str, Any]] | None) -> str:
        identity = json.dumps(
            {"boss_sims": boss_sims, "forced": forced},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def run(*, boss_sims: int | None, forced=None):
        nonlocal budget_expired
        key = variant_key(boss_sims, forced)
        cached = completed.get(key)
        if cached is not None:
            failures.extend(list(cached.get("failures") or []))
            return dict(cached["result"]), list(cached["records"])

        prior_failure = failed_variants.get(key)
        if prior_failure and prior_failure.get("terminal"):
            return None, []
        if prior_failure and int(prior_failure.get("rounds", 0)) >= 2:
            prior_failure["terminal"] = True
            save_checkpoint()
            return None, []
        if deadline is not None and time.monotonic() >= deadline:
            budget_expired = True
            return None, []

        result, records, errs = _isolated_variant(
            seed=seed,
            module_dir=module_dir,
            armg_root=armg_root,
            weight=weight,
            heldout=heldout,
            boss_sims=boss_sims,
            forced=forced,
            timeout_seconds=timeout_seconds,
            retries=retries,
            deadline=deadline,
        )
        failures.extend(errs)
        if result is not None:
            completed[key] = {
                "result": result,
                "records": records,
                "failures": errs,
            }
            failed_variants.pop(key, None)
        else:
            rounds = int(prior_failure.get("rounds", 0)) + 1 if prior_failure else 1
            retryable = bool(errs) and all(
                str(error.get("kind")) in {"timeout", "native_signal", "child_exit", "seed_deadline"}
                for error in errs
            )
            failed_variants[key] = {
                "rounds": rounds,
                "terminal": not retryable or rounds >= 2,
                "errors": list((prior_failure or {}).get("errors") or []) + errs,
            }
        if deadline is not None and time.monotonic() >= deadline:
            budget_expired = True
        save_checkpoint()
        return result, records

    def incomplete_result(base, baseline50=None, target=None, attempted_one=0, attempted_two=0, candidate_states=0):
        if not any(item.get("kind") == "seed_deadline" for item in failures):
            failures.append(
                {
                    "kind": "seed_deadline",
                    "seed_timeout_seconds": seed_timeout_seconds,
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                }
            )
        return {
            "seed": seed,
            "status": "SEED_SEARCH_INCOMPLETE",
            "base": base,
            "baseline_50k": baseline50,
            "target_boss_floor": target,
            "candidate_states": candidate_states,
            "attempted_one_step": attempted_one,
            "attempted_two_step": attempted_two,
            "negative_examples": list(negative_examples),
            "teachers": [],
            "failures": failures,
        }

    base, base_records = run(boss_sims=None)
    if base is None:
        if budget_expired:
            return incomplete_result(None)
        return {
            "seed": seed,
            "status": "BASE_REPLAY_FAILURE",
            "failures": failures,
            "negative_examples": list(negative_examples),
            "teachers": [],
        }
    if _is_win(base):
        return {
            "seed": seed,
            "status": "BASE_UNEXPECTED_WIN",
            "base": base,
            "negative_examples": list(negative_examples),
            "teachers": [],
            "failures": failures,
        }

    target = _target_boss(_floor(base))
    baseline50, _ = run(boss_sims=50000)
    if baseline50 is None:
        if budget_expired:
            return incomplete_result(base, target=target)
        return {
            "seed": seed,
            "status": "BASE_50K_FAILURE",
            "base": base,
            "target_boss_floor": target,
            "negative_examples": list(negative_examples),
            "teachers": [],
            "failures": failures,
        }
    if _passes_target(baseline50, target):
        return {
            "seed": seed,
            "status": "COMBAT_ONLY_RESCUED_50K",
            "base": base,
            "baseline_50k": baseline50,
            "target_boss_floor": target,
            "negative_examples": list(negative_examples),
            "teachers": [],
            "failures": failures,
        }

    records = [
        r
        for r in base_records
        if int(r.get("floor", 0) or 0) <= target
        and len(r.get("descs", [])) >= 2
    ]
    records = list(reversed(records[-max_states:]))
    if not records:
        return {
            "seed": seed,
            "status": "NO_REPLAYABLE_BUILD_STATE",
            "base": base,
            "baseline_50k": baseline50,
            "target_boss_floor": target,
            "negative_examples": list(negative_examples),
            "teachers": [],
            "failures": failures,
        }

    attempted_one = 0
    attempted_two = 0
    first_step_cache: list[
        tuple[dict[str, Any], int, dict[str, Any], list[dict[str, Any]]]
    ] = []

    for record in records:
        if budget_expired:
            break
        for alternative in _rank_alternatives(record, max_alternatives):
            if budget_expired:
                break
            attempted_one += 1
            forced = {
                int(record["branch_index"]): _force_spec(record, alternative)
            }
            boss10, records10 = run(boss_sims=10000, forced=forced)
            if boss10 is None:
                continue
            first_step_cache.append((record, alternative, boss10, records10))
            if not _passes_target(boss10, target):
                negative_examples.append(
                    _negative_example(
                        seed=seed, record=record, alternative=alternative,
                        target=target, stage="boss10_failed", boss10=boss10,
                    )
                )
                continue
            boss50, _ = run(boss_sims=50000, forced=forced)
            if boss50 is None:
                continue
            if not _passes_target(boss50, target):
                negative_examples.append(
                    _negative_example(
                        seed=seed, record=record, alternative=alternative,
                        target=target, stage="boss50_failed",
                        boss10=boss10, boss50=boss50,
                    )
                )
                continue

            teacher = _teacher(
                seed=seed,
                record=record,
                alternative=alternative,
                target=target,
                boss10=boss10,
                boss50=boss50,
                step=1,
            )
            return {
                "seed": seed,
                "status": (
                    "NEW_FULL_WIN_ONE_STEP"
                    if _is_win(boss50)
                    else "NEW_BOSS_RESCUE_ONE_STEP"
                ),
                "base": base,
                "baseline_50k": baseline50,
                "target_boss_floor": target,
                "attempted_one_step": attempted_one,
                "attempted_two_step": 0,
                "rescue": {
                    "first": {
                        "branch_index": int(record["branch_index"]),
                        "floor": int(record["floor"]),
                        "kind": str(record["kind"]),
                        "current_index": int(record["selected_index"]),
                        "alternative_index": int(alternative),
                    },
                    "boss_10k": boss10,
                    "boss_50k": boss50,
                },
                "negative_examples": list(negative_examples),
                "teachers": [teacher],
                "failures": failures,
            }

    if budget_expired:
        return incomplete_result(
            base, baseline50, target, attempted_one, attempted_two, len(records)
        )

    first_candidates = first_step_cache[: max_two_step_states * max_alternatives]
    for record, alternative, first10, records10 in first_candidates:
        if budget_expired:
            break
        first_idx = int(record["branch_index"])
        subsequent = [
            row
            for row in records10
            if int(row["branch_index"]) > first_idx
            and int(row.get("floor", 0) or 0) <= target
            and len(row.get("descs", [])) >= 2
        ]
        subsequent = list(reversed(subsequent[-max_second_states:]))
        if not subsequent:
            continue
        first_force = _force_spec(record, alternative)

        for second in subsequent:
            if budget_expired:
                break
            for second_alt in _rank_alternatives(
                second, max_second_alternatives
            ):
                if budget_expired:
                    break
                attempted_two += 1
                forced = {
                    first_idx: first_force,
                    int(second["branch_index"]): _force_spec(second, second_alt),
                }
                boss10, _ = run(boss_sims=10000, forced=forced)
                if boss10 is None or not _passes_target(boss10, target):
                    continue
                boss50, _ = run(boss_sims=50000, forced=forced)
                if boss50 is None or not _passes_target(boss50, target):
                    continue

                teachers = [
                    _teacher(
                        seed=seed,
                        record=record,
                        alternative=alternative,
                        target=target,
                        boss10=boss10,
                        boss50=boss50,
                        step=1,
                    ),
                    _teacher(
                        seed=seed,
                        record=second,
                        alternative=second_alt,
                        target=target,
                        boss10=boss10,
                        boss50=boss50,
                        step=2,
                    ),
                ]
                return {
                    "seed": seed,
                    "status": (
                        "NEW_FULL_WIN_TWO_STEP"
                        if _is_win(boss50)
                        else "NEW_BOSS_RESCUE_TWO_STEP"
                    ),
                    "base": base,
                    "baseline_50k": baseline50,
                    "target_boss_floor": target,
                    "attempted_one_step": attempted_one,
                    "attempted_two_step": attempted_two,
                    "rescue": {
                        "first": {
                            "branch_index": first_idx,
                            "floor": int(record["floor"]),
                            "kind": str(record["kind"]),
                            "current_index": int(record["selected_index"]),
                            "alternative_index": int(alternative),
                        },
                        "second": {
                            "branch_index": int(second["branch_index"]),
                            "floor": int(second["floor"]),
                            "kind": str(second["kind"]),
                            "current_index": int(second["selected_index"]),
                            "alternative_index": int(second_alt),
                        },
                        "boss_10k": boss10,
                        "boss_50k": boss50,
                    },
                    "negative_examples": list(negative_examples),
                    "teachers": teachers,
                    "failures": failures,
                }

    if budget_expired:
        return incomplete_result(
            base, baseline50, target, attempted_one, attempted_two, len(records)
        )

    return {
        "seed": seed,
        "status": "NO_NEW_WIN_RESCUE",
        "base": base,
        "baseline_50k": baseline50,
        "target_boss_floor": target,
        "candidate_states": len(records),
        "attempted_one_step": attempted_one,
        "attempted_two_step": attempted_two,
        "negative_examples": list(negative_examples),
            "teachers": [],
        "failures": failures,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--module-dir", type=Path, required=True)
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--weight", type=Path, required=True)
    p.add_argument("--heldout-seeds-file", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--teacher-output", type=Path, required=True)
    p.add_argument("--negative-output", type=Path, required=True)
    p.add_argument("--max-states", type=int, default=20)
    p.add_argument("--max-alternatives", type=int, default=2)
    p.add_argument("--max-two-step-states", type=int, default=8)
    p.add_argument("--max-second-states", type=int, default=4)
    p.add_argument("--max-second-alternatives", type=int, default=2)
    p.add_argument("--timeout-seconds", type=int, default=180)
    p.add_argument("--seed-timeout-seconds", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--retries", type=int, default=1)
    a = p.parse_args()

    if not 1 <= a.max_states <= 20:
        raise RuntimeError("max-states outside 1..20")
    if not 1 <= a.max_alternatives <= 4:
        raise RuntimeError("max-alternatives outside 1..4")
    if not 1 <= a.max_two_step_states <= 12:
        raise RuntimeError("max-two-step-states outside 1..12")
    if not 1 <= a.max_second_states <= 8:
        raise RuntimeError("max-second-states outside 1..8")
    if not 1 <= a.max_second_alternatives <= 3:
        raise RuntimeError("max-second-alternatives outside 1..3")
    if not 30 <= a.timeout_seconds <= 600:
        raise RuntimeError("timeout-seconds outside 30..600")
    if not 0 <= a.retries <= 2:
        raise RuntimeError("retries outside 0..2")
    if a.seed_timeout_seconds != 0 and not 60 <= a.seed_timeout_seconds <= 14400:
        raise RuntimeError("seed-timeout-seconds outside 60..14400")

    heldout = _read_seeds(a.heldout_seeds_file)
    if a.seed not in heldout:
        raise RuntimeError("mined seed must belong to the frozen Dev30 heldout set")

    row = mine_seed(
        seed=a.seed,
        module_dir=a.module_dir,
        armg_root=a.armg_root,
        weight=a.weight,
        heldout=heldout,
        max_states=a.max_states,
        max_alternatives=a.max_alternatives,
        max_two_step_states=a.max_two_step_states,
        max_second_states=a.max_second_states,
        max_second_alternatives=a.max_second_alternatives,
        timeout_seconds=a.timeout_seconds,
        retries=a.retries,
        seed_timeout_seconds=a.seed_timeout_seconds,
        checkpoint=a.checkpoint,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    teachers = list(row.get("teachers") or [])
    a.teacher_output.parent.mkdir(parents=True, exist_ok=True)
    a.teacher_output.write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in teachers),
        encoding="utf-8",
    )
    negatives = list(row.get("negative_examples") or [])
    a.negative_output.parent.mkdir(parents=True, exist_ok=True)
    a.negative_output.write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in negatives),
        encoding="utf-8",
    )
    print(
        "V35_NEW_WIN_MINER_RESULT",
        json.dumps(
            {
                "seed": a.seed,
                "status": row["status"],
                "base_floor": _floor(row.get("base") or {}),
                "baseline50_floor": _floor(row.get("baseline_50k") or {}),
                "target_boss_floor": row.get("target_boss_floor"),
                "attempted_one_step": row.get("attempted_one_step", 0),
                "attempted_two_step": row.get("attempted_two_step", 0),
                "teacher_examples": len(teachers),
                "native_failures": len(row.get("failures") or []),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
