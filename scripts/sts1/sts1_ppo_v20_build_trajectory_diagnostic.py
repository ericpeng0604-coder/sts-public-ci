#!/usr/bin/env python3
"""Trace Boss-50k-unrescued runs back to earlier non-combat build decisions."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR=Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0,str(_SCRIPT_DIR))

import sts1_armg_ppo_rollout_v14 as rollout
from roguelike_ai.sts1_phase3.simulator import _load_sts, run_simulator_game

ACT_BY_BOSS={16:1,33:2,50:3}


def _read_seeds(path: Path) -> list[int]:
    rows=[
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows)!=50 or len(set(rows))!=50:
        raise RuntimeError("seed file must contain exactly 50 unique seeds")
    return rows


def _result_passes(row: dict[str,Any]) -> bool:
    return bool(row.get("passed_boss"))


def _second_teacher_row(
    *,
    seed:int,
    intervention:dict[str,Any],
    target_floor:int,
) -> dict[str,Any]:
    return {
        "schema_version":"sts1-armg-strategy-branch-dataset-v1",
        "type":"v20_early_two_step_boss_rescue_followup",
        "source":"sts1-ppo-v20-build-trajectory-diagnostic-v2",
        "seed":int(seed),
        "floor":int(intervention["floor"]),
        "act":int(intervention["act"]),
        "kind":str(intervention["kind"]),
        "obs":intervention["obs"],
        "descs":intervention["descs"],
        "current_armg_index":int(intervention["current_armg_index"]),
        "teacher_best_index":int(intervention["teacher_best_index"]),
        "target_probs":intervention["target_probs"],
        "priority":4.25,
        "teacher_margin":9.0,
        "confidence_weight":1.0,
        "teacher_consensus_fraction":1.0,
        "combat_policy":"mcts_2000",
        "confirmation_policy":"boss_mcts_10000_and_50000",
        "target_boss_floor":int(target_floor),
    }


def _teacher_row(
    *,
    seed:int,
    record:dict[str,Any],
    alternative:int,
    target_floor:int,
    boss10:dict[str,Any],
    boss50:dict[str,Any],
) -> dict[str,Any]:
    target=[0.0]*len(record["descs"])
    target[int(alternative)]=1.0
    return {
        "schema_version":"sts1-armg-strategy-branch-dataset-v1",
        "type":"v20_early_build_boss_rescue",
        "source":"sts1-ppo-v20-build-trajectory-diagnostic-v1",
        "seed":int(seed),
        "floor":int(record["floor"]),
        "act":int(record["act"]),
        "kind":str(record["kind"]),
        "obs":record["obs"],
        "descs":record["descs"],
        "current_armg_index":int(record["selected_index"]),
        "teacher_best_index":int(alternative),
        "target_probs":target,
        "priority":4.5,
        "teacher_margin":10.0,
        "confidence_weight":1.0,
        "teacher_consensus_fraction":1.0,
        "combat_policy":"mcts_2000",
        "confirmation_policy":"boss_mcts_10000_and_50000",
        "target_boss_floor":int(target_floor),
        "boss_10k":boss10,
        "boss_50k":boss50,
    }



class ReplayFromSeedArmG(rollout.SamplingArmG):
    """Replay ArmG from the original seed and force selected strategic decisions.

    This deliberately disables GameContext cloning. Every counterfactual starts
    from a fresh GameContext, so native MCTS never runs on a copied deep state.
    """

    def __init__(
        self,
        *,
        root: Path,
        weight_path: Path,
        seed: int,
        forced: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            root=root,
            weight_path=weight_path,
            temperature=1.0,
            torch_seed=seed,
        )
        self.forced = dict(forced or {})
        self.branch_index = 0
        self.records: list[dict[str, Any]] = []

    def capture_conversion_state(self, **kwargs: Any) -> None:
        # Production Build rescue uses replay-from-seed, never native deep clones.
        return None

    def decide(self, gc: Any, sts: Any):
        kind, selected, descs, execs, scores = super().decide(gc, sts)
        if selected < 0 or len(descs) < 2:
            return kind, selected, descs, execs, scores

        branch_index = self.branch_index
        self.branch_index += 1
        snapshot = self.training_vector_snapshot(gc, descs)
        record = {
            "branch_index": int(branch_index),
            "kind": str(kind),
            "selected_index": int(selected),
            "scores": [float(v) for v in scores],
            "floor": int(getattr(gc, "floor_num", 0) or 0),
            "act": int(getattr(gc, "act", 0) or 0),
            "hp": int(getattr(gc, "cur_hp", 0) or 0),
            "max_hp": int(getattr(gc, "max_hp", 1) or 1),
            "obs": list(snapshot["obs_412"]),
            "descs": [list(row) for row in snapshot["candidate_desc_368"]],
        }

        force = self.forced.get(branch_index)
        if force is not None:
            if str(force["kind"]) != str(kind):
                raise RuntimeError(
                    f"forced decision kind drift at branch {branch_index}: "
                    f"{force['kind']} != {kind}"
                )
            if force["descs"] != record["descs"]:
                raise RuntimeError(
                    f"forced candidate identity drift at branch {branch_index}"
                )
            forced_index = int(force["index"])
            if not 0 <= forced_index < len(descs):
                raise RuntimeError(
                    f"forced index {forced_index} outside {len(descs)} candidates"
                )
            record["forced_index"] = forced_index
            selected = forced_index

        self.records.append(record)
        return kind, int(selected), descs, execs, scores


def _passes_target_boss(result: dict[str, Any], target_floor: int) -> bool:
    if str(result.get("outcome", "")).lower() == "victory":
        return True
    floor = int(result.get("final_floor") or result.get("max_floor") or 0)
    return floor > int(target_floor)


def _force_spec(record: dict[str, Any], alternative: int) -> dict[str, Any]:
    return {
        "kind": str(record["kind"]),
        "descs": record["descs"],
        "index": int(alternative),
    }


def _run_seed_variant(
    *,
    seed: int,
    module_dir: Path,
    armg_root: Path,
    weight: Path,
    heldout: list[int],
    boss_sims: int | None,
    forced: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], ReplayFromSeedArmG]:
    sts = _load_sts(module_dir)
    policy = ReplayFromSeedArmG(
        root=armg_root,
        weight_path=weight,
        seed=seed,
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
        combat_mcts_boss_floors=(16, 33, 50),
        heldout_seeds=heldout,
    )
    if result.get("result") != "PASS_SIMULATOR_COMPLETE_RUN":
        raise RuntimeError(f"seed {seed} replay incomplete: {result}")
    for key in ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count"):
        if int(result.get(key, 0) or 0) != 0:
            raise RuntimeError(f"seed {seed} replay safety failure {key}: {result}")
    return dict(result), policy


def _isolated_run_seed_variant(
    *,
    seed:int,
    module_dir:Path,
    armg_root:Path,
    weight:Path,
    heldout:list[int],
    boss_sims:int | None,
    forced:dict[int,dict[str,Any]] | None,
    isolate:bool,
    timeout_seconds:int,
    retries:int,
) -> tuple[dict[str,Any] | None, list[dict[str,Any]], list[dict[str,Any]]]:
    """Run one replay-from-seed variant in a child so native crashes stay local."""
    failures:list[dict[str,Any]]=[]
    if not isolate:
        try:
            result,policy=_run_seed_variant(
                seed=seed,
                module_dir=module_dir,
                armg_root=armg_root,
                weight=weight,
                heldout=heldout,
                boss_sims=boss_sims,
                forced=forced,
            )
            return result,list(policy.records),failures
        except BaseException as exc:
            failures.append({
                "attempt":0,
                "kind":"python_exception",
                "error_type":type(exc).__name__,
                "error":str(exc),
            })
            return None,[],failures

    if not hasattr(os,"fork"):
        raise RuntimeError("isolated replay-from-seed requires os.fork")

    for attempt in range(int(retries)+1):
        fd,path=tempfile.mkstemp(prefix="sts1-v20-replay-",suffix=".json")
        os.close(fd)
        pid=os.fork()
        if pid==0:
            try:
                result,policy=_run_seed_variant(
                    seed=seed,
                    module_dir=module_dir,
                    armg_root=armg_root,
                    weight=weight,
                    heldout=heldout,
                    boss_sims=boss_sims,
                    forced=forced,
                )
                Path(path).write_text(
                    json.dumps({
                        "ok":True,
                        "result":result,
                        "records":list(policy.records),
                    },sort_keys=True)+"\n",
                    encoding="utf-8",
                )
                os._exit(0)
            except BaseException as exc:
                try:
                    Path(path).write_text(
                        json.dumps({
                            "ok":False,
                            "error_type":type(exc).__name__,
                            "error":str(exc),
                        },sort_keys=True)+"\n",
                        encoding="utf-8",
                    )
                finally:
                    os._exit(2)

        deadline=time.monotonic()+int(timeout_seconds)
        status=None
        timed_out=False
        while status is None:
            waited,raw=os.waitpid(pid,os.WNOHANG)
            if waited==pid:
                status=raw
                break
            if time.monotonic()>=deadline:
                timed_out=True
                try:
                    os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _,status=os.waitpid(pid,0)
                break
            time.sleep(0.05)

        payload=None
        try:
            p=Path(path)
            if p.exists() and p.stat().st_size:
                payload=json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append({
                "attempt":attempt,
                "kind":"result_decode_error",
                "error_type":type(exc).__name__,
                "error":str(exc),
            })
        finally:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

        if (
            not timed_out
            and status is not None
            and os.WIFEXITED(status)
            and os.WEXITSTATUS(status)==0
            and payload
            and payload.get("ok") is True
        ):
            return dict(payload["result"]),list(payload.get("records") or []),failures

        if timed_out:
            failure={"attempt":attempt,"kind":"timeout","timeout_seconds":int(timeout_seconds)}
        elif status is not None and os.WIFSIGNALED(status):
            sig=int(os.WTERMSIG(status))
            try:
                sig_name=signal.Signals(sig).name
            except Exception:
                sig_name=str(sig)
            failure={"attempt":attempt,"kind":"native_signal","signal":sig,"signal_name":sig_name}
        elif payload and payload.get("ok") is False:
            failure={
                "attempt":attempt,
                "kind":"python_exception",
                "error_type":payload.get("error_type"),
                "error":payload.get("error"),
            }
        else:
            failure={
                "attempt":attempt,
                "kind":"child_exit",
                "exit_code":(
                    int(os.WEXITSTATUS(status))
                    if status is not None and os.WIFEXITED(status)
                    else None
                ),
            }
        failures.append(failure)

    return None,[],failures


def _replay_teacher_row(
    *,
    seed: int,
    record: dict[str, Any],
    alternative: int,
    target_floor: int,
    boss10: dict[str, Any],
    boss50: dict[str, Any],
    type_name: str,
    priority: float,
) -> dict[str, Any]:
    row = _teacher_row(
        seed=seed,
        record=record,
        alternative=alternative,
        target_floor=target_floor,
        boss10=boss10,
        boss50=boss50,
    )
    row["type"] = type_name
    row["source"] = "sts1-ppo-v20-build-replay-from-seed-v1"
    row["priority"] = float(priority)
    row["replay_from_seed"] = True
    return row


def diagnose_seed_replay(
    *,
    seed: int,
    module_dir: Path,
    armg_root: Path,
    weight: Path,
    heldout: list[int],
    max_states: int,
    max_alternatives: int,
    max_two_step_states: int,
    max_second_alternatives: int,
    isolate_branches: bool=False,
    branch_timeout_seconds: int=180,
    branch_retries: int=1,
) -> dict[str, Any]:
    branch_failures:list[dict[str,Any]]=[]

    def run_variant(*, boss_sims:int | None, forced:dict[int,dict[str,Any]] | None=None):
        result,records,failures=_isolated_run_seed_variant(
            seed=seed,
            module_dir=module_dir,
            armg_root=armg_root,
            weight=weight,
            heldout=heldout,
            boss_sims=boss_sims,
            forced=forced,
            isolate=isolate_branches,
            timeout_seconds=branch_timeout_seconds,
            retries=branch_retries,
        )
        branch_failures.extend(failures)
        return result,records

    base,base_records=run_variant(boss_sims=None)
    if base is None:
        return {
            "seed":seed,
            "status":"NATIVE_BASE_REPLAY_FAILURE",
            "attempted_states":0,
            "two_step_attempts":0,
            "native_branch_failures":len(branch_failures),
            "branch_failure_details":branch_failures,
            "rescue":None,
        }
    floor = int(base.get("final_floor") or base.get("max_floor") or 0)
    if str(base.get("outcome", "")).lower() == "victory" or not rollout._near_boss_failure(floor):
        return {
            "seed": seed,
            "status": "NOT_REPRODUCED_AS_NEAR_BOSS_LOSS",
            "base": base,
            "captured_state_count": len(base_records),
            "attempted_states": 0,
            "two_step_attempts": 0,
            "native_branch_failures":len(branch_failures),
            "branch_failure_details":branch_failures,
            "rescue": None,
        }

    target = rollout._boss_target_floor(floor)
    baseline50,_=run_variant(boss_sims=50000)
    if baseline50 is None:
        return {
            "seed":seed,
            "status":"NATIVE_BASELINE50_FAILURE",
            "base":base,
            "target_boss_floor":target,
            "captured_state_count":len(base_records),
            "attempted_states":0,
            "two_step_attempts":0,
            "native_branch_failures":len(branch_failures),
            "branch_failure_details":branch_failures,
            "rescue":None,
        }
    if _passes_target_boss(baseline50, target):
        return {
            "seed": seed,
            "status": "BASELINE_50K_RESCUED_ON_REPLAY",
            "base": base,
            "baseline_50k": baseline50,
            "target_boss_floor": target,
            "captured_state_count": len(base_records),
            "attempted_states": 0,
            "two_step_attempts": 0,
            "native_branch_failures":len(branch_failures),
            "branch_failure_details":branch_failures,
            "rescue": None,
        }

    records = [
        row for row in base_records
        if int(row.get("floor", 0) or 0) <= target
        and len(row.get("descs", [])) >= 2
    ]
    records = list(reversed(records[-max_states:]))
    if not records:
        return {
            "seed": seed,
            "status": "NO_REPLAYABLE_STRATEGIC_STATE",
            "base": base,
            "target_boss_floor": target,
            "captured_state_count": len(base_records),
            "attempted_states": 0,
            "two_step_attempts": 0,
            "native_branch_failures":len(branch_failures),
            "branch_failure_details":branch_failures,
            "rescue": None,
        }

    attempted = 0
    first_step_runs: list[tuple[dict[str, Any], int, dict[str, Any], list[dict[str,Any]]]] = []

    for record in records:
        selected = int(record["selected_index"])
        alternatives = [i for i in range(len(record["descs"])) if i != selected]
        alternatives.sort(
            key=lambda i: float(record["scores"][i]) if i < len(record["scores"]) else -1e30,
            reverse=True,
        )
        for alternative in alternatives[:max_alternatives]:
            attempted += 1
            forced = {
                int(record["branch_index"]): _force_spec(record, alternative)
            }
            boss10,records10=run_variant(boss_sims=10000,forced=forced)
            if boss10 is None:
                continue
            first_step_runs.append((record,int(alternative),boss10,records10))
            if not _passes_target_boss(boss10, target):
                continue
            boss50,_=run_variant(boss_sims=50000,forced=forced)
            if boss50 is None:
                continue
            if not _passes_target_boss(boss50, target):
                continue
            teacher = _replay_teacher_row(
                seed=seed,
                record=record,
                alternative=alternative,
                target_floor=target,
                boss10=boss10,
                boss50=boss50,
                type_name="v20_replay_single_build_rescue",
                priority=4.5,
            )
            return {
                "seed": seed,
                "status": "EARLY_BUILD_RESCUE_FOUND",
                "base": base,
                "baseline_50k": baseline50,
                "target_boss_floor": target,
                "captured_state_count": len(base_records),
                "candidate_states": len(records),
                "attempted_states": attempted,
                "two_step_attempts": 0,
                "native_branch_failures":len(branch_failures),
                "branch_failure_details":branch_failures,
                "rescue": {
                    "decision_floor": int(record["floor"]),
                    "decision_act": int(record["act"]),
                    "kind": str(record["kind"]),
                    "branch_index": int(record["branch_index"]),
                    "current_index": selected,
                    "alternative_index": int(alternative),
                    "boss_10k": boss10,
                    "boss_50k": boss50,
                    "teacher": teacher,
                },
            }

    two_step_attempts = 0
    # Reuse already-computed first-step 10k replays. Only the nearest
    # max_two_step_states first interventions are eligible for a second change.
    for record, alternative, first10, first_records in first_step_runs[:max_two_step_states]:
        first_idx = int(record["branch_index"])
        subsequent = [
            row for row in first_records
            if int(row["branch_index"]) > first_idx
            and int(row.get("floor", 0) or 0) <= target
            and len(row.get("descs", [])) >= 2
        ]
        if not subsequent:
            continue
        second_record = subsequent[0]
        second_selected = int(second_record["selected_index"])
        second_alts = [
            i for i in range(len(second_record["descs"]))
            if i != second_selected
        ]
        second_alts.sort(
            key=lambda i: float(second_record["scores"][i])
            if i < len(second_record["scores"]) else -1e30,
            reverse=True,
        )
        for second_alt in second_alts[:max_second_alternatives]:
            two_step_attempts += 1
            forced = {
                first_idx: _force_spec(record, alternative),
                int(second_record["branch_index"]): _force_spec(second_record, second_alt),
            }
            boss10,_=run_variant(boss_sims=10000,forced=forced)
            if boss10 is None:
                continue
            if not _passes_target_boss(boss10, target):
                continue
            boss50,_=run_variant(boss_sims=50000,forced=forced)
            if boss50 is None:
                continue
            if not _passes_target_boss(boss50, target):
                continue

            first_teacher = _replay_teacher_row(
                seed=seed,
                record=record,
                alternative=alternative,
                target_floor=target,
                boss10=boss10,
                boss50=boss50,
                type_name="v20_replay_two_step_build_rescue",
                priority=4.5,
            )
            second_teacher = _replay_teacher_row(
                seed=seed,
                record=second_record,
                alternative=second_alt,
                target_floor=target,
                boss10=boss10,
                boss50=boss50,
                type_name="v20_replay_two_step_build_rescue_followup",
                priority=4.25,
            )
            return {
                "seed": seed,
                "status": "EARLY_TWO_STEP_BUILD_RESCUE_FOUND",
                "base": base,
                "baseline_50k": baseline50,
                "target_boss_floor": target,
                "captured_state_count": len(base_records),
                "candidate_states": len(records),
                "attempted_states": attempted,
                "two_step_attempts": two_step_attempts,
                "native_branch_failures":len(branch_failures),
                "branch_failure_details":branch_failures,
                "rescue": {
                    "decision_floor": int(record["floor"]),
                    "decision_act": int(record["act"]),
                    "kind": str(record["kind"]),
                    "branch_index": first_idx,
                    "current_index": int(record["selected_index"]),
                    "alternative_index": int(alternative),
                    "second_decision_floor": int(second_record["floor"]),
                    "second_kind": str(second_record["kind"]),
                    "second_branch_index": int(second_record["branch_index"]),
                    "second_current_index": second_selected,
                    "second_alternative_index": int(second_alt),
                    "boss_10k": boss10,
                    "boss_50k": boss50,
                    "teachers": [first_teacher, second_teacher],
                },
            }

    return {
        "seed": seed,
        "status": "NO_EARLY_BUILD_RESCUE",
        "base": base,
        "baseline_50k": baseline50,
        "target_boss_floor": target,
        "captured_state_count": len(base_records),
        "candidate_states": len(records),
        "attempted_states": attempted,
        "two_step_attempts": two_step_attempts,
        "native_branch_failures":len(branch_failures),
        "branch_failure_details":branch_failures,
        "rescue": None,
    }


def _isolated_force_choice_and_finish(
    record:dict[str,Any],
    *,
    sts:Any,
    policy:Any,
    timeout_seconds:int,
    retries:int,
    isolate:bool,
    **kwargs:Any,
) -> tuple[dict[str,Any] | None, list[dict[str,Any]]]:
    """Run one native counterfactual in a fork so SIGSEGV cannot kill the controller."""
    force_fn=getattr(rollout, "_force_choice_and_finish")
    failures:list[dict[str,Any]]=[]

    if not isolate:
        try:
            return force_fn(record, sts=sts, policy=policy, **kwargs), failures
        except BaseException as exc:
            failures.append({
                "attempt":0,
                "kind":"python_exception",
                "error_type":type(exc).__name__,
                "error":str(exc),
            })
            return None, failures

    if not hasattr(os, "fork"):
        raise RuntimeError("isolated Build tracing requires os.fork on this runner")

    for attempt in range(int(retries)+1):
        fd,path=tempfile.mkstemp(prefix="sts1-v20-branch-",suffix=".json")
        os.close(fd)
        pid=os.fork()
        if pid==0:
            try:
                result=force_fn(record, sts=sts, policy=policy, **kwargs)
                Path(path).write_text(
                    json.dumps({"ok":True,"result":result},sort_keys=True)+"\n",
                    encoding="utf-8",
                )
                os._exit(0)
            except BaseException as exc:
                try:
                    Path(path).write_text(
                        json.dumps({
                            "ok":False,
                            "error_type":type(exc).__name__,
                            "error":str(exc),
                        },sort_keys=True)+"\n",
                        encoding="utf-8",
                    )
                finally:
                    os._exit(2)

        deadline=time.monotonic()+int(timeout_seconds)
        status=None
        timed_out=False
        while status is None:
            waited,raw=os.waitpid(pid,os.WNOHANG)
            if waited==pid:
                status=raw
                break
            if time.monotonic()>=deadline:
                timed_out=True
                try:
                    os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _,status=os.waitpid(pid,0)
                break
            time.sleep(0.05)

        payload=None
        try:
            p=Path(path)
            if p.exists() and p.stat().st_size:
                payload=json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append({
                "attempt":attempt,
                "kind":"result_decode_error",
                "error_type":type(exc).__name__,
                "error":str(exc),
            })
        finally:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

        if (
            not timed_out
            and status is not None
            and os.WIFEXITED(status)
            and os.WEXITSTATUS(status)==0
            and payload
            and payload.get("ok") is True
        ):
            return dict(payload["result"]),failures

        if timed_out:
            failure={"attempt":attempt,"kind":"timeout","timeout_seconds":int(timeout_seconds)}
        elif status is not None and os.WIFSIGNALED(status):
            sig=int(os.WTERMSIG(status))
            failure={
                "attempt":attempt,
                "kind":"native_signal",
                "signal":sig,
                "signal_name":signal.Signals(sig).name if sig in signal.Signals.__members__.values() else str(sig),
            }
        elif payload and payload.get("ok") is False:
            failure={
                "attempt":attempt,
                "kind":"python_exception",
                "error_type":payload.get("error_type"),
                "error":payload.get("error"),
            }
        else:
            failure={
                "attempt":attempt,
                "kind":"child_exit",
                "exit_code":(
                    int(os.WEXITSTATUS(status))
                    if status is not None and os.WIFEXITED(status)
                    else None
                ),
            }
        failures.append(failure)

    return None,failures


def diagnose_seed(
    *,
    seed:int,
    module_dir:Path,
    armg_root:Path,
    weight:Path,
    heldout:list[int],
    max_states:int,
    max_alternatives:int,
    max_two_step_states:int,
    max_second_alternatives:int,
    isolate_branches:bool=False,
    branch_timeout_seconds:int=180,
    branch_retries:int=1,
) -> dict[str,Any]:
    sts=_load_sts(module_dir)
    policy=rollout.SamplingArmG(
        root=armg_root,
        weight_path=weight,
        temperature=1.0,
        torch_seed=seed,
    )
    branch_failures:list[dict[str,Any]]=[]

    def run_force(record:dict[str,Any], **kwargs:Any) -> dict[str,Any]:
        # Existing call sites pass sts/policy; the controller owns those.
        kwargs.pop("sts",None)
        kwargs.pop("policy",None)
        result,failures=_isolated_force_choice_and_finish(
            record,
            sts=sts,
            policy=policy,
            timeout_seconds=branch_timeout_seconds,
            retries=branch_retries,
            isolate=isolate_branches,
            **kwargs,
        )
        branch_failures.extend(failures)
        if result is None:
            return {
                "passed_boss":False,
                "victory":False,
                "isolated_failure":True,
                "failure_details":failures,
            }
        return result
    base=run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=None,
        armg_policy=policy,
        combat_mcts_sims=2000,
        heldout_seeds=heldout,
    )
    if base.get("result")!="PASS_SIMULATOR_COMPLETE_RUN":
        raise RuntimeError(f"seed {seed} base rerun incomplete: {base}")
    floor=int(base.get("final_floor") or base.get("max_floor") or 0)
    if str(base.get("outcome","")).lower()=="victory" or not rollout._near_boss_failure(floor):
        return {
            "seed":seed,
            "status":"NOT_REPRODUCED_AS_NEAR_BOSS_LOSS",
            "base":base,
            "captured_states":len(policy.conversion_states),
            "captured_state_summary":[
                {
                    "floor":int(row.get("floor",0) or 0),
                    "act":int(row.get("act",0) or 0),
                    "kind":str(row.get("kind","unknown")),
                    "choices":len(row.get("descs",[])),
                }
                for row in policy.conversion_states
            ],
            "attempted_states":0,
            "rescue":None,
        }

    target=rollout._boss_target_floor(floor)
    captured=list(policy.conversion_states)
    captured_summary=[
        {
            "floor":int(row.get("floor",0) or 0),
            "act":int(row.get("act",0) or 0),
            "kind":str(row.get("kind","unknown")),
            "choices":len(row.get("descs",[])),
        }
        for row in captured
    ]
    # Floor is the stable cross-version key. The simulator's act field can
    # transition around Boss boundaries, so filtering on exact act caused
    # valid reversible states to be discarded.
    records=[
        row
        for row in captured
        if int(row.get("floor",0) or 0) <= target
        and len(row.get("descs",[]))>=2
    ]
    records=list(reversed(records[-max_states:]))
    attempted=0
    if not records:
        return {
            "seed":seed,
            "status":"NO_CAPTURED_REVERSIBLE_STATE",
            "base":base,
            "target_boss_floor":target,
            "captured_state_count":len(policy.conversion_states),
            "attempted_states":0,
            "rescue":None,
        }

    for record in records:
        selected=int(record["selected_index"])
        baseline50=run_force(
            record,
            choice_index=selected,
            sts=sts,
            policy=policy,
            mcts_sims=2000,
            boss_mcts_sims=50000,
            target_floor=target,
        )
        if baseline50.get("isolated_failure") or _result_passes(baseline50):
            continue

        alternatives=[i for i in range(len(record["descs"])) if i!=selected]
        alternatives.sort(
            key=lambda i: float(record["scores"][i]) if i < len(record["scores"]) else -1e30,
            reverse=True,
        )
        for alternative in alternatives[:max_alternatives]:
            attempted += 1
            boss10=run_force(
                record,
                choice_index=alternative,
                sts=sts,
                policy=policy,
                mcts_sims=2000,
                boss_mcts_sims=10000,
                target_floor=target,
            )
            if not _result_passes(boss10):
                continue
            boss50=run_force(
                record,
                choice_index=alternative,
                sts=sts,
                policy=policy,
                mcts_sims=2000,
                boss_mcts_sims=50000,
                target_floor=target,
            )
            if not _result_passes(boss50):
                continue

            return {
                "seed":seed,
                "status":"EARLY_BUILD_RESCUE_FOUND",
                "base":base,
                "target_boss_floor":target,
                "captured_states":len(captured),
                "captured_state_summary":captured_summary,
                "candidate_states":len(records),
                "captured_state_count":len(policy.conversion_states),
                "attempted_states":attempted,
                "native_branch_failures":len(branch_failures),
                "branch_failure_details":branch_failures,
                "rescue":{
                    "decision_floor":int(record["floor"]),
                    "decision_act":int(record["act"]),
                    "kind":str(record["kind"]),
                    "current_index":selected,
                    "alternative_index":int(alternative),
                    "boss_10k":boss10,
                    "boss_50k":boss50,
                    "teacher":_teacher_row(
                        seed=seed,
                        record=record,
                        alternative=alternative,
                        target_floor=target,
                        boss10=boss10,
                        boss50=boss50,
                    ),
                },
            }

    two_step_attempts=0
    for record in records[:max_two_step_states]:
        selected=int(record["selected_index"])
        baseline50=run_force(
            record,
            choice_index=selected,
            sts=sts,
            policy=policy,
            mcts_sims=2000,
            boss_mcts_sims=50000,
            target_floor=target,
        )
        if baseline50.get("isolated_failure") or _result_passes(baseline50):
            continue

        alternatives=[i for i in range(len(record["descs"])) if i!=selected]
        alternatives.sort(
            key=lambda i: float(record["scores"][i]) if i < len(record["scores"]) else -1e30,
            reverse=True,
        )
        for alternative in alternatives[:max_alternatives]:
            for second_rank in range(max_second_alternatives):
                two_step_attempts += 1
                boss10=run_force(
                    record,
                    choice_index=alternative,
                    sts=sts,
                    policy=policy,
                    mcts_sims=2000,
                    boss_mcts_sims=10000,
                    target_floor=target,
                    second_alternative_rank=second_rank,
                )
                if not _result_passes(boss10):
                    continue
                boss50=run_force(
                    record,
                    choice_index=alternative,
                    sts=sts,
                    policy=policy,
                    mcts_sims=2000,
                    boss_mcts_sims=50000,
                    target_floor=target,
                    second_alternative_rank=second_rank,
                )
                if not _result_passes(boss50):
                    continue

                second=(
                    boss50.get("second_intervention")
                    or boss10.get("second_intervention")
                )
                first_teacher=_teacher_row(
                    seed=seed,
                    record=record,
                    alternative=alternative,
                    target_floor=target,
                    boss10=boss10,
                    boss50=boss50,
                )
                first_teacher["type"]="v20_early_two_step_boss_rescue"
                first_teacher["source"]="sts1-ppo-v20-build-trajectory-diagnostic-v2"
                teachers=[first_teacher]
                if second is not None:
                    teachers.append(
                        _second_teacher_row(
                            seed=seed,
                            intervention=second,
                            target_floor=target,
                        )
                    )
                return {
                    "seed":seed,
                    "status":"EARLY_TWO_STEP_BUILD_RESCUE_FOUND",
                    "base":base,
                    "target_boss_floor":target,
                    "captured_states":len(captured),
                    "captured_state_summary":captured_summary,
                    "candidate_states":len(records),
                    "captured_state_count":len(policy.conversion_states),
                    "attempted_states":attempted,
                    "two_step_attempts":two_step_attempts,
                    "native_branch_failures":len(branch_failures),
                    "branch_failure_details":branch_failures,
                    "rescue":{
                        "decision_floor":int(record["floor"]),
                        "decision_act":int(record["act"]),
                        "kind":str(record["kind"]),
                        "current_index":selected,
                        "alternative_index":int(alternative),
                        "second_alternative_rank":int(second_rank),
                        "boss_10k":boss10,
                        "boss_50k":boss50,
                        "teachers":teachers,
                    },
                }

    return {
        "seed":seed,
        "status":"NO_EARLY_BUILD_RESCUE",
        "base":base,
        "target_boss_floor":target,
        "captured_states":len(captured),
        "captured_state_summary":captured_summary,
        "candidate_states":len(records),
        "captured_state_count":len(policy.conversion_states),
        "attempted_states":attempted,
        "two_step_attempts":two_step_attempts,
        "native_branch_failures":len(branch_failures),
        "branch_failure_details":branch_failures,
        "rescue":None,
    }


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--armg-root",type=Path,required=True)
    p.add_argument("--weight",type=Path,required=True)
    p.add_argument("--seeds-file",type=Path,required=True)
    p.add_argument("--boss-summary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--teacher-output",type=Path,required=True)
    p.add_argument("--max-seeds",type=int,default=10)
    p.add_argument("--max-states",type=int,default=20)
    p.add_argument("--max-alternatives",type=int,default=2)
    p.add_argument("--max-two-step-states",type=int,default=8)
    p.add_argument("--max-second-alternatives",type=int,default=2)
    p.add_argument("--candidate-index",type=int)
    p.add_argument("--isolate-branches",type=int,choices=(0,1),default=1)
    p.add_argument("--branch-timeout-seconds",type=int,default=180)
    p.add_argument("--branch-retries",type=int,default=1)
    args=p.parse_args()
    if not 1<=args.max_seeds<=40:
        raise RuntimeError("max-seeds must be within 1..40")
    if not 1<=args.max_states<=20:
        raise RuntimeError("max-states must be within 1..20")
    if not 1<=args.max_alternatives<=4:
        raise RuntimeError("max-alternatives must be within 1..4")
    if not 1<=args.max_two_step_states<=20:
        raise RuntimeError("max-two-step-states must be within 1..20")
    if not 1<=args.max_second_alternatives<=3:
        raise RuntimeError("max-second-alternatives must be within 1..3")
    if not 10<=args.branch_timeout_seconds<=600:
        raise RuntimeError("branch-timeout-seconds must be within 10..600")
    if not 0<=args.branch_retries<=2:
        raise RuntimeError("branch-retries must be within 0..2")

    heldout=_read_seeds(args.seeds_file)
    summary=json.loads(args.boss_summary.read_text(encoding="utf-8"))
    candidates=[
        int(v)
        for v in summary["not_rescued_by_boss_50k"]["seeds"]
    ][:args.max_seeds]
    if args.candidate_index is not None:
        if not 0<=args.candidate_index<len(candidates):
            raise RuntimeError("candidate-index is outside selected Build-limited seeds")
        candidates=[candidates[args.candidate_index]]

    rows=[
        diagnose_seed_replay(
            seed=seed,
            module_dir=args.module_dir,
            armg_root=args.armg_root,
            weight=args.weight,
            heldout=heldout,
            max_states=args.max_states,
            max_alternatives=args.max_alternatives,
            max_two_step_states=args.max_two_step_states,
            max_second_alternatives=args.max_second_alternatives,
            isolate_branches=bool(args.isolate_branches),
            branch_timeout_seconds=args.branch_timeout_seconds,
            branch_retries=args.branch_retries,
        )
        for seed in candidates
    ]
    teacher=[]
    for row in rows:
        rescue=row.get("rescue") or {}
        if rescue.get("teacher"):
            teacher.append(rescue["teacher"])
        teacher.extend(rescue.get("teachers") or [])
    payload={
        "schema_version":"sts1-ppo-v20-build-trajectory-diagnostic-v3-replay-from-seed",
        "input_build_limited":len(summary["not_rescued_by_boss_50k"]["seeds"]),
        "diagnosed_seeds":len(rows),
        "early_build_rescue_seeds":sum(
            row["status"] in {"EARLY_BUILD_RESCUE_FOUND","EARLY_TWO_STEP_BUILD_RESCUE_FOUND"}
            for row in rows
        ),
        "single_step_rescues":sum(row["status"]=="EARLY_BUILD_RESCUE_FOUND" for row in rows),
        "two_step_rescues":sum(row["status"]=="EARLY_TWO_STEP_BUILD_RESCUE_FOUND" for row in rows),
        "teacher_examples":len(teacher),
        "no_early_rescue":sum(row["status"]=="NO_EARLY_BUILD_RESCUE" for row in rows),
        "capture_missing":sum(row["status"]=="NO_CAPTURED_REVERSIBLE_STATE" for row in rows),
        "not_reproduced":sum(row["status"]=="NOT_REPRODUCED_AS_NEAR_BOSS_LOSS" for row in rows),
        "captured_state_total":sum(int(row.get("captured_state_count",0) or 0) for row in rows),
        "native_branch_failures":sum(int(row.get("native_branch_failures",0) or 0) for row in rows),
        "rows":rows,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    args.teacher_output.parent.mkdir(parents=True,exist_ok=True)
    args.teacher_output.write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in teacher),
        encoding="utf-8",
    )
    print("PPO_V20_BUILD_TRAJECTORY_RESULT",json.dumps({
        "diagnosed_seeds":len(rows),
        "early_build_rescue_seeds":payload["early_build_rescue_seeds"],
        "single_step_rescues":payload["single_step_rescues"],
        "two_step_rescues":payload["two_step_rescues"],
        "teacher_examples":payload["teacher_examples"],
        "no_early_rescue":payload["no_early_rescue"],
        "capture_missing":payload["capture_missing"],
        "captured_state_total":payload["captured_state_total"],
        "native_branch_failures":payload["native_branch_failures"],
        "not_reproduced":payload["not_reproduced"],
    },sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
