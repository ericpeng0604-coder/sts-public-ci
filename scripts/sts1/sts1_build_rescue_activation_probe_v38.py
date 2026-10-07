#!/usr/bin/env python3
"""Audit v3.5 residual-adapter activation on a frozen, dev-only seed set."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import shutil
from typing import Any

SAFETY = ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count")
KINDS = ("card", "map", "shop", "rest", "event")


def _read_seeds(path: Path) -> list[int]:
    seeds = [int(x.strip()) for x in path.read_text(encoding="utf-8").splitlines()
             if x.strip() and not x.lstrip().startswith("#")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise RuntimeError(f"seed file must be non-empty and unique: {path}")
    return seeds


def _safe(row: dict[str, Any]) -> bool:
    return row.get("result") == "PASS_SIMULATOR_COMPLETE_RUN" and all(
        int(row.get(key, 0) or 0) == 0 for key in SAFETY
    )


def _is_win(row: dict[str, Any]) -> bool:
    return str(row.get("outcome", "")).lower() == "victory"


def paired_win_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parent_wins = sum(_is_win(row["parent"]) for row in rows)
    candidate_wins = sum(_is_win(row["candidate"]) for row in rows)
    candidate_only = sum(not _is_win(row["parent"]) and _is_win(row["candidate"]) for row in rows)
    parent_only = sum(_is_win(row["parent"]) and not _is_win(row["candidate"]) for row in rows)
    discordant = candidate_only + parent_only
    pvalue = (sum(math.comb(discordant, k) for k in range(candidate_only, discordant + 1))
              / (2.0 ** discordant)) if discordant else 1.0
    return {
        "parent_wins": parent_wins,
        "candidate_wins": candidate_wins,
        "candidate_only_wins": candidate_only,
        "parent_only_wins": parent_only,
        "net_new_wins": candidate_only - parent_only,
        "paired_sign_pvalue_one_sided": pvalue,
    }


def _pair_task(task: tuple[int, str, str, str, str, int, tuple[int, ...]]) -> dict[str, Any]:
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, _load_sts, run_simulator_game

    seed, module_dir, armg_root, parent_weight, candidate_weight, mcts_sims, heldout = task
    sts = _load_sts(Path(module_dir))
    parent = ArmGNoncombatPolicy(root=Path(armg_root), weight_path=Path(parent_weight))
    candidate = ArmGNoncombatPolicy(
        root=Path(armg_root), weight_path=Path(candidate_weight), activation_probe=True
    )
    kwargs = dict(student=None, sts=sts, seed=int(seed), evidence_path=None,
                  combat_mcts_sims=int(mcts_sims), heldout_seeds=list(heldout))
    parent_result = run_simulator_game(armg_policy=parent, **kwargs)
    candidate_result = run_simulator_game(armg_policy=candidate, **kwargs)
    if not _safe(parent_result) or not _safe(candidate_result):
        raise RuntimeError(
            f"unsafe/incomplete paired seed {seed}: parent={parent_result} candidate={candidate_result}"
        )
    return {
        "seed": int(seed),
        "parent": parent_result,
        "candidate": candidate_result,
        "total_noncombat_decisions": candidate.activation_total_noncombat_decisions,
        "multi_choice_decisions": candidate.activation_multi_choice_decisions,
        "decision_kinds": candidate.activation_decision_kinds,
        "activation_records": candidate.activation_records,
    }


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p75": None, "p90": None, "p95": None, "max": None}
    ordered = sorted(float(x) for x in values)

    def q(p: float) -> float:
        at = (len(ordered) - 1) * p
        lo = math.floor(at)
        hi = math.ceil(at)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)

    return {"p50": q(.50), "p75": q(.75), "p90": q(.90), "p95": q(.95), "max": ordered[-1]}


def _kind_stats(records: list[dict[str, Any]], total_decisions: int,
                multi_choice_decisions: int) -> dict[str, Any]:
    checks = [x for x in records if x["gate_checked"]]
    allowed = [x for x in checks if x["gate_allowed"]]
    changed = [x for x in records if x["top1_changed"]]
    margins = [float(x["margin_before"]) for x in records]
    residuals = [abs(float(v)) for x in records for v in x["adapter_residual_scores"]]
    d_pos = [float(x["d_positive"]) for x in checks if x["d_positive"] is not None]
    d_neg = [float(x["d_negative"]) for x in checks if x["d_negative"] is not None]
    margins_after = [float(x["margin_after"]) for x in records]
    return {
        "total_noncombat_decisions": total_decisions,
        "multi_choice_decisions": multi_choice_decisions,
        "gate_checks": len(checks),
        "gate_allowed": len(allowed),
        "gate_blocked": len(checks) - len(allowed),
        "adapter_applied": sum(bool(x["adapter_applied"]) for x in records),
        "top1_changed": len(changed),
        "gate_allow_rate": len(allowed) / len(checks) if checks else 0.0,
        "top1_change_rate_given_allow": len(changed) / len(allowed) if allowed else 0.0,
        "top1_change_rate_total": len(changed) / len(records) if records else 0.0,
        "d_positive": _percentiles(d_pos),
        "d_negative": _percentiles(d_neg),
        "margin_before": _percentiles(margins),
        "margin_after": _percentiles(margins_after),
        "residual_abs": _percentiles(residuals),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [item for row in rows for item in row["activation_records"]]
    total = sum(int(row["total_noncombat_decisions"]) for row in rows)
    multi = sum(int(row["multi_choice_decisions"]) for row in rows)
    per_kind = {}
    for kind in KINDS:
        kind_rows = [r for r in rows]
        kind_records = [x for x in records if x["decision_kind"] == kind]
        kind_total = sum(int(r["decision_kinds"].get(kind, 0)) for r in kind_rows)
        kind_multi = sum(1 for x in kind_records)
        per_kind[kind] = _kind_stats(kind_records, kind_total, kind_multi)

    overall = _kind_stats(records, total, multi)
    if overall["gate_checks"] == 0 or overall["gate_allow_rate"] < 0.05:
        case = "A_GATE_RARELY_ALLOWS"
    elif overall["top1_changed"] == 0:
        case = "B_ALLOWED_BUT_TOP1_UNCHANGED"
    else:
        case = "C_TOP1_CHANGES"
    paired = paired_win_summary(rows)
    return {
        "schema_version": "sts1-v38-activation-probe",
        "probe_seeds": [int(row["seed"]) for row in rows],
        "total_noncombat_decisions": total,
        "multi_choice_decisions": multi,
        "gate_checks": overall["gate_checks"],
        "gate_allowed": overall["gate_allowed"],
        "gate_blocked": overall["gate_blocked"],
        "adapter_applied": overall["adapter_applied"],
        "top1_changed": overall["top1_changed"],
        "gate_allow_rate": overall["gate_allow_rate"],
        "top1_change_rate_given_allow": overall["top1_change_rate_given_allow"],
        "top1_change_rate_total": overall["top1_change_rate_total"],
        "distance_and_margin_diagnostics": {
            key: overall[key] for key in (
                "d_positive", "d_negative", "margin_before", "margin_after", "residual_abs"
            )
        },
        "by_kind": per_kind,
        "diagnostic_case": case,
        "games": len(rows),
        **paired,
        "all_games_complete_and_safe": all(_safe(r["parent"]) and _safe(r["candidate"]) for r in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--armg-root", type=Path, required=True)
    parser.add_argument("--parent-weight", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mcts-sims", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.mcts_sims < 1:
        raise RuntimeError("workers must be 1..8 and mcts-sims must be positive")
    seeds = _read_seeds(args.seed_file)
    tasks = [(seed, str(args.module_dir), str(args.armg_root), str(args.parent_weight),
              str(args.candidate_weight), args.mcts_sims, tuple(seeds)) for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        by_seed = {row["seed"]: row for row in pool.map(_pair_task, tasks)}
    rows = [by_seed[seed] for seed in seeds]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.seed_file, args.output_dir / "probe-seeds.txt")
    decisions_path = args.output_dir / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            for decision in row["activation_records"]:
                stream.write(json.dumps({"seed": row["seed"], **decision}, sort_keys=True) + "\n")
    summary = summarize(rows)
    for row in rows:
        row.pop("activation_records", None)
    (args.output_dir / "paired-games.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("V38_ACTIVATION_PROBE", json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
