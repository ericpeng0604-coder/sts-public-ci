#!/usr/bin/env python3
"""Trace G7-win -> candidate-loss regressions and emit preservation anchors.

This is intentionally narrow: it compares deterministic non-combat ArmG decision
traces for known paired-regression seeds. Combat remains the same MCTS policy.
The first strategic divergence is recorded and converted into a parent-action
preservation example that can be rehearsed by the next Rescue candidate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.simulator import _load_sts, run_simulator_game
from sts1_ppo_v20_build_trajectory_diagnostic import ReplayFromSeedArmG


SCHEMA = "sts1-build-preservation-v1"
SAFETY = ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count")


def _read_seeds(path: Path) -> list[int]:
    seeds = [
        int(x.strip())
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]
    if not seeds or len(seeds) != len(set(seeds)):
        raise RuntimeError("seed file must be non-empty and unique")
    return seeds


def _safe(row: dict[str, Any]) -> bool:
    return (
        row.get("result") == "PASS_SIMULATOR_COMPLETE_RUN"
        and all(int(row.get(k, 0) or 0) == 0 for k in SAFETY)
    )


def _run(
    *,
    seed: int,
    module_dir: Path,
    armg_root: Path,
    weight: Path,
    heldout: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sts = _load_sts(module_dir)
    policy = ReplayFromSeedArmG(
        root=armg_root,
        weight_path=weight,
        seed=seed,
        forced=None,
    )
    result = run_simulator_game(
        student=None,
        sts=sts,
        seed=seed,
        evidence_path=None,
        armg_policy=policy,
        combat_mcts_sims=2000,
        heldout_seeds=heldout,
    )
    if not _safe(result):
        raise RuntimeError(f"unsafe/incomplete seed {seed}: {result}")
    return dict(result), list(policy.records)


def _softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    es = [math.exp(float(x) - m) for x in xs]
    z = sum(es)
    return [v / z for v in es]


def _gap(scores: list[float]) -> tuple[float, float, list[float]]:
    probs = _softmax(scores)
    if not probs:
        return 0.0, 0.0, probs
    order = sorted(probs, reverse=True)
    top1 = order[0]
    top2 = order[1] if len(order) > 1 else 0.0
    return top1 - top2, top1, probs


def _first_divergence(parent: list[dict[str, Any]], cand: list[dict[str, Any]]) -> dict[str, Any] | None:
    n = min(len(parent), len(cand))
    for i in range(n):
        p = parent[i]
        c = cand[i]
        structural = (
            int(p["branch_index"]) != int(c["branch_index"])
            or str(p["kind"]) != str(c["kind"])
            or p["descs"] != c["descs"]
        )
        selected_diff = int(p["selected_index"]) != int(c["selected_index"])
        if structural or selected_diff:
            gap, top1_prob, probs = _gap([float(x) for x in p.get("scores", [])])
            return {
                "trace_index": i,
                "branch_index_parent": int(p["branch_index"]),
                "branch_index_candidate": int(c["branch_index"]),
                "floor_parent": int(p.get("floor", 0) or 0),
                "floor_candidate": int(c.get("floor", 0) or 0),
                "act_parent": int(p.get("act", 0) or 0),
                "act_candidate": int(c.get("act", 0) or 0),
                "kind_parent": str(p["kind"]),
                "kind_candidate": str(c["kind"]),
                "structural_drift": bool(structural),
                "candidate_identity_match": p["descs"] == c["descs"],
                "candidate_count_parent": len(p["descs"]),
                "candidate_count_candidate": len(c["descs"]),
                "parent_selected_index": int(p["selected_index"]),
                "candidate_selected_index": int(c["selected_index"]),
                "parent_scores": [float(x) for x in p.get("scores", [])],
                "candidate_scores": [float(x) for x in c.get("scores", [])],
                "parent_probs": probs,
                "parent_top1_prob": top1_prob,
                "parent_prob_gap": gap,
                "parent_hp": int(p.get("hp", 0) or 0),
                "candidate_hp": int(c.get("hp", 0) or 0),
                "obs": p["obs"],
                "descs": p["descs"],
            }
    if len(parent) != len(cand):
        return {
            "trace_index": n,
            "structural_drift": True,
            "candidate_identity_match": False,
            "reason": "trace_length_diverged",
            "parent_trace_len": len(parent),
            "candidate_trace_len": len(cand),
        }
    return None


def _anchor(seed: int, div: dict[str, Any]) -> dict[str, Any] | None:
    if not div or not div.get("candidate_identity_match"):
        return None
    descs = div["descs"]
    selected = int(div["parent_selected_index"])
    target = [0.0] * len(descs)
    target[selected] = 1.0
    return {
        "schema_version": SCHEMA,
        "source": "sts1-build-rescue-regression-trace-v1",
        "seed": int(seed),
        "floor": int(div["floor_parent"]),
        "act": int(div["act_parent"]),
        "kind": str(div["kind_parent"]),
        "obs": div["obs"],
        "descs": descs,
        "parent_selected_index": selected,
        "candidate_selected_index": int(div["candidate_selected_index"]),
        "target_probs": target,
        "parent_prob_gap": float(div["parent_prob_gap"]),
        "parent_top1_prob": float(div["parent_top1_prob"]),
        "priority": 6.0,
        "reason": "parent_only_win_first_divergence",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--module-dir", type=Path, required=True)
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--parent-weight", type=Path, required=True)
    p.add_argument("--candidate-weight", type=Path, required=True)
    p.add_argument("--seed-file", type=Path, required=True)
    p.add_argument("--heldout-seeds-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preservation-output", type=Path, required=True)
    a = p.parse_args()

    seeds = _read_seeds(a.seed_file)
    heldout = _read_seeds(a.heldout_seeds_file)
    rows = []
    anchors = []

    for seed in seeds:
        pr, pt = _run(
            seed=seed,
            module_dir=a.module_dir,
            armg_root=a.armg_root,
            weight=a.parent_weight,
            heldout=heldout,
        )
        cr, ct = _run(
            seed=seed,
            module_dir=a.module_dir,
            armg_root=a.armg_root,
            weight=a.candidate_weight,
            heldout=heldout,
        )
        if str(pr.get("outcome", "")).lower() != "victory":
            raise RuntimeError(f"expected parent-only seed {seed} parent to win")
        if str(cr.get("outcome", "")).lower() == "victory":
            raise RuntimeError(f"expected parent-only seed {seed} candidate to lose")

        div = _first_divergence(pt, ct)
        row = {
            "seed": seed,
            "parent_outcome": pr["outcome"],
            "parent_final_floor": pr.get("final_floor"),
            "parent_final_hp": pr.get("final_hp"),
            "candidate_outcome": cr["outcome"],
            "candidate_final_floor": cr.get("final_floor"),
            "candidate_final_hp": cr.get("final_hp"),
            "parent_trace_len": len(pt),
            "candidate_trace_len": len(ct),
            "first_divergence": div,
        }
        anchor = _anchor(seed, div) if div else None
        if anchor is not None:
            anchors.append(anchor)
            row["preservation_anchor_created"] = True
        else:
            row["preservation_anchor_created"] = False
        rows.append(row)
        print(
            "V23_REGRESSION_TRACE",
            json.dumps(
                {
                    "seed": seed,
                    "parent_floor": pr.get("final_floor"),
                    "candidate_floor": cr.get("final_floor"),
                    "divergence": None if div is None else {
                        k: div.get(k)
                        for k in (
                            "trace_index",
                            "floor_parent",
                            "kind_parent",
                            "parent_selected_index",
                            "candidate_selected_index",
                            "parent_prob_gap",
                            "structural_drift",
                        )
                    },
                    "anchor": anchor is not None,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    payload = {
        "schema_version": "sts1-build-rescue-regression-trace-v1",
        "seeds": seeds,
        "parent_only_regressions": len(rows),
        "preservation_anchors": len(anchors),
        "rows": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    a.preservation_output.parent.mkdir(parents=True, exist_ok=True)
    a.preservation_output.write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in anchors),
        encoding="utf-8",
    )
    if len(anchors) != len(seeds):
        raise RuntimeError(
            f"expected one preservation anchor per regression seed, got {len(anchors)}/{len(seeds)}"
        )
    print("V23_REGRESSION_TRACE_PASS", json.dumps({
        "seeds": len(seeds),
        "preservation_anchors": len(anchors),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
