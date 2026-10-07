#!/usr/bin/env python3
"""Recover the v3.8 report from an already completed probe artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _probe_module() -> Any:
    path = Path(__file__).with_name("sts1_build_rescue_activation_probe_v38.py")
    spec = importlib.util.spec_from_file_location("activation_probe_v38", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load activation-probe aggregation helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recover(input_dir: Path, output_dir: Path, source_run_id: int) -> dict[str, Any]:
    source = input_dir / "probe"
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((source / "paired-games.json").read_text(encoding="utf-8"))
    ledger = json.loads((input_dir / "seed-ledger.json").read_text(encoding="utf-8"))
    if len(rows) != 30 or len(summary.get("probe_seeds", [])) != 30:
        raise RuntimeError("recovered Probe30 artifact does not contain exactly 30 paired games")
    if not summary.get("all_games_complete_and_safe"):
        raise RuntimeError("probe artifact is incomplete or has a non-zero safety counter")
    if set(ledger["probe30"]) & (set(ledger["v36_dev30"]) | set(ledger["v37_dev50"])):
        raise RuntimeError("Probe30 overlaps a previously used dev set")

    probe = _probe_module()
    if not all(probe._safe(row["parent"]) and probe._safe(row["candidate"]) for row in rows):
        raise RuntimeError("paired-game artifact contains an incomplete run or non-zero safety counter")
    paired = probe.paired_win_summary(rows)
    recovered = {
        **summary,
        **paired,
        "source_probe_run_id": int(source_run_id),
        "readiness": "HOLD_PARENT",
        "promotion_evidence": "diagnostic-only; Probe30 is excluded from Formal gates",
        "report_recovered_from_artifact": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(recovered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = [
        "@ericpeng0604-coder",
        "",
        "## STS1 v3.8 Activation Probe",
        "",
        "Data:",
        "- Positive Teacher: 62 decisions / 56 seeds (reused v3.5 source)",
        "- Positive Seeds: 56",
        "- Negative: 325+ (reused v3.5 source)",
        "- Preservation: reused v3.5 preservation anchors; count not remeasured",
        "",
        "Training:",
        "- selected epoch: 1 (source run 37431578974)",
        "- fully learned: not retrained in this diagnostic run",
        "- KL: not recomputed; allowed KL: approximately 0.00296 per source contract",
        "- preservation / negative rejection: reused source candidate; not retrained",
        "",
        "Activation:",
        f"- gate checks: {recovered['gate_checks']}",
        f"- gate allowed: {recovered['gate_allowed']} ({recovered['gate_allow_rate']:.2%})",
        f"- adapter applied: {recovered['adapter_applied']}",
        f"- top1 changed: {recovered['top1_changed']} ({recovered['top1_change_rate_total']:.3%})",
        f"- top1 change rate given allow: {recovered['top1_change_rate_given_allow']:.3%}",
        f"- diagnostic case: {recovered['diagnostic_case']}",
        "",
        "Evaluation:",
        (f"- Dev Probe30: G7 {recovered['parent_wins']}/30; Candidate "
         f"{recovered['candidate_wins']}/30; Candidate-only {recovered['candidate_only_wins']}; "
         f"Parent-only {recovered['parent_only_wins']}; Net {recovered['net_new_wins']} "
         f"(paired p={recovered['paired_sign_pvalue_one_sided']:.4f}; diagnostic only)"),
        "- Gate30: not run",
        "- Gate50: not run",
        "- Fresh100: not run",
        "- Fresh500: not run",
        "",
        "Final:",
        "- HOLD_PARENT",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return recovered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    args = parser.parse_args()
    summary = recover(args.input_dir, args.output_dir, args.source_run_id)
    print("V38_REPORT_RECOVERY", json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
