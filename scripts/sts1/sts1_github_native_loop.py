#!/usr/bin/env python3
"""Prepare GitHub-native STS1 loop reports and next-control payloads.

This script is intentionally network-free. GitHub Actions posts the generated
Markdown and dispatches the next workflow using the job's GITHUB_TOKEN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_control(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def render_control(values: dict[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in values.items())


def latest_reports(root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in root.glob("**/round-report.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("round"), int):
            reports.append(payload)
    reports.sort(key=lambda row: int(row["round"]))
    return reports


def safety_text(value: Any) -> str:
    if not isinstance(value, dict):
        return "未提供"
    keys = ("illegal_action_count", "crash_count", "timeout_count", "remote_error_count")
    vals = [int(value.get(key, 0) or 0) for key in keys]
    return "/".join(str(v) for v in vals) + " (illegal/crash/timeout/remote)"


def student_comment(report: dict[str, Any], run_url: str) -> str:
    distill = report.get("distillation") or {}
    gate = report.get("distill_gate") or {}
    state = report.get("loop_state") or {}
    ppo = report.get("ppo") or {}
    champion = report.get("champion") or {}
    candidate = gate.get("candidate") or {}
    accepted = report.get("accepted_stages") or []
    result = "ACCEPT" if report.get("learner_changed") else "ROLLBACK"
    return "\n".join([
        "@ericpeng0604-coder",
        "",
        f"## Student + MCTS — Round {report.get('round')}",
        "",
        f"- 結果：**{result}**" + (f"（{', '.join(accepted)}）" if accepted else ""),
        f"- Win delta：{gate.get('win_delta', 'n/a')}",
        f"- Floor delta：{gate.get('floor_delta', 'n/a')}",
        f"- Teacher top-1：{distill.get('teacher_top1_accuracy', 'n/a')}",
        f"- Teacher games：{report.get('teacher_seed_count', 'n/a')} seeds / {report.get('teacher_victories', 'n/a')} wins",
        f"- PPO：{ppo.get('status', 'UNKNOWN')} — {ppo.get('reason', '')}",
        f"- Safety：{safety_text(candidate.get('safety'))}",
        f"- Stagnation：{state.get('stagnation_count', 'n/a')}",
        f"- Offline Champion：{champion.get('status', 'n/a')}",
        f"- Learner SHA：`{report.get('learner_sha_after', 'n/a')}`",
        f"- Run：{run_url}",
    ])


def armg_gate_line(name: str, gate: Any) -> str:
    if not isinstance(gate, dict):
        return f"- {name}：未提供"
    status = gate.get("status", "UNKNOWN")
    if status == "SKIPPED":
        return f"- {name}：**SKIPPED** — {', '.join(gate.get('reasons') or [])}"
    candidate = gate.get("candidate") or {}
    current = gate.get("current") or {}
    return (
        f"- {name}：**{status}** | wins {current.get('victories', 'n/a')} → "
        f"{candidate.get('victories', 'n/a')} (Δ {gate.get('win_delta', 'n/a')}) | "
        f"mean floor {current.get('mean_final_floor', 'n/a')} → "
        f"{candidate.get('mean_final_floor', 'n/a')} (Δ {gate.get('floor_delta', 'n/a')})"
    )


def armg_comment(report: dict[str, Any], run_url: str) -> str:
    promotion = report.get("promotion") or {}
    fast = promotion.get("fast_gate") or {}
    formal = promotion.get("formal_gate") or {}
    dataset = report.get("dataset") or {}
    candidate_safety = ((formal.get("candidate") or {}).get("safety")
                        if formal.get("status") != "SKIPPED"
                        else (fast.get("candidate") or {}).get("safety"))
    return "\n".join([
        "@ericpeng0604-coder",
        "",
        f"## ArmG Map — Round {report.get('round')}",
        "",
        f"- Map Gen：**{report.get('map_generation_before')} → {report.get('map_generation_after')}**",
        f"- 結果：**{promotion.get('decision', 'UNKNOWN')}**",
        f"- 新資料：{dataset.get('example_count', 'n/a')} examples",
        f"- Replay：{report.get('replay_example_count', 'n/a')} examples",
        f"- Teacher agreement：{dataset.get('teacher_agreement', 'n/a')}",
        armg_gate_line("30 Seeds", fast),
        armg_gate_line("50 Seeds", formal),
        f"- Safety：{safety_text(candidate_safety)}",
        f"- Stagnation：{report.get('stagnation_count', 'n/a')}",
        f"- Current Map SHA：`{report.get('current_map_sha256', 'n/a')}`",
        f"- Run：{run_url}",
    ])


def write_payload(
    *,
    kind: str,
    report_root: Path,
    control_path: Path,
    out_dir: Path,
    run_url: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = latest_reports(report_root)
    if not reports:
        raise SystemExit("no round-report.json found")

    if kind == "student":
        comments = [student_comment(report, run_url) for report in reports]
        summary_path = report_root / "loop-summary.json"
        if not summary_path.is_file():
            raise SystemExit("Student loop-summary.json missing")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        paused = bool(summary.get("paused_for_stagnation"))
        continue_loop = not paused
        reason = "paused_for_stagnation" if paused else "continue_after_success"
        current = parse_control(control_path)
        current["attempt"] = str(int(current.get("attempt", "0")) + 1)
        current["rounds"] = "1"
        current["reason"] = "github_native_continue_student"
    else:
        latest = reports[-1]
        comments = [armg_comment(latest, run_url)]
        paused = bool(latest.get("paused_for_stagnation"))
        continue_loop = not paused
        reason = "paused_for_stagnation" if paused else "continue_after_success"
        current = parse_control(control_path)
        current["attempt"] = str(int(current.get("attempt", "0")) + 1)
        current["mode"] = "full"
        current["reason"] = "github_native_continue_armg"

    (out_dir / "comments.json").write_text(
        json.dumps(comments, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "decision.json").write_text(
        json.dumps(
            {"continue": continue_loop, "reason": reason},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (out_dir / "next-control.request").write_text(
        render_control(current),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("student", "armg"), required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    args = parser.parse_args()
    write_payload(
        kind=args.kind,
        report_root=args.report_root,
        control_path=args.control,
        out_dir=args.out_dir,
        run_url=args.run_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
