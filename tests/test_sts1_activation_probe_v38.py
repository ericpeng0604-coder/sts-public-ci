from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sts1_activation_probe_v38",
    ROOT / "scripts/sts1/sts1_build_rescue_activation_probe_v38.py",
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
REPORT_SPEC = importlib.util.spec_from_file_location(
    "sts1_activation_report_v38",
    ROOT / "scripts/sts1/sts1_build_rescue_activation_report_v38.py",
)
assert REPORT_SPEC and REPORT_SPEC.loader
report = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(report)


def _record(kind: str, *, allowed: bool, changed: bool, d_pos: float, d_neg: float):
    return {
        "decision_kind": kind,
        "gate_checked": True,
        "gate_allowed": allowed,
        "d_positive": d_pos,
        "d_negative": d_neg,
        "adapter_applied": allowed,
        "top1_changed": changed,
        "margin_before": 0.5,
        "margin_after": 0.25 if changed else 0.5,
        "adapter_residual_scores": [0.0, 0.3 if changed else 0.02],
    }


def _row(seed: int, records: list[dict], kinds: dict[str, int]):
    return {
        "seed": seed,
        "activation_records": records,
        "total_noncombat_decisions": sum(kinds.values()),
        "multi_choice_decisions": len(records),
        "decision_kinds": kinds,
        "parent": {"outcome": "defeat"},
        "candidate": {"outcome": "victory"},
    }


def test_summary_counts_probe_activation_and_kind_rates():
    rows = [
        _row(101, [_record("card", allowed=True, changed=True, d_pos=.1, d_neg=.5)], {"card": 2}),
        _row(102, [_record("map", allowed=False, changed=False, d_pos=.8, d_neg=.2)], {"map": 1}),
    ]
    summary = probe.summarize(rows)
    assert summary["total_noncombat_decisions"] == 3
    assert summary["multi_choice_decisions"] == 2
    assert summary["gate_checks"] == 2
    assert summary["gate_allowed"] == 1
    assert summary["gate_blocked"] == 1
    assert summary["adapter_applied"] == 1
    assert summary["top1_changed"] == 1
    assert summary["gate_allow_rate"] == .5
    assert summary["top1_change_rate_given_allow"] == 1.0
    assert summary["top1_change_rate_total"] == .5
    assert summary["by_kind"]["card"]["total_noncombat_decisions"] == 2
    assert summary["by_kind"]["map"]["gate_blocked"] == 1
    assert summary["diagnostic_case"] == "C_TOP1_CHANGES"


def test_summary_selects_gate_and_margin_bottleneck_cases():
    blocked = _row(1, [_record("event", allowed=False, changed=False, d_pos=.9, d_neg=.1)], {"event": 1})
    assert probe.summarize([blocked])["diagnostic_case"] == "A_GATE_RARELY_ALLOWS"

    unchanged = _row(2, [_record("shop", allowed=True, changed=False, d_pos=.1, d_neg=.8)], {"shop": 1})
    assert probe.summarize([unchanged])["diagnostic_case"] == "B_ALLOWED_BUT_TOP1_UNCHANGED"


def test_paired_win_summary_counts_only_discordant_seed_outcomes():
    rows = [
        {"parent": {"outcome": "defeat"}, "candidate": {"outcome": "victory"}},
        {"parent": {"outcome": "victory"}, "candidate": {"outcome": "defeat"}},
        {"parent": {"outcome": "victory"}, "candidate": {"outcome": "victory"}},
        {"parent": {"outcome": "defeat"}, "candidate": {"outcome": "defeat"}},
    ]
    assert probe.paired_win_summary(rows) == {
        "parent_wins": 2,
        "candidate_wins": 2,
        "candidate_only_wins": 1,
        "parent_only_wins": 1,
        "net_new_wins": 0,
        "paired_sign_pvalue_one_sided": 0.75,
    }


def test_report_recovery_uses_saved_probe_without_simulator(tmp_path: Path):
    source = tmp_path / "input/probe"
    source.mkdir(parents=True)
    rows = []
    for seed in range(30):
        parent = "victory" if seed in {0, 2} else "defeat"
        candidate = "victory" if seed in {1, 2} else "defeat"
        safe_result = "PASS_SIMULATOR_COMPLETE_RUN"
        rows.append({
            "seed": seed,
            "parent": {"outcome": parent, "result": safe_result},
            "candidate": {"outcome": candidate, "result": safe_result},
        })
    (source / "paired-games.json").write_text(json.dumps(rows), encoding="utf-8")
    (source / "summary.json").write_text(
        json.dumps({
            "probe_seeds": list(range(30)),
            "all_games_complete_and_safe": True,
            "gate_checks": 10,
            "gate_allowed": 1,
            "gate_allow_rate": .1,
            "adapter_applied": 1,
            "top1_changed": 1,
            "top1_change_rate_total": .1,
            "top1_change_rate_given_allow": 1.0,
            "diagnostic_case": "C_TOP1_CHANGES",
        }), encoding="utf-8"
    )
    (tmp_path / "input/seed-ledger.json").write_text(
        json.dumps({"probe30": list(range(30)), "v36_dev30": [100], "v37_dev50": [200]}),
        encoding="utf-8",
    )
    result = report.recover(tmp_path / "input", tmp_path / "output", 37621983723)
    assert result["candidate_wins"] == 2
    assert result["parent_wins"] == 2
    assert result["candidate_only_wins"] == 1
    assert result["parent_only_wins"] == 1
    assert result["net_new_wins"] == 0
    assert result["report_recovered_from_artifact"] is True
    assert "Probe30" in (tmp_path / "output/report.md").read_text(encoding="utf-8")
