from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sts1_activation_probe_v38",
    ROOT / "scripts/sts1/sts1_build_rescue_activation_probe_v38.py",
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


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
