#!/usr/bin/env python3
"""Serialization-only wrapper for the frozen two-card V2 discovery experiment.

The underlying experiment can produce +inf as a diagnostic semantic margin when
there is only one ranked semantic action. JSON evidence forbids non-finite float
values, so this wrapper normalizes only evidence hashing: +inf/-inf become the
strings "Infinity"/"-Infinity" and NaN becomes "NaN". Candidate selection,
thresholds, seeds, Search scores, oracle scores, and eligibility are untouched.
"""
from __future__ import annotations

import hashlib
import math

import run_phase1_public_search_v2_jaw_worm_turn_one_two_card_midturn_discovery as experiment
from roguelike_ai.sts1_teacher.contract import canonical_json


def _finite_evidence(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _finite_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_evidence(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_evidence(item) for item in value]
    return value


def evidence_digest(value) -> str:
    normalized = _finite_evidence(value)
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


experiment.digest = evidence_digest
experiment.main()
