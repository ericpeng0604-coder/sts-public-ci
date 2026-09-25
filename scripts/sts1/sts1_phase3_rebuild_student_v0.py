#!/usr/bin/env python3
"""Rebuild the exact frozen STS1 Student v0 artifact from accepted Phase-2 evidence.

This helper intentionally uses the already-frozen Phase-2 reconstruction code.
It writes only the Phase-3 inference envelope after the model hash has matched
the accepted frozen candidate exactly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


EXPECTED_PREDECESSOR_HEAD = "046854ecfbb497f2fc5e4379990300e46be0c248"
EXPECTED_SOURCE_SHA256 = "06e206b7f04e6e7b77ad1fc8c35ba1bcdca1c978596fbb176933323e3940aed3"
EXPECTED_CONFIG_HASH = "906afcaec38a2050f48e28f2351745408bbc0c72607704dbce2e134cdff4192c"
EXPECTED_MODEL_SHA256 = "e15604b95247615a6d424f834e8b8a1e9fd680af4fe9e22a6754372027e89513"
EXPECTED_DATASET_HASH = "4509f9c48206606638f24f264c0dbc471b1cc46c60f2633b3659f72cf7c6ccea"
EXPECTED_FEATURE_COUNT = 3028
ARTIFACT_SCHEMA_VERSION = "sts1-phase3-frozen-student-v0-artifact-v1"
STUDENT_SCHEMA_VERSION = "sts1-student-v0-linear-v1"


def _load_holdout_module(phase2_root: Path):
    phase2_src = phase2_root / "src"
    sys.path.insert(0, str(phase2_src))
    script = phase2_root / "scripts" / "sts1" / "sts1_phase2_student_v0_holdout_eval.py"
    spec = importlib.util.spec_from_file_location("phase2_student_v0_holdout_eval_rebuild", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Phase-2 holdout evaluator: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phase2_root = args.phase2_root.resolve()
    if not phase2_root.is_dir():
        raise RuntimeError(f"Phase-2 checkout missing: {phase2_root}")

    module = _load_holdout_module(phase2_root)
    pilot = module._load_dataset_pilot(phase2_root)

    with tempfile.TemporaryDirectory(prefix="sts1-phase3-rebuild-v0-") as tmp:
        _, weights, _, frozen = module._freeze_candidate(
            phase2_root,
            pilot,
            Path(tmp) / "dataset",
        )

    exact = {
        "student_source_sha256": EXPECTED_SOURCE_SHA256,
        "student_config_hash": EXPECTED_CONFIG_HASH,
        "student_model_sha256": EXPECTED_MODEL_SHA256,
        "dataset_hash": EXPECTED_DATASET_HASH,
        "feature_count": EXPECTED_FEATURE_COUNT,
    }
    for key, value in exact.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"frozen Student identity drift for {key}: {frozen.get(key)!r}")
    if len(weights) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(f"feature-count drift: {len(weights)}")

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "predecessor_head": EXPECTED_PREDECESSOR_HEAD,
        "student_source_sha256": EXPECTED_SOURCE_SHA256,
        "student_config_hash": EXPECTED_CONFIG_HASH,
        "student_model_sha256": EXPECTED_MODEL_SHA256,
        "dataset_hash": EXPECTED_DATASET_HASH,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "student_schema_version": STUDENT_SCHEMA_VERSION,
        "weights": {key: weights[key] for key in sorted(weights)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "result": "PASS_EXACT_STUDENT_V0_REBUILT",
        "output": str(args.output),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "feature_count": len(weights),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
