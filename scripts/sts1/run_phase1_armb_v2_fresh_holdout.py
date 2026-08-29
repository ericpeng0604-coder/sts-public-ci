#!/usr/bin/env python3
"""Run one shard of the frozen Arm B v2 fresh-holdout benchmark.

This wrapper reuses the already-validated oracle harness unchanged. It only
copies the predeclared fresh 50-seed file into the hydrated upstream layout and
points the runner at that file. The original Jialeiv 50-seed benchmark remains
immutable discovery/failure evidence.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
FRESH_SOURCE = ROOT / "tests" / "data" / "sts1_phase1_armb_v2_fresh_seeds_50.txt"
RUNNER_PATH = ROOT / "scripts" / "sts1" / "run_phase1_frozen_oracle_benchmark.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("phase1_frozen_oracle_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable_to_load_frozen_oracle_runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner()
    original = set(runner.load_heldout_seeds(runner.SEEDS_PATH))
    fresh = tuple(runner.load_heldout_seeds(FRESH_SOURCE))
    if original.intersection(fresh):
        raise RuntimeError("fresh_holdout_overlaps_original_50")

    target = runner.AGENT / "eval" / "phase1_armb_v2_fresh_seeds_50.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FRESH_SOURCE, target)
    runner.SEEDS_PATH = target
    runner.main()


if __name__ == "__main__":
    main()
