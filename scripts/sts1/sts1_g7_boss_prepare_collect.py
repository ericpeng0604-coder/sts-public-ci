"""Round017 fresh, unchanged-G7 Train preparation facts; eleven episodes only."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sts1_g7_live_train_collect as base
from roguelike_ai.sts1_phase3.boss_preparation import profile_trace, public_summary

ROUND = "round-017-20261011"
SEEDS_NAME = "round-017-seeds"


def configure(args):
    args.extra_identity_files = {
        "boss_preparation_collector": Path(__file__),
        "boss_preparation_profiler": base.REPO / "src/roguelike_ai/sts1_phase3/boss_preparation.py",
    }
    return args


def prepare(args, root):
    prior = base.private_path(args.prior_root) / "round-016-seeds"
    lock_path = prior / "h22-identity-private.json"
    old = json.loads(lock_path.read_text())
    actual = base.evaluation._sha256(base.runtime_files(args)["simulator_binding_sha256"])
    if actual != old["expected_sha256"]["simulator_binding_sha256"]:
        raise ValueError("corrected H22 native binding must remain unchanged")
    return base.prepare(args, root, round_id=ROUND, seeds_name=SEEDS_NAME,
                        prior_seeds_dir=prior, prior_lock_path=lock_path)


def collect(args, root):
    profiles = []

    def observe(events, metadata):
        profile = profile_trace(events, metadata)
        # Preserve every diagnostic privately; no snapshot is sent to a model.
        base.write_new(root / f"preparation-pair-{metadata['pair_index']:02d}-private.json", profile)
        profiles.append(profile)

    summary = base.collect(args, root, round_id=ROUND, seeds_name=SEEDS_NAME, trace_observer=observe)
    if len(profiles) != 10 or summary["actual_episodes"] != 11 or summary["trace_invariance"] != "PASS":
        raise ValueError("ten complete Train profiles and one invariant replay required")
    report = public_summary(profiles)
    base.write_new(root / "boss-preparation-public.json", report)
    return {**summary, "preparation": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "collect"))
    for name in ("private-root", "prior-root", "module-dir", "armg-root", "g7-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = configure(parser.parse_args())
    root = base.private_path(args.private_root)
    try:
        if args.mode == "prepare":
            result = prepare(args, root)
        elif args.mode == "preflight":
            base.validate_preflight(args, root, ROUND, SEEDS_NAME)
            result = {"status": "PREFLIGHT_PASS", "expected_episodes": 11}
        else:
            with (root / "runtime-private.log").open("x", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = collect(args, root)
    except Exception as exc:
        result = {"status": "NOT_VERIFIED", "error_class": type(exc).__name__}
        if root.is_dir():
            with (root / "failure-private.txt").open("a", encoding="utf-8") as handle:
                handle.write(repr(exc) + "\n")
        print(json.dumps(result))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
