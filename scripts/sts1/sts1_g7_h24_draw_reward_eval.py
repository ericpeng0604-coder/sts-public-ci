"""Frozen H24 Train10/Probe10/Dev30; no automatic retry or extra sample."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sts1_g7_live_train_collect as collection
import sts1_g7_h16_lethal_defend_eval as evaluator
from roguelike_ai.sts1_phase3.card_preparation import load_native_types

ROUND = "round-019-20261011"
SEEDS = "round-019-seeds"


def configure(args):
    args.extra_identity_files = {**evaluator.h24_identity_paths(args.native_source),
        "candidate_selector_sha256": collection.REPO / "src/roguelike_ai/sts1_phase3/draw_reward.py",
        "h24_parent_summary": Path(evaluator.h3.__file__),
        "h24_pair_integrity": Path(evaluator.h2.__file__),
        "h24_transition_helper": Path(evaluator.h15.__file__)}
    return args


def prepare(args, root):
    prior = collection.private_path(args.prior_root)
    old = json.loads((prior / "identity-private.json").read_text())
    load_native_types(args.native_source)
    if evaluator._sha256(args.module_dir / "slaythespire.cp312-win_amd64.pyd") != old["expected_sha256"]["simulator_binding_sha256"]:
        raise ValueError("corrected native binding drift")
    result = collection.prepare(args, root, round_id=ROUND, seeds_name=SEEDS,
        prior_seeds_dir=prior / "round-018-seeds", prior_lock_path=prior / "identity-private.json", expected_episodes=21)
    lock = json.loads((root / "identity-private.json").read_text())
    keys = set(evaluator.PRIVATE_IDENTITY_KEYS) | {"candidate_selector_sha256"} | set(evaluator.h24_identity_paths(args.native_source))
    collection.write_new(root / SEEDS / "h24-identity-private.json", {
        "schema_version": evaluator.PRIVATE_IDENTITY_SCHEMA_VERSION, "trial_id": "h24",
        "candidate_commit": lock["code_commit"], "simulator_gameplay_commit": evaluator.PINNED_GAMEPLAY_COMMIT,
        "mcts_sims": 2000, "expected_sha256": {k: lock["expected_sha256"][k] for k in keys}})
    return {**result, "expected_episodes": 21, "paired_seeds": 10}


def run_stage(args, root, preflight=False):
    lock, _, _ = collection.validate_preflight(args, root, ROUND, SEEDS)
    stage, seeds = args.stage, root / SEEDS
    argv = ["--trial-id", "h24", "--stage", stage, "--candidate-commit", lock["code_commit"]]
    for key, value in {
        "pool-file": seeds / "pools" / ("train_hypothesis_1.json" if stage == "train" else stage + ".json"),
        "pools-dir": seeds / "pools", "exclusion-inventory": seeds / "exclusion-inventory.json",
        "module-dir": args.module_dir, "armg-root": args.armg_root, "g7-checkpoint": args.g7_checkpoint,
        "native-source": args.native_source,
        "private-output-dir": root / f"round-019-h24-{stage}-20261011",
        "private-usage-ledger": seeds / "h24-usage-private.jsonl",
        "private-identity-lock": seeds / "h24-identity-private.json"}.items():
        argv.extend(["--" + key, str(value)])
    if preflight:
        argv.append("--preflight-only")
    return evaluator.main(argv)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "preflight", "run"))
    p.add_argument("--stage", choices=("train", "probe", "dev"), default="train")
    for name in ("private-root", "prior-root", "native-source", "module-dir", "armg-root", "g7-checkpoint"):
        p.add_argument("--" + name, required=True, type=Path)
    args = configure(p.parse_args())
    root = collection.private_path(args.private_root)
    if args.mode == "prepare":
        print(json.dumps(prepare(args, root)))
        return 0
    with (root / f"{args.stage}-{args.mode}-runtime-private.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                result = run_stage(args, root, args.mode == "preflight")
            except Exception as exc:
                print(repr(exc))
                result = 2
    if args.mode == "run" and result == 0:
        s = json.loads((root / f"round-019-h24-{args.stage}-20261011/private-summary.json").read_text())
        print(json.dumps({k: s[k] for k in ("status", "stage", "episodes", "paired", "candidate_override_count",
                                          "effective_override_seed_count", "advance_eligible", "safety")}))
    else:
        print(json.dumps({"status": "PASS" if result == 0 else "NOT_VERIFIED", "stage": args.stage}))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
