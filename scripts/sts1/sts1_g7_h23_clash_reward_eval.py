"""Prospective H23 pools, freeze and staged evaluation; private evidence only."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sts1_g7_live_train_collect as collection
import sts1_g7_h16_lethal_defend_eval as evaluator
from roguelike_ai.sts1_phase3.card_preparation import load_native_types

ROUND = "round-018-20261011"
SEEDS = "round-018-seeds"


def configure(args):
    args.extra_identity_files = {
        **evaluator.h23_identity_paths(args.native_source),
        "candidate_selector_sha256": collection.REPO / "src/roguelike_ai/sts1_phase3/clash_reward_skip.py",
        "h23_parent_summary": Path(evaluator.h3.__file__),
        "h23_pair_integrity": Path(evaluator.h2.__file__),
        "h23_transition_helper": Path(evaluator.h15.__file__),
    }
    return args


def prepare(args, root):
    prior = collection.private_path(args.prior_root)
    old = json.loads((prior / "identity-private.json").read_text())
    load_native_types(args.native_source)
    actual = evaluator._sha256(args.module_dir / "slaythespire.cp312-win_amd64.pyd")
    if actual != old["expected_sha256"]["simulator_binding_sha256"]:
        raise ValueError("corrected native binding drift")
    summary = collection.prepare(args, root, round_id=ROUND, seeds_name=SEEDS,
        prior_seeds_dir=prior / "round-017-seeds", prior_lock_path=prior / "identity-private.json", expected_episodes=21)
    lock = json.loads((root / "identity-private.json").read_text())
    expected_keys = set(evaluator.PRIVATE_IDENTITY_KEYS) | {"candidate_selector_sha256"} | set(evaluator.h23_identity_paths(args.native_source))
    collection.write_new(root / SEEDS / "h23-identity-private.json", {
        "schema_version": evaluator.PRIVATE_IDENTITY_SCHEMA_VERSION, "trial_id": "h23",
        "candidate_commit": lock["code_commit"], "simulator_gameplay_commit": evaluator.PINNED_GAMEPLAY_COMMIT,
        "mcts_sims": 2000, "expected_sha256": {k: lock["expected_sha256"][k] for k in expected_keys},
    })
    return {**summary, "expected_episodes": 21, "paired_seeds": 10}


def run_stage(args, root, preflight=False):
    lock, _, _ = collection.validate_preflight(args, root, ROUND, SEEDS)
    stage = args.stage
    seeds_dir = root / SEEDS
    argv = ["--trial-id", "h23", "--stage", stage, "--candidate-commit", lock["code_commit"]]
    for key, value in {
        "pool-file": seeds_dir / "pools" / ("train_hypothesis_1.json" if stage == "train" else stage + ".json"),
        "pools-dir": seeds_dir / "pools", "exclusion-inventory": seeds_dir / "exclusion-inventory.json",
        "module-dir": args.module_dir, "armg-root": args.armg_root, "g7-checkpoint": args.g7_checkpoint,
        "native-source": args.native_source,
        "private-output-dir": root / f"round-018-h23-{stage}-20261011",
        "private-usage-ledger": seeds_dir / "h23-usage-private.jsonl",
        "private-identity-lock": seeds_dir / "h23-identity-private.json",
    }.items():
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
    # Preserve native stdout, exact failures and private hashes outside Git.
    with (root / f"{args.stage}-{args.mode}-runtime-private.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                result = run_stage(args, root, args.mode == "preflight")
            except Exception as exc:
                print(repr(exc))
                result = 2
    if args.mode == "run" and result == 0:
        summary = json.loads((root / f"round-018-h23-{args.stage}-20261011/private-summary.json").read_text())
        print(json.dumps({k: summary[k] for k in ("status", "stage", "episodes", "paired",
            "candidate_override_count", "effective_override_seed_count", "advance_eligible", "safety")}))
    else:
        print(json.dumps({"status": "PASS" if result == 0 else "NOT_VERIFIED", "stage": args.stage}))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
