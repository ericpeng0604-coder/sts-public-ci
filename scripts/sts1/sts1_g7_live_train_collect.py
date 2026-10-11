"""Collect fresh G7 Train diagnostics; never select a candidate or open held-out data."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/sts1"))
import sts1_g7_h16_lethal_defend_eval as evaluation
import sts1_g7_seed_ledger as ledger

ROUND = "round-014-20261011"
POOL_NAMES = tuple(ledger.EXPLORATION_POOL_SIZES)
IDENTITY_FILES = {
    "collector": Path(__file__),
    "simulator_policy_source_sha256": REPO / "src/roguelike_ai/sts1_phase3/simulator.py",
    "candidate_evaluator_sha256": Path(evaluation.__file__),
    "trace_validator": Path(evaluation.h7.__file__),
    "seed_ledger": Path(ledger.__file__),
}


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def head() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "rev-parse", "HEAD"],
        cwd=REPO, text=True,
    ).strip()


def private_path(path: Path) -> Path:
    path = path.resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("private evidence must remain outside the Git repository")
    return path


def runtime_files(args) -> dict[str, Path]:
    return {
        **IDENTITY_FILES,
        **getattr(args, "extra_identity_files", {}),
        "simulator_binding_sha256": args.module_dir / "slaythespire.cp312-win_amd64.pyd",
        "armg_source_sha256": args.armg_root / "armG_train.py",
        "armg_vocab_sha256": args.armg_root / "armS_card_vocab.json",
        "g7_checkpoint_sha256": args.g7_checkpoint,
    }


def prepare(args, root: Path, *, round_id: str = ROUND,
            seeds_name: str = "round-014-seeds", prior_seeds_dir: Path | None = None,
            prior_lock_path: Path | None = None) -> dict:
    prior = private_path(args.prior_root)
    prior_seeds_dir = private_path(prior_seeds_dir or prior / "round-013-seeds")
    prior_lock_path = private_path(prior_lock_path or prior / "round-013-seeds/h20-identity-private.json")
    inventory = json.loads((prior_seeds_dir / "exclusion-inventory.json").read_text())
    pools = [json.loads((prior_seeds_dir / "pools" / f"{name}.json").read_text())
             for name in POOL_NAMES]
    inventory = ledger.extend_inventory_with_pools(
        inventory, pools, inventory_id=round_id + "-exclusions",
    )
    excluded, sources = ledger.validate_inventory(inventory)
    old_lock = json.loads(prior_lock_path.read_text())
    hashes = {key: evaluation._sha256(path) for key, path in runtime_files(args).items()}
    for key in ("armg_source_sha256", "armg_vocab_sha256", "g7_checkpoint_sha256"):
        if hashes[key] != old_lock["expected_sha256"][key]:
            raise ValueError("frozen G7 or ArmG identity differs")
    evaluation._validate_pinned_gameplay_manifest(REPO / "external/sts_lightspeed/UPSTREAM.json")
    if Path(str(args.g7_checkpoint) + ".adapter.pt").exists():
        raise ValueError("G7 adapter sidecar is forbidden")
    root.mkdir(exist_ok=False)
    seeds_dir = root / seeds_name
    seeds_dir.mkdir()
    inventory_path = seeds_dir / "exclusion-inventory.json"
    write_new(inventory_path, inventory)
    key = secrets.token_hex(32)
    (seeds_dir / "generation-key.txt").write_text(key, encoding="utf-8")
    generated = ledger.generate_exploration_round(
        inventory, inventory_sha256=evaluation._sha256(inventory_path),
        round_id=round_id, generation_key=key,
    )
    ledger._verify_generated(generated, excluded)
    (seeds_dir / "pools").mkdir()
    write_new(seeds_dir / "pools/ledger.json", generated)
    for name, pool in generated["pools"].items():
        write_new(seeds_dir / "pools" / f"{name}.json", pool)
    lock = {"round_id": round_id, "code_commit": head(), "expected_sha256": hashes,
            "pool_ledger_sha256": evaluation._sha256(seeds_dir / "pools/ledger.json"),
            "inventory_sha256": evaluation._sha256(inventory_path),
            "prior_identity_lock_sha256": evaluation._sha256(prior_lock_path),
            "source_manifest_count": len(sources), "excluded_seed_count": len(excluded),
            "expected_episodes": 11, "baseline_seed_count": 10, "mcts_sims": 2000}
    write_new(root / "identity-private.json", lock)
    return {"status": "PREPARED", "excluded_seed_count": len(excluded),
            "source_manifest_count": len(sources), "reserved_seed_count": 70,
            "baseline_seed_count": 10, "expected_episodes": 11}


def validate_preflight(args, root: Path, round_id: str = ROUND,
                       seeds_name: str = "round-014-seeds") -> tuple[dict, tuple[int, ...], dict]:
    lock = json.loads((root / "identity-private.json").read_text())
    if lock.get("round_id") != round_id or lock.get("code_commit") != head():
        raise ValueError("frozen collection code differs")
    actual = {key: evaluation._sha256(path) for key, path in runtime_files(args).items()}
    if actual != lock["expected_sha256"] or lock.get("mcts_sims") != 2000:
        raise ValueError("runtime identity differs")
    if os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1":
        raise ValueError("Teacher reranking must be disabled")
    if Path(str(args.g7_checkpoint) + ".adapter.pt").exists():
        raise ValueError("G7 adapter sidecar is forbidden")
    seeds_dir = root / seeds_name
    inventory_path = seeds_dir / "exclusion-inventory.json"
    pool_path = seeds_dir / "pools/ledger.json"
    if (evaluation._sha256(inventory_path) != lock["inventory_sha256"]
            or evaluation._sha256(pool_path) != lock["pool_ledger_sha256"]):
        raise ValueError("frozen pool or exclusions differ")
    inventory = json.loads(inventory_path.read_text())
    excluded, _ = ledger.validate_inventory(inventory)
    generated = json.loads(pool_path.read_text())
    expected = ledger.generate_exploration_round(
        inventory, inventory_sha256=evaluation._sha256(inventory_path), round_id=round_id,
        generation_key=(seeds_dir / "generation-key.txt").read_text(),
    )
    if generated != expected:
        raise ValueError("pool generation differs")
    ledger._verify_generated(generated, excluded)
    pool = generated["pools"]["train_hypothesis_1"]
    seeds = tuple(pool["seed_ids"])
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("ten unique Train seeds required")
    for name, manifest in generated["pools"].items():
        if json.loads((seeds_dir / "pools" / f"{name}.json").read_text()) != manifest:
            raise ValueError("pool manifest differs")
    return lock, seeds, pool


def collect(args, root: Path, *, round_id: str = ROUND,
            seeds_name: str = "round-014-seeds", trace_observer=None) -> dict:
    lock, seeds, pool = validate_preflight(args, root, round_id, seeds_name)
    usage = root / "usage-private.jsonl"
    if usage.exists() or (root / "episodes").exists():
        raise ValueError("collection already started; no automatic rerun")
    output = root / "episodes"
    output.mkdir()
    evaluation._append_jsonl(usage, {"status": "RUNNING", **lock,
                                   "reserved_pools": list(POOL_NAMES)})
    summary = {"status": "RUNNING", "baseline_completed": 0, "baseline_wins": 0,
               "actual_episodes": 0, "trace_invariance": "NOT_RUN",
               "safety": {key: 0 for key in evaluation.SAFETY_COUNTERS}}
    sts = evaluation._load_sts(args.module_dir)
    policy = evaluation.ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.g7_checkpoint)
    first = None
    for index, seed in enumerate(seeds, 1):
        metadata = {"round_id": round_id, "stage": "train", "purpose": "baseline_live_resource_diagnosis",
                    "code_commit": lock["code_commit"], "pair_index": index,
                    "seed_id": seed, "pool_manifest_sha256": pool["manifest_sha256"],
                    "private_runtime": lock["expected_sha256"], "mcts_sims": 2000}
        modes = ("on", "off") if index == 1 else ("on",)
        for mode in modes:
            evidence = output / f"pair-{index:02d}-parent-{mode}.evidence.ndjson"
            trace = output / f"pair-{index:02d}-parent.trace.ndjson"
            evaluation._append_jsonl(usage, {"status": "EPISODE_STARTED", "seed_id": seed,
                                           "index": index, "trace_mode": mode})
            result = evaluation.run_simulator_game(
                student=None, sts=sts, seed=seed, evidence_path=evidence,
                armg_policy=policy, combat_mcts_sims=2000, training_seeds=seeds,
                collect_ppo=False, collect_teacher=False,
                diagnostic_trace_path=trace if mode == "on" else None,
                diagnostic_metadata=metadata,
            )
            summary["actual_episodes"] += 1
            evaluation._check_pair_integrity(evidence, result)
            evaluation._accumulate_safety(summary["safety"], result)
            signature, count = evaluation._evidence_action_signature(evidence)
            if mode == "on":
                evaluation.h7._validate_trace(trace, result, metadata)
                events = list(evaluation.h7._read_jsonl(trace))
                if events[0].get("live_combat_resources_required") is not True:
                    raise ValueError("new live resource trace header is required")
                evaluation._validate_action_trace_completeness(events)
                if trace_observer is not None:
                    trace_observer(events, metadata)
                summary["baseline_completed"] += 1
                summary["baseline_wins"] += result["outcome"] == "victory"
                if index == 1:
                    first = (result, signature, count)
            elif not evaluation._trace_replay_matches(first[0], result, first[1], signature,
                                                     first[2], count):
                raise ValueError("full-episode trace invariance failed")
            else:
                summary["trace_invariance"] = "PASS"
            evaluation._append_jsonl(usage, {"status": "EPISODE_COMPLETE", "index": index,
                                           "trace_mode": mode, "result": result})
            (root / "status-public.json").write_text(json.dumps(summary), encoding="utf-8")
    summary["status"] = "COMPLETE"
    (root / "status-public.json").write_text(json.dumps(summary), encoding="utf-8")
    evaluation._append_jsonl(usage, {"status": "COMPLETE", "summary": summary})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "preflight", "collect"))
    for name in ("private-root", "prior-root", "module-dir", "armg-root", "g7-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = private_path(args.private_root)
    try:
        if args.mode == "prepare":
            result = prepare(args, root)
        elif args.mode == "preflight":
            validate_preflight(args, root)
            result = {"status": "PREFLIGHT_PASS", "expected_episodes": 11}
        else:
            with (root / "runtime-private.log").open("x", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = collect(args, root)
    except Exception as exc:
        result = {"status": "NOT_VERIFIED", "error_class": type(exc).__name__}
        if root.is_dir():
            with (root / "failure-private.txt").open("a", encoding="utf-8") as log:
                log.write(repr(exc) + "\n")
        print(json.dumps(result))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
