#!/usr/bin/env python3
"""Hydrate the exact upstream sources for STS1 Jialeiv Baseline v0.

This script intentionally does not alter gameplay logic. It clones fixed commits,
verifies provenance, applies the upstream-published simulator patch, and records
cryptographic identities for the frozen reproduction evidence.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_META = ROOT / "external" / "sts-rl-agent" / "UPSTREAM.json"
SIM_META = ROOT / "external" / "sts_lightspeed" / "UPSTREAM.json"
AGENT_DST = ROOT / "external" / "sts-rl-agent" / "source"
SIM_DST = ROOT / "external" / "sts_lightspeed" / "source"
EVIDENCE = ROOT / "evidence" / "sts1" / "baseline_v0"


def run(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        list(args), cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc.stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hydrate(meta: dict, dst: Path, *, submodules: bool = False) -> str:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--no-checkout", meta["repository"], str(dst))
    run("git", "checkout", "--detach", meta["commit"], cwd=dst)
    head = run("git", "rev-parse", "HEAD", cwd=dst)
    if head != meta["commit"]:
        raise RuntimeError(f"Pinned HEAD mismatch for {dst}: {head} != {meta['commit']}")
    if submodules:
        run("git", "submodule", "update", "--init", "--recursive", cwd=dst)
    return head


def main() -> None:
    agent = json.loads(AGENT_META.read_text(encoding="utf-8"))
    sim = json.loads(SIM_META.read_text(encoding="utf-8"))

    agent_head = hydrate(agent, AGENT_DST)
    sim_head = hydrate(sim, SIM_DST, submodules=True)

    pybind_meta = sim["pybind11_build_compat"]
    pybind_dir = SIM_DST / "pybind11"
    pybind_original = run("git", "rev-parse", "HEAD", cwd=pybind_dir)
    if pybind_original != pybind_meta["upstream_submodule_commit"]:
        raise RuntimeError(
            f"Pinned pybind11 submodule mismatch: {pybind_original} != "
            f"{pybind_meta['upstream_submodule_commit']}"
        )
    run("git", "fetch", "origin", pybind_meta["override_tag"], cwd=pybind_dir)
    run("git", "checkout", "--detach", pybind_meta["override_commit"], cwd=pybind_dir)
    pybind_active = run("git", "rev-parse", "HEAD", cwd=pybind_dir)
    if pybind_active != pybind_meta["override_commit"]:
        raise RuntimeError(
            f"pybind11 compatibility override mismatch: {pybind_active} != "
            f"{pybind_meta['override_commit']}"
        )

    patch = AGENT_DST / agent["patch_path"]
    weight = AGENT_DST / agent["weight_path"]
    seeds = AGENT_DST / agent["eval_seed_path"]
    if not patch.is_file() or not weight.is_file() or not seeds.is_file():
        raise RuntimeError("Pinned upstream snapshot is missing patch, weight, or eval seeds")

    seed_values = [
        line.strip() for line in seeds.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(seed_values) != 50 or len(set(seed_values)) != 50:
        raise RuntimeError(f"Expected exactly 50 unique held-out seeds, got {len(seed_values)}")

    # Apply exactly the patch shipped in the pinned sts-rl-agent commit.
    run("git", "apply", "--check", str(patch), cwd=SIM_DST)
    run("git", "apply", str(patch), cwd=SIM_DST)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    manifest_path = EVIDENCE / "manifest.json"
    manifest = {
        "schema": "sts1-jialeiv-baseline-v0-evidence-v1",
        "baseline": "STS1_BASELINE_V0",
        "status": "NOT_FROZEN",
        "sts_rl_agent": {
            "repository": agent["repository"],
            "commit": agent_head,
            "tree": agent["tree"],
            "license": agent["license"],
        },
        "sts_lightspeed": {
            "repository": sim["repository"],
            "commit": sim_head,
            "license": sim["license"],
        },
        "build_dependencies": {
            "pybind11": {
                "repository": pybind_meta["override_repository"],
                "upstream_submodule_commit": pybind_original,
                "upstream_version": pybind_meta["upstream_version"],
                "active_commit": pybind_active,
                "active_tag": pybind_meta["override_tag"],
                "compatibility_only": True,
                "reason": pybind_meta["reason"],
            }
        },
        "artifacts": {
            "sim_patch_path": agent["patch_path"],
            "sim_patch_sha256": sha256(patch),
            "model_path": agent["weight_path"],
            "model_sha256": sha256(weight),
            "eval_seeds_path": agent["eval_seed_path"],
            "eval_seeds_sha256": sha256(seeds),
            "eval_seed_count": len(seed_values),
        },
        "evaluation_contract": {
            "character": "IRONCLAD",
            "ascension": 0,
            "held_out_seeds": 50,
            "mcts_budgets": [2000, 50000],
            "workers": 12,
            "upstream_command": "python eval/armB_blind.py 2000,50000 12",
            "upstream_targets": agent["upstream_target"],
        },
        "hidden_rng": {
            "upstream_behavior_preserved": True,
            "public_state_fix_applied": False,
            "note": "Baseline v0 intentionally reproduces upstream search, including its hidden-future RNG advantage.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "sts_rl_agent_sha": agent_head,
        "sts_lightspeed_sha": sim_head,
        "sim_patch_sha256": manifest["artifacts"]["sim_patch_sha256"],
        "model_sha256": manifest["artifacts"]["model_sha256"],
        "eval_seeds_sha256": manifest["artifacts"]["eval_seeds_sha256"],
        "eval_seed_count": len(seed_values),
        "pybind11_upstream_sha": pybind_original,
        "pybind11_active_sha": pybind_active,
    }, indent=2))


if __name__ == "__main__":
    main()
