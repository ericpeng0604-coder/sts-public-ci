#!/usr/bin/env python3
"""Apply one PPO v1.4 round result to durable loop state.

The legacy file name offline-champion.pt is retained for compatibility with the
existing rollout workflow, but from v1.4 onward it is the cumulative Training
Parent. The production Champion remains untouched until real-game validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--candidate-critic", type=Path, required=True)
    p.add_argument("--gate-summary", type=Path, required=True)
    p.add_argument("--run-id", type=int, required=True)
    args = p.parse_args()

    state_path = args.state_dir / "state.json"
    parent_path = args.state_dir / "offline-champion.pt"
    critic_path = args.state_dir / "training-critic.pt"
    pending_path = args.state_dir / "pending-real-game.pt"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))

    if gate.get("schema_version") != "sts1-armg-ppo-v14-two-level-gate-v1":
        raise RuntimeError("unexpected v1.4 gate schema")
    if not args.candidate.is_file() or not args.candidate_critic.is_file():
        raise RuntimeError("candidate actor/critic missing")

    adopt_parent = gate["decision"] == "ADOPT_PARENT"
    ready_real = gate["final_readiness"] == "READY_FOR_REAL_GAME"

    state["round_index"] = int(state.get("round_index", 0)) + 1
    state["parent_generation"] = int(
        state.get("parent_generation", state.get("generation", 0))
    )

    if adopt_parent:
        shutil.copy2(args.candidate, parent_path)
        state["parent_generation"] += 1
        state["generation"] = state["parent_generation"]
        state["accepted_parent_rounds"] = int(
            state.get("accepted_parent_rounds", 0)
        ) + 1
        state["stagnation_count"] = 0
    else:
        state["rejected_parent_rounds"] = int(
            state.get("rejected_parent_rounds", 0)
        ) + 1
        state["stagnation_count"] = int(state.get("stagnation_count", 0)) + 1

    # The critic learns from the parent-policy rollouts even when the actor
    # candidate is rejected, so it is safe and useful to keep its newer value
    # estimate rather than reinitializing next round.
    shutil.copy2(args.candidate_critic, critic_path)

    if ready_real:
        shutil.copy2(args.candidate, pending_path)
        state["pending_real_game_validation"] = True
        state["pending_real_game_candidate_sha256"] = sha256(pending_path)
        state["final_ready_rounds"] = int(state.get("final_ready_rounds", 0)) + 1

    state.update(
        {
            "schema_version": "sts1-armg-ppo-v14-loop-state-v1",
            "training_parent_sha256": sha256(parent_path),
            # Keep this legacy key aligned for old integrity checks.
            "offline_champion_sha256": sha256(parent_path),
            "training_critic_sha256": sha256(critic_path),
            "last_candidate_sha256": sha256(args.candidate),
            "last_decision": gate["decision"],
            "last_dev_gate": gate["dev_gate"],
            "last_final_gate_30": gate["final_gate_30"],
            "last_final_gate_50": gate["final_gate_50"],
            "last_final_readiness": gate["final_readiness"],
            "last_run_id": args.run_id,
            "production_champion_replaced": False,
        }
    )

    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PPO_V14_LOOP_STATE",
        json.dumps(
            {
                "round_index": state["round_index"],
                "parent_generation": state["parent_generation"],
                "decision": state["last_decision"],
                "final_readiness": state["last_final_readiness"],
                "stagnation_count": state["stagnation_count"],
                "training_parent_sha256": state["training_parent_sha256"],
                "training_critic_sha256": state["training_critic_sha256"],
                "pending_real_game_validation": state.get(
                    "pending_real_game_validation", False
                ),
                "production_champion_replaced": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
