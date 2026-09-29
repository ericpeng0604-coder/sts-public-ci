#!/usr/bin/env python3
"""Choose the next safe PPO v1.4 training profile after a completed loop round.

The evaluator/gate never changes here. Only training-data volume and bounded PPO
hyperparameters are adapted. Technical failures must bypass this script so a
retry uses the exact same training profile.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Profile:
    name: str
    games_per_worker: int
    temperature: float
    epochs: int
    learning_rate: float
    clip: float
    target_kl: float
    entropy: float
    anchor_coef: float


PROFILES = {
    "stable": Profile("stable", 50, 1.00, 6, 3e-5, 0.20, 0.020, 0.0010, 0.020),
    "diversify": Profile("diversify", 60, 1.04, 6, 3.5e-5, 0.20, 0.018, 0.0015, 0.018),
    "broaden": Profile("broaden", 75, 1.07, 6, 4e-5, 0.18, 0.015, 0.0020, 0.015),
    "escape": Profile("escape", 100, 1.10, 6, 4.5e-5, 0.16, 0.012, 0.0030, 0.012),
    "wide_explore": Profile("wide_explore", 100, 1.12, 6, 4e-5, 0.15, 0.010, 0.0040, 0.010),
    "near_miss_refine": Profile("near_miss_refine", 80, 0.98, 6, 2.5e-5, 0.12, 0.008, 0.0005, 0.030),
}


def parse_control(path: Path) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"invalid control line: {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise RuntimeError("empty control key")
        if key not in values:
            order.append(key)
        values[key] = value.strip()
    return order, values


def render_control(order: list[str], values: dict[str, str]) -> str:
    return "".join(f"{key}={values[key]}\n" for key in order)


def _last_dev_gate(state: dict[str, Any]) -> dict[str, Any]:
    gate = state.get("last_dev_gate")
    if not isinstance(gate, dict):
        gate = state.get("last_gate_30")
    return gate if isinstance(gate, dict) else {}


def _last_win_delta(state: dict[str, Any]) -> int:
    value = _last_dev_gate(state).get("win_delta", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _last_floor_delta(state: dict[str, Any]) -> float:
    value = _last_dev_gate(state).get("mean_paired_floor_delta", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def choose_profile(state: dict[str, Any]) -> Profile:
    stagnation = int(state.get("stagnation_count", 0))
    if stagnation < 0:
        raise RuntimeError("stagnation_count cannot be negative")

    if state.get("last_decision") in {"PROMOTE_OFFLINE", "ADOPT_PARENT"} or stagnation <= 1:
        return PROFILES["stable"]
    if stagnation == 2:
        return PROFILES["diversify"]
    if stagnation == 3:
        return PROFILES["broaden"]
    if stagnation == 4:
        return PROFILES["escape"]

    # Long plateaus must not collapse into an endless wide-explore loop.
    # If the candidate is already close on paired floor quality without losing
    # wins, refine it. Otherwise alternate broad exploration and conservative
    # refinement so a failed exploration profile is not repeated forever.
    win_delta = _last_win_delta(state)
    floor_delta = _last_floor_delta(state)
    if win_delta >= 0 and floor_delta >= 0.25:
        return PROFILES["near_miss_refine"]
    if stagnation % 2 == 1:
        return PROFILES["wide_explore"]
    return PROFILES["near_miss_refine"]


def validate_profile(profile: Profile) -> None:
    if not 20 <= profile.games_per_worker <= 120:
        raise RuntimeError("games_per_worker outside safety bounds")
    if not 0.70 <= profile.temperature <= 1.20:
        raise RuntimeError("temperature outside safety bounds")
    if profile.epochs != 6:
        raise RuntimeError("all PPO v1.4 profiles must keep a 6-epoch ceiling")
    if not 1e-5 <= profile.learning_rate <= 8e-5:
        raise RuntimeError("learning_rate outside safety bounds")
    if not 0.10 <= profile.clip <= 0.25:
        raise RuntimeError("clip outside safety bounds")
    if not 0.005 <= profile.target_kl <= 0.020:
        raise RuntimeError("target_kl outside safety bounds")
    if not 0.0001 <= profile.entropy <= 0.005:
        raise RuntimeError("entropy outside v1.4 safety bounds")
    if not 0.0 <= profile.anchor_coef <= 0.10:
        raise RuntimeError("anchor_coef outside safety bounds")


def adapt(
    *,
    state: dict[str, Any],
    control_order: list[str],
    control: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    profile = choose_profile(state)
    validate_profile(profile)

    updated = dict(control)
    updated.update({
        "games_per_worker": str(profile.games_per_worker),
        "temperature": f"{profile.temperature:g}",
        "epochs": str(profile.epochs),
        "learning_rate": f"{profile.learning_rate:g}",
        "clip": f"{profile.clip:.2f}",
        "target_kl": f"{profile.target_kl:g}",
        "entropy": f"{profile.entropy:g}",
        "anchor_coef": f"{profile.anchor_coef:g}",
        "reward_mode": "v14_dense",
        "ppo_version": "1.4",
        "strategy_profile": profile.name,
        "stagnation_seen": str(int(state.get("stagnation_count", 0))),
        "reason": f"adaptive_stagnation_{profile.name}",
    })

    for key in (
        "strategy_profile",
        "stagnation_seen",
        "reason",
    ):
        if key not in control_order:
            control_order.append(key)

    report = {
        "schema_version": "sts1-armg-ppo-v14-adaptation-v1",
        "round_index": int(state.get("round_index", 0)),
        "stagnation_count": int(state.get("stagnation_count", 0)),
        "last_decision": state.get("last_decision"),
        "last_gate_30_win_delta": _last_win_delta(state),
        "last_dev_mean_paired_floor_delta": _last_floor_delta(state),
        "profile": asdict(profile),
        "smart_early_stop_epoch_ceiling": 6,
        "gate_policy_changed": True,
        "gate_policy_note": "Dev Parent accumulates; strict Final 30/50 remains unchanged",
        "production_champion_changed": False,
    }
    return updated, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output-control", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state.get("schema_version") not in {
        "sts1-armg-ppo-v13-loop-state-v1",
        "sts1-armg-ppo-v14-loop-state-v1",
    }:
        raise RuntimeError("unexpected PPO loop state schema")

    order, control = parse_control(args.control)
    updated, report = adapt(
        state=state,
        control_order=order,
        control=control,
    )

    args.output_control.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_control.write_text(render_control(order, updated), encoding="utf-8")
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PPO_V13_ADAPTATION", json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
