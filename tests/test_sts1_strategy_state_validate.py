from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sts1"
    / "sts1_strategy_state_validate.py"
)
SPEC = importlib.util.spec_from_file_location("strategy_state_validate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_module)
validate_state_dir = _module.validate_state_dir


def _write_valid(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    weight = state / "current-strategy.pt"
    weight.write_bytes(b"strategy-weight")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    (state / "strategy-state.json").write_text(
        json.dumps(
            {
                "schema_version": "sts1-armg-strategy-loop-state-v1",
                "generation": 2,
                "accepted_rounds": 2,
                "rejected_rounds": 1,
                "stagnation_count": 0,
                "used_training_seeds": [1, 2],
                "used_evaluation_seeds": [3, 4],
                "current_strategy_sha256": digest,
                "base_strategy_sha256": "a" * 64,
                "combat_mcts_sims": 2000,
            }
        ),
        encoding="utf-8",
    )
    return state


def test_valid_state_passes(tmp_path: Path) -> None:
    state = _write_valid(tmp_path)
    result = validate_state_dir(state)
    assert result["result"] == "PASS_STRATEGY_STATE"
    assert result["combat_mcts_sims"] == 2000


def test_bad_weight_sha_fails(tmp_path: Path) -> None:
    state = _write_valid(tmp_path)
    (state / "current-strategy.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        validate_state_dir(state)


def test_duplicate_seed_history_fails(tmp_path: Path) -> None:
    state = _write_valid(tmp_path)
    path = state / "strategy-state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["used_evaluation_seeds"] = [7, 7]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_state_dir(state)


def test_bad_replay_schema_fails(tmp_path: Path) -> None:
    state = _write_valid(tmp_path)
    (state / "strategy-replay.jsonl").write_text(
        json.dumps({"schema_version": "wrong"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="replay schema mismatch"):
        validate_state_dir(state)


def test_training_and_evaluation_seed_overlap_fails(tmp_path: Path) -> None:
    state = _write_valid(tmp_path)
    path = state / "strategy-state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["used_training_seeds"] = [1, 2, 3]
    payload["used_evaluation_seeds"] = [3, 4, 5]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="seed history overlap"):
        validate_state_dir(state)
