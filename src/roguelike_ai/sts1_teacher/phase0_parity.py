"""Reproducible Phase 0 Real↔Simulator parity gate for the public Arm B Teacher.

The Phase 0 live-validation trajectories are immutable local evidence.  This
module verifies their exact SHA-256 identities, skips checkpoint 0 because its
transient real-game DEBUG intent was already classified as not observable, and
compares checkpoints 1..11 through the *actual* public Arm B inference path.

No simulator RNG, seed, hidden move history, or future information enters this
comparison.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .contract import DecisionContext, PUBLIC_STATE_SCHEMA, sha256_json
from .jialeiv_armb import ArmBPolicyScores, JialeivArmBPolicy, encode_public_battle

PHASE0_REAL_TRAJECTORY_SHA256 = "1e3c4c6b1f5a2d5026237297f88ed745b678309cf4af6f711f0f21174c034e2c"
PHASE0_SIM_TRAJECTORY_SHA256 = "d67fbf425e54d3d1a18b1ef18a76d7af37d821b7356f1d9abc42e9722541c91e"
PHASE0_PARITY_CHECKPOINTS = tuple(range(1, 12))


class Phase0TeacherParityError(RuntimeError):
    """Raised when frozen evidence identity or the parity contract is invalid."""


class ArmBPolicyLike(Protocol):
    policy_id: str
    uses_hidden_information: bool
    vocab: Mapping[str, int]

    def score_actions(
        self,
        context: DecisionContext,
        *,
        tie_tolerance: float = 1e-9,
    ) -> ArmBPolicyScores:
        """Score the public legal actions for one decision."""


@dataclass(frozen=True)
class Phase0TeacherCheckpoint:
    checkpoint: int
    real_decision_signature: str
    simulator_decision_signature: str
    encoder_equal: bool
    action_ids_equal: bool
    scores_equal: bool
    ties_equal: bool
    unique_best_equal: bool
    real_deterministic: bool
    simulator_deterministic: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.encoder_equal,
                self.action_ids_equal,
                self.scores_equal,
                self.ties_equal,
                self.unique_best_equal,
                self.real_deterministic,
                self.simulator_deterministic,
            )
        )


@dataclass(frozen=True)
class Phase0TeacherParityReport:
    result: str
    teacher_id: str
    hidden_information: bool
    real_sha256: str
    simulator_sha256: str
    requested_checkpoints: tuple[int, ...]
    matched_checkpoints: int
    checkpoints: tuple[Phase0TeacherCheckpoint, ...]
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "teacher_id": self.teacher_id,
            "hidden_information": self.hidden_information,
            "real_sha256": self.real_sha256,
            "simulator_sha256": self.simulator_sha256,
            "requested_checkpoints": list(self.requested_checkpoints),
            "matched_checkpoints": self.matched_checkpoints,
            "checkpoints": [
                {**asdict(item), "passed": item.passed}
                for item in self.checkpoints
            ],
            "evidence_hash": self.evidence_hash,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_combat_states(path: Path, *, record_type: str) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise Phase0TeacherParityError(f"invalid_ndjson:{path}:{line_number}") from exc
        if not isinstance(record, Mapping) or record.get("type") != record_type:
            continue
        state = record.get("state")
        if not isinstance(state, Mapping) or not bool(state.get("combat_active")):
            continue
        states.append(deepcopy(dict(state)))
    return states


def _simulator_public_state(state: Mapping[str, Any]) -> dict[str, Any]:
    projected = deepcopy(dict(state))
    projected["schema_version"] = PUBLIC_STATE_SCHEMA
    projected["source"] = "simulator"
    return projected


def _score_twice(
    policy: ArmBPolicyLike,
    context: DecisionContext,
    *,
    tie_tolerance: float,
) -> tuple[tuple[float, ...], ArmBPolicyScores, ArmBPolicyScores]:
    encoded = tuple(float(value) for value in encode_public_battle(context, policy.vocab))
    first = policy.score_actions(context, tie_tolerance=tie_tolerance)
    second = policy.score_actions(context, tie_tolerance=tie_tolerance)
    return encoded, first, second


def run_phase0_teacher_parity(
    real_path: str | Path,
    simulator_path: str | Path,
    policy: ArmBPolicyLike,
    *,
    checkpoints: Sequence[int] = PHASE0_PARITY_CHECKPOINTS,
    expected_real_sha256: str = PHASE0_REAL_TRAJECTORY_SHA256,
    expected_simulator_sha256: str = PHASE0_SIM_TRAJECTORY_SHA256,
    tie_tolerance: float = 1e-9,
) -> Phase0TeacherParityReport:
    """Run the frozen Phase 0 Arm B Real↔Simulator parity gate.

    Exact score equality is intentional.  Both sides feed the same CPU model
    process and the gate is specifically proving source-projection identity.
    """

    if bool(getattr(policy, "uses_hidden_information", True)):
        raise Phase0TeacherParityError("formal_parity_rejects_hidden_teacher")

    real_file = Path(real_path)
    simulator_file = Path(simulator_path)
    real_sha = sha256_file(real_file)
    simulator_sha = sha256_file(simulator_file)
    if real_sha != expected_real_sha256:
        raise Phase0TeacherParityError(
            f"real_trajectory_sha256_mismatch:{real_sha}:{expected_real_sha256}"
        )
    if simulator_sha != expected_simulator_sha256:
        raise Phase0TeacherParityError(
            f"simulator_trajectory_sha256_mismatch:{simulator_sha}:{expected_simulator_sha256}"
        )

    requested = tuple(int(value) for value in checkpoints)
    if not requested or any(value < 0 for value in requested):
        raise Phase0TeacherParityError("invalid_checkpoint_set")

    real_states = _load_combat_states(real_file, record_type="decision_state")
    simulator_states = _load_combat_states(simulator_file, record_type="sim_state")
    required_index = max(requested)
    if len(real_states) <= required_index:
        raise Phase0TeacherParityError(
            f"real_trajectory_missing_checkpoint:{required_index}:{len(real_states)}"
        )
    if len(simulator_states) <= required_index:
        raise Phase0TeacherParityError(
            f"simulator_trajectory_missing_checkpoint:{required_index}:{len(simulator_states)}"
        )

    results: list[Phase0TeacherCheckpoint] = []
    for checkpoint in requested:
        real_context = DecisionContext.from_public_state(real_states[checkpoint])
        simulator_context = DecisionContext.from_public_state(
            _simulator_public_state(simulator_states[checkpoint])
        )

        real_encoder, real_first, real_second = _score_twice(
            policy,
            real_context,
            tie_tolerance=tie_tolerance,
        )
        sim_encoder, sim_first, sim_second = _score_twice(
            policy,
            simulator_context,
            tie_tolerance=tie_tolerance,
        )

        results.append(
            Phase0TeacherCheckpoint(
                checkpoint=checkpoint,
                real_decision_signature=real_context.decision_signature,
                simulator_decision_signature=simulator_context.decision_signature,
                encoder_equal=real_encoder == sim_encoder,
                action_ids_equal=real_first.action_ids == sim_first.action_ids,
                scores_equal=real_first.scores == sim_first.scores,
                ties_equal=real_first.tie_action_ids == sim_first.tie_action_ids,
                unique_best_equal=(
                    real_first.unique_best_action_id == sim_first.unique_best_action_id
                ),
                real_deterministic=real_first == real_second,
                simulator_deterministic=sim_first == sim_second,
            )
        )

    matched = sum(item.passed for item in results)
    result = "PASS" if matched == len(results) else "FAIL"
    teacher_id = str(getattr(policy, "policy_id", "unknown"))
    evidence_payload = {
        "result": result,
        "teacher_id": teacher_id,
        "hidden_information": False,
        "real_sha256": real_sha,
        "simulator_sha256": simulator_sha,
        "requested_checkpoints": list(requested),
        "matched_checkpoints": matched,
        "checkpoints": [
            {**asdict(item), "passed": item.passed}
            for item in results
        ],
    }
    return Phase0TeacherParityReport(
        result=result,
        teacher_id=teacher_id,
        hidden_information=False,
        real_sha256=real_sha,
        simulator_sha256=simulator_sha,
        requested_checkpoints=requested,
        matched_checkpoints=matched,
        checkpoints=tuple(results),
        evidence_hash=sha256_json(evidence_payload),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STS1 Phase 0 Arm B public-state parity gate")
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--simulator", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--weight", required=True, type=Path)
    args = parser.parse_args(argv)

    policy = JialeivArmBPolicy.from_files(vocab_path=args.vocab, weight_path=args.weight)
    report = run_phase0_teacher_parity(args.real, args.simulator, policy)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PHASE0_PARITY_CHECKPOINTS",
    "PHASE0_REAL_TRAJECTORY_SHA256",
    "PHASE0_SIM_TRAJECTORY_SHA256",
    "Phase0TeacherCheckpoint",
    "Phase0TeacherParityError",
    "Phase0TeacherParityReport",
    "run_phase0_teacher_parity",
    "sha256_file",
]
