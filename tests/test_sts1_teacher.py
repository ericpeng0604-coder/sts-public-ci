from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from roguelike_ai.sts1_teacher.jialeiv_armb import ArmBPolicyScores, CAND_DIM, STATE_DIM
from roguelike_ai.sts1_teacher.phase0_parity import (
    Phase0TeacherParityError,
    run_phase0_teacher_parity,
    sha256_file,
)
from roguelike_ai.sts1_teacher import (
    DecisionContext,
    HiddenInformationError,
    encode_candidate,
    encode_public_battle,
    JIALEIV_BATTLE_SEARCH_HIDDEN_INFORMATION,
    JIALEIV_BATTLE_SEARCH_ROLE,
    PUBLIC_STATE_SCHEMA,
    PairedPublicSampleEvaluator,
    PublicStateContractError,
    PublicStateSearch,
    SearchConfig,
    SimulatorCombatAdapter,
    deterministic_random_legal,
    simple_public_heuristic,
)


def _public_state() -> dict:
    return {
        "schema_version": "sts1-real-game-public-state-v1",
        "source": "real_game",
        "hp": 70,
        "max_hp": 80,
        "block": 5,
        "energy": 2,
        "hand": [
            {
                "position": 1,
                "id": "Strike_R",
                "name": "Strike",
                "type": "ATTACK",
                "cost": 1,
                "upgrades": 0,
                "is_playable": True,
                "has_target": True,
            },
            {
                "position": 2,
                "id": "Strike_R",
                "name": "Strike",
                "type": "ATTACK",
                "cost": 1,
                "upgrades": 0,
                "is_playable": True,
                "has_target": True,
            },
            {
                "position": 3,
                "id": "Defend_R",
                "name": "Defend",
                "type": "SKILL",
                "cost": 1,
                "upgrades": 0,
                "is_playable": True,
                "has_target": False,
            },
        ],
        "draw_pile": [
            {"id": "Bash", "name": "Bash", "type": "ATTACK", "cost": 2, "upgrades": 0},
            {"id": "Defend_R", "name": "Defend", "type": "SKILL", "cost": 1, "upgrades": 0},
        ],
        "discard_pile": [],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [
            {
                "index": 0,
                "id": "JawWorm",
                "name": "Jaw Worm",
                "hp": 40,
                "max_hp": 40,
                "block": 0,
                "intent": "ATTACK",
                "intent_damage": 11,
                "intent_hits": 1,
                "powers": [],
            }
        ],
        "turn": 1,
        "combat_active": True,
        "relics": [{"id": "Burning Blood", "name": "Burning Blood", "counter": -1}],
        "potions": [],
        "gold": 99,
        "floor": 1,
        "act": 1,
        "character": "IRONCLAD",
        "ascension_level": 0,
        "room": "COMBAT",
        "screen_type": "NONE",
        "screen_choices": [],
        "rewards": [],
        "map_choices": [],
        "legal_actions": [
            {"kind": "play_card", "hand_index": 1, "target_index": 0, "command": "play 1 0"},
            {"kind": "play_card", "hand_index": 2, "target_index": 0, "command": "play 2 0"},
            {"kind": "play_card", "hand_index": 3, "command": "play 3"},
            {"kind": "end_turn", "command": "end"},
        ],
        "protocol_commands": ["play", "end", "state"],
        "unmapped_commands": [],
        "decision_signature": "legacy-source-specific-signature-is-not-trusted",
    }


class _FixedPublicEvaluator:
    evaluator_id = "fixed-public-v1"
    uses_hidden_information = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def score(self, context, action, *, sample_index, config):
        self.calls.append((action.semantic_key, sample_index))
        kind = action.payload["kind"]
        if kind == "play_card":
            hand_index = action.payload["hand_index"]
            card = next(card for card in context.state["hand"] if card["position"] == hand_index)
            return 10.0 if card["type"] == "ATTACK" else 4.0
        if kind == "end_turn":
            return 0.0
        return 1.0


class _HiddenOracleEvaluator(_FixedPublicEvaluator):
    evaluator_id = "jialeiv-hidden-clone-oracle"
    uses_hidden_information = True


class _RecordingPublicBackend:
    backend_id = "recording-public-backend-v1"
    uses_hidden_information = False

    def __init__(self) -> None:
        self.samples: list[tuple[str, int, int, str]] = []

    def rollout(self, context, action, sample, *, config):
        self.samples.append((action.semantic_key, sample.sample_index, sample.sample_seed, sample.sample_key))
        return float(sample.sample_seed % 1000) + (1.0 if action.payload["kind"] == "play_card" else 0.0)


class _HiddenSampleBackend(_RecordingPublicBackend):
    backend_id = "hidden-sample-backend"
    uses_hidden_information = True


class _FakeArmBPolicy:
    policy_id = "fake-public-armb-v1"
    uses_hidden_information = False
    vocab = {"Strike": 0, "Defend": 1, "Bash": 2}

    def score_actions(self, context, *, tie_tolerance=1e-9):
        action_ids = tuple(action.action_id for action in context.legal_actions)
        scores = tuple(float(index) for index, _ in enumerate(action_ids))
        if not scores:
            return ArmBPolicyScores((), (), (), None)
        best = max(scores)
        ties = tuple(
            action_id
            for action_id, score in zip(action_ids, scores, strict=True)
            if abs(score - best) <= tie_tolerance
        )
        return ArmBPolicyScores(
            action_ids=action_ids,
            scores=scores,
            tie_action_ids=ties,
            unique_best_action_id=ties[0] if len(ties) == 1 else None,
        )


def _phase0_fixture_files(tmp_path, *, drift_checkpoint: int | None = None):
    real_rows = []
    simulator_rows = []
    for checkpoint in range(12):
        real_state = deepcopy(_public_state())
        simulator_state = deepcopy(_public_state())
        if checkpoint == drift_checkpoint:
            simulator_state["hp"] -= 1
        real_rows.append({"type": "decision_state", "sequence": checkpoint + 1, "state": real_state})
        simulator_rows.append({"type": "sim_state", "checkpoint": checkpoint, "state": simulator_state})

    real_path = tmp_path / "real.ndjson"
    simulator_path = tmp_path / "sim.ndjson"
    real_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in real_rows) + "\n",
        encoding="utf-8",
    )
    simulator_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in simulator_rows) + "\n",
        encoding="utf-8",
    )
    return real_path, simulator_path


def test_decision_context_migrates_legacy_schema_and_ignores_transport_identity() -> None:
    real = _public_state()
    simulator = deepcopy(real)
    simulator["schema_version"] = PUBLIC_STATE_SCHEMA
    simulator["source"] = "simulator"
    simulator["draw_pile"].reverse()
    simulator["protocol_commands"] = ["sim-only-command"]
    simulator["decision_signature"] = "different-source-signature"
    for action in simulator["legal_actions"]:
        action["command"] = f"sim:{action['command']}"

    real_context = DecisionContext.from_public_state(real)
    sim_context = DecisionContext.from_public_state(simulator)

    assert real_context.schema_version == PUBLIC_STATE_SCHEMA
    assert sim_context.schema_version == PUBLIC_STATE_SCHEMA
    assert real_context.source == "real_game"
    assert sim_context.source == "simulator"
    assert real_context.decision_signature == sim_context.decision_signature
    assert [a.payload for a in real_context.legal_actions] == [a.payload for a in sim_context.legal_actions]


def test_hidden_information_fails_closed_in_formal_teacher_input() -> None:
    state = _public_state()
    state["seed"] = 123456789
    with pytest.raises(PublicStateContractError, match="hidden_information_forbidden"):
        DecisionContext.from_public_state(state)

    state = _public_state()
    state["enemies"][0]["last_move_id"] = 7
    with pytest.raises(PublicStateContractError, match="hidden_information_forbidden"):
        DecisionContext.from_public_state(state)


def test_duplicate_semantic_cards_share_one_search_score_and_remain_explicit_tie() -> None:
    context = DecisionContext.from_public_state(_public_state())
    strikes = [a for a in context.legal_actions if a.payload.get("hand_index") in {1, 2}]
    assert len(strikes) == 2
    assert strikes[0].action_id != strikes[1].action_id
    assert strikes[0].semantic_key == strikes[1].semantic_key

    evaluator = _FixedPublicEvaluator()
    search = PublicStateSearch(
        evaluator,
        SearchConfig(samples_per_semantic_action=2, rollout_budget=32, timeout_ms=5_000),
    )
    result = search.run(context)

    strike_scores = [item for item in result.candidate_scores if item.action_id in {a.action_id for a in strikes}]
    assert {item.score for item in strike_scores} == {10.0}
    assert {item.samples for item in strike_scores} == {2}
    assert set(result.tie_action_ids) == {a.action_id for a in strikes}
    assert result.unique_best_action_id is None
    assert result.unresolved_action_ids == ()
    assert result.resolved is True

    # Four executable actions become three semantic groups: Strike, Defend, End.
    assert len(evaluator.calls) == 3 * 2


def test_public_search_is_reproducible_and_evidence_hash_is_stable() -> None:
    context = DecisionContext.from_public_state(_public_state())
    config = SearchConfig(samples_per_semantic_action=3, rollout_budget=64, sampling_seed=17)

    first = PublicStateSearch(_FixedPublicEvaluator(), config).run(context)
    second = PublicStateSearch(_FixedPublicEvaluator(), config).run(context)

    assert first == second
    assert first.config_hash == config.config_hash
    assert len(first.evidence_hash) == 64


def test_jialeiv_armb_encoder_reproduces_public_inference_shape_without_slot_drift() -> None:
    context = DecisionContext.from_public_state(_public_state())
    vocab = {"Strike": 0, "Defend": 1, "Bash": 2}

    state_vector = encode_public_battle(context, vocab)
    assert len(state_vector) == STATE_DIM

    strikes = [action for action in context.legal_actions if action.payload.get("hand_index") in {1, 2}]
    assert len(strikes) == 2
    first = encode_candidate(context, strikes[0], vocab)
    second = encode_candidate(context, strikes[1], vocab)
    assert len(first) == CAND_DIM
    assert first == second

    reversed_state = _public_state()
    reversed_state["draw_pile"].reverse()
    reversed_context = DecisionContext.from_public_state(reversed_state)
    assert encode_public_battle(reversed_context, vocab) == state_vector


def test_paired_public_sampling_uses_same_future_sample_for_every_candidate() -> None:
    context = DecisionContext.from_public_state(_public_state())
    backend = _RecordingPublicBackend()
    config = SearchConfig(samples_per_semantic_action=3, rollout_budget=64, sampling_seed=123)
    result = PublicStateSearch(PairedPublicSampleEvaluator(backend), config).run(context)

    by_index: dict[int, set[tuple[int, str]]] = {}
    for _, sample_index, sample_seed, sample_key in backend.samples:
        by_index.setdefault(sample_index, set()).add((sample_seed, sample_key))
    assert set(by_index) == {0, 1, 2}
    assert all(len(values) == 1 for values in by_index.values())
    assert result.unresolved_action_ids == ()

    repeat_backend = _RecordingPublicBackend()
    repeat = PublicStateSearch(PairedPublicSampleEvaluator(repeat_backend), config).run(context)
    assert backend.samples == repeat_backend.samples
    assert result == repeat

    with pytest.raises(ValueError, match="paired_public_sampler_rejects_hidden_backend"):
        PairedPublicSampleEvaluator(_HiddenSampleBackend())


def test_formal_search_rejects_hidden_rng_oracle() -> None:
    context = DecisionContext.from_public_state(_public_state())
    with pytest.raises(HiddenInformationError, match="formal_teacher_rejects_hidden_evaluator"):
        PublicStateSearch(_HiddenOracleEvaluator()).run(context)

    assert JIALEIV_BATTLE_SEARCH_ROLE == "oracle_reference"
    assert JIALEIV_BATTLE_SEARCH_HIDDEN_INFORMATION is True


def test_simulator_combat_adapter_projects_only_public_fields() -> None:
    strike = SimpleNamespace(
        id="STRIKE_RED",
        name="Strike",
        type="ATTACK",
        cost=1,
        cost_for_turn=1,
        upgraded=False,
        requires_target=True,
    )
    defend = SimpleNamespace(
        id="DEFEND_RED",
        name="Defend",
        type="SKILL",
        cost=1,
        cost_for_turn=1,
        upgraded=False,
        requires_target=False,
    )
    bash = SimpleNamespace(
        id="BASH",
        name="Bash",
        type="ATTACK",
        cost=2,
        cost_for_turn=2,
        upgraded=False,
        requires_target=True,
    )
    enemy = SimpleNamespace(
        name="Jaw Worm",
        cur_hp=40,
        max_hp=40,
        block=0,
        intent="ATTACK",
        intent_damage=lambda battle: SimpleNamespace(damage=11, attack_count=1),
        alive=True,
        strength=0,
        vulnerable=0,
        weak=0,
        poison=0,
        move_id=999,
        last_move_id=998,
    )
    battle = SimpleNamespace(
        player=SimpleNamespace(
            cur_hp=70,
            max_hp=80,
            block=5,
            energy=2,
            strength=1,
            dexterity=0,
            focus=0,
            artifact=0,
        ),
        hand=[strike, defend],
        draw_pile=[bash, strike],
        discard_pile=[],
        exhaust_pile=[],
        monsters=[enemy],
        turn=1,
        outcome="UNDECIDED",
        rng_state="must-never-be-read",
    )
    actions = [
        SimpleNamespace(action_type="CARD", source_idx=0, target_idx=0),
        SimpleNamespace(action_type="CARD", source_idx=1, target_idx=0),
        SimpleNamespace(action_type="END_TURN", source_idx=-1, target_idx=-1),
    ]

    adapted = SimulatorCombatAdapter().adapt(
        battle,
        legal_actions=actions,
        run_state={
            "gold": 99,
            "floor": 1,
            "act": 1,
            "character": "IRONCLAD",
            "ascension_level": 0,
            "relics": [{"id": "Burning Blood", "name": "Burning Blood", "counter": -1}],
        },
    )

    assert adapted["schema_version"] == PUBLIC_STATE_SCHEMA
    assert adapted["source"] == "simulator"
    assert adapted["hp"] == 70
    assert adapted["energy"] == 2
    assert adapted["enemies"][0]["intent"] == "ATTACK"
    assert adapted["enemies"][0]["intent_damage"] == 11
    assert adapted["enemies"][0]["intent_hits"] == 1
    assert "move_id" not in adapted["enemies"][0]
    assert "last_move_id" not in adapted["enemies"][0]
    assert "rng_state" not in adapted
    assert len(adapted["decision_signature"]) == 64
    assert {a["kind"] for a in adapted["legal_actions"]} == {"play_card", "end_turn"}
    assert DecisionContext.from_public_state(adapted).decision_signature == adapted["decision_signature"]

    reversed_battle = deepcopy(battle)
    reversed_battle.draw_pile.reverse()
    reversed_state = SimulatorCombatAdapter().adapt(
        reversed_battle,
        legal_actions=actions,
        run_state={"gold": 99, "floor": 1, "act": 1, "character": "IRONCLAD", "ascension_level": 0,
                   "relics": [{"id": "Burning Blood", "name": "Burning Blood", "counter": -1}]},
    )
    assert reversed_state["decision_signature"] == adapted["decision_signature"]


def test_random_and_simple_baselines_only_return_legal_actions() -> None:
    context = DecisionContext.from_public_state(_public_state())

    first_random = deterministic_random_legal(context, benchmark_seed=42)
    second_random = deterministic_random_legal(context, benchmark_seed=42)
    heuristic = simple_public_heuristic(context)

    assert first_random is not None
    assert second_random is not None
    assert first_random.action_id == second_random.action_id
    assert first_random in context.legal_actions
    assert heuristic in context.legal_actions
    assert heuristic is not None
    assert heuristic.payload["kind"] == "play_card"
    chosen_card = next(
        card for card in context.state["hand"] if card["position"] == heuristic.payload["hand_index"]
    )
    assert chosen_card["type"] == "ATTACK"


def test_phase0_teacher_parity_runner_passes_all_frozen_checkpoints_on_equal_public_states(tmp_path) -> None:
    real_path, simulator_path = _phase0_fixture_files(tmp_path)
    report = run_phase0_teacher_parity(
        real_path,
        simulator_path,
        _FakeArmBPolicy(),
        expected_real_sha256=sha256_file(real_path),
        expected_simulator_sha256=sha256_file(simulator_path),
    )

    assert report.result == "PASS"
    assert report.matched_checkpoints == 11
    assert report.requested_checkpoints == tuple(range(1, 12))
    assert all(item.passed for item in report.checkpoints)
    assert report.hidden_information is False
    assert len(report.evidence_hash) == 64


def test_phase0_teacher_parity_runner_detects_public_encoder_drift(tmp_path) -> None:
    real_path, simulator_path = _phase0_fixture_files(tmp_path, drift_checkpoint=5)
    report = run_phase0_teacher_parity(
        real_path,
        simulator_path,
        _FakeArmBPolicy(),
        expected_real_sha256=sha256_file(real_path),
        expected_simulator_sha256=sha256_file(simulator_path),
    )

    assert report.result == "FAIL"
    assert report.matched_checkpoints == 10
    drift = next(item for item in report.checkpoints if item.checkpoint == 5)
    assert drift.encoder_equal is False
    assert drift.passed is False


def test_phase0_teacher_parity_runner_rejects_wrong_frozen_evidence_hash(tmp_path) -> None:
    real_path, simulator_path = _phase0_fixture_files(tmp_path)
    with pytest.raises(Phase0TeacherParityError, match="real_trajectory_sha256_mismatch"):
        run_phase0_teacher_parity(
            real_path,
            simulator_path,
            _FakeArmBPolicy(),
            expected_real_sha256="0" * 64,
            expected_simulator_sha256=sha256_file(simulator_path),
        )
