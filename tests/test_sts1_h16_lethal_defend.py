from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sts1"))

import sts1_g7_h16_lethal_defend_eval as h16_runner
from roguelike_ai.sts1_phase3.simulator import _apply_lethal_intent_defend_rescue


class _Card:
    def __init__(self, card_id: str, upgraded: bool | None = False) -> None:
        self.id = card_id
        self.upgraded = upgraded


class _Action:
    action_type = "CARD"

    def __init__(self, source_idx: int) -> None:
        self.source_idx = source_idx
        self.target_idx = -1


class _Enemy:
    cur_hp = 20
    alive = True

    def __init__(self, damage: int | None, hits: int = 1) -> None:
        self.damage = damage
        self.hits = hits

    def intent_damage(self, _battle: object) -> object:
        if self.damage is None:
            raise RuntimeError("unknown intent")
        return SimpleNamespace(damage=self.damage, attack_count=self.hits)


def _battle(*, hp: int = 8, block: int = 2, damage: int | None = 16) -> object:
    return SimpleNamespace(
        player=SimpleNamespace(cur_hp=hp, block=block),
        monsters=[_Enemy(damage)],
    )


def test_h16_selects_strongest_legal_defend_that_closes_visible_deficit() -> None:
    hand = [_Card("DEFEND_RED"), _Card("DEFEND_RED", upgraded=True), _Card("BASH")]
    recommended = _Action(2)
    selected, overridden, reason, detail = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1), _Action(2)],
        hand=hand,
        battle=_battle(hp=8, block=2, damage=16),
    )

    assert overridden is True
    assert reason == "legal_defend_closes_visible_lethal_deficit"
    assert selected is not recommended
    assert selected.source_idx == 1
    assert detail["projected_deficit"] == 6
    assert detail["selected_defend_base_block"] == 8
    assert detail["selected_defend_hand_index"] == 2


def test_h16_recognizes_native_simulator_defend_red_id() -> None:
    hand = [_Card("DEFEND_RED", upgraded=True), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, detail = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(),
    )

    assert overridden is True
    assert reason == "legal_defend_closes_visible_lethal_deficit"
    assert selected.source_idx == 0
    assert detail["selected_defend_hand_index"] == 1


def test_h16_tie_breaks_by_lowest_hand_index_not_legal_action_order() -> None:
    hand = [_Card("DEFEND_RED", upgraded=True), _Card("DEFEND_RED", upgraded=True), _Card("BASH")]
    recommended = _Action(2)
    selected, overridden, _, detail = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(1), _Action(0), _Action(2)],
        hand=hand,
        battle=_battle(),
    )

    assert overridden is True
    assert selected.source_idx == 0
    assert detail["selected_defend_hand_index"] == 1


def test_h16_keeps_g7_action_when_intent_is_not_lethal() -> None:
    hand = [_Card("DEFEND_RED"), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(hp=8, block=2, damage=9),
    )

    assert selected is recommended
    assert overridden is False
    assert reason == "visible_intent_not_lethal"


def test_h16_keeps_g7_defend_recommendation() -> None:
    hand = [_Card("DEFEND_RED"), _Card("BASH")]
    recommended = _Action(0)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(),
    )

    assert selected is recommended
    assert overridden is False
    assert reason == "g7_already_selected_defend"


def test_h16_fails_closed_when_defend_cannot_close_deficit_or_identity_is_unknown() -> None:
    hand = [_Card("DEFEND_RED"), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(hp=8, block=0, damage=20),
    )
    assert selected is recommended
    assert overridden is False
    assert reason == "defend_block_does_not_close_deficit"

    unknown_upgrade = [_Card("DEFEND_RED", upgraded=None), _Card("BASH")]
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=unknown_upgrade,
        battle=_battle(),
    )
    assert selected is recommended
    assert overridden is False
    assert reason == "defend_upgrade_status_unknown"


def test_h16_fails_closed_when_visible_intent_is_unavailable() -> None:
    hand = [_Card("DEFEND_RED"), _Card("BASH")]
    recommended = _Action(1)
    selected, overridden, reason, _ = _apply_lethal_intent_defend_rescue(
        recommended,
        [_Action(0), _Action(1)],
        hand=hand,
        battle=_battle(damage=None),
    )

    assert selected is recommended
    assert overridden is False
    assert reason == "unknown_attack_intent"


def test_h17_uses_registered_hypothesis_three_pool_and_manifest() -> None:
    assert h16_runner.TRIAL_POOL_ROLE["h16"]["train"] == "train_hypothesis_2"
    assert h16_runner.TRIAL_POOL_ROLE["h17"]["train"] == "train_hypothesis_3"
    assert h16_runner._without_private_hashes(
        {
            "manifest_sha256": "private",
            "nested": {"sha256": "private", "count": 10},
        }
    ) == {"nested": {"count": 10}}


def test_h17_private_paths_are_separate_from_h16_and_repo(tmp_path) -> None:
    pools_dir = tmp_path / "round-008-seeds" / "pools"
    h17_output = tmp_path / "round-008-h17-train-20261009"
    h17_usage = pools_dir.parent / "h17-usage-private.jsonl"

    h16_runner._validate_private_paths(pools_dir, "h17", "train", h17_output, h17_usage)

    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_private_paths(
            pools_dir,
            "h17",
            "train",
            tmp_path / "round-008-h16-train-20261009",
            h17_usage,
        )


def test_h17_train_transition_requires_exact_pool_allocation_history() -> None:
    allocation = deepcopy(h16_runner.EXPECTED_H17_ALLOCATION)
    candidate_commit = "a" * 40
    identities = {key: f"{key}-sha256" for key in (
        "simulator_policy_source_sha256",
        "candidate_evaluator_sha256",
        "simulator_binding_sha256",
        "armg_source_sha256",
        "armg_vocab_sha256",
        "g7_checkpoint_sha256",
    )}

    h16_runner._validate_transition(
        trial_id="h17",
        stage="train",
        events=[allocation],
        candidate_commit=candidate_commit,
        identities=identities,
    )

    altered = deepcopy(allocation)
    altered["prior_h16"]["probe"]["status"] = "COMPLETE"
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_transition(
            trial_id="h17",
            stage="train",
            events=[altered],
            candidate_commit=candidate_commit,
            identities=identities,
        )


def test_h17_probe_requires_its_own_eligible_train_summary() -> None:
    candidate_commit = "b" * 40
    identities = {key: f"{key}-sha256" for key in (
        "simulator_policy_source_sha256",
        "candidate_evaluator_sha256",
        "simulator_binding_sha256",
        "armg_source_sha256",
        "armg_vocab_sha256",
        "g7_checkpoint_sha256",
    )}
    summary = {
        "record_type": "h17_stage_summary",
        "trial_id": "h17",
        "stage": "train",
        "status": "COMPLETE",
        "candidate_commit": candidate_commit,
        "advance_eligible": True,
        **identities,
    }

    h16_runner._validate_transition(
        trial_id="h17",
        stage="probe",
        events=[deepcopy(h16_runner.EXPECTED_H17_ALLOCATION), summary],
        candidate_commit=candidate_commit,
        identities=identities,
    )

    wrong_trial = {**summary, "trial_id": "h16"}
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_transition(
            trial_id="h17",
            stage="probe",
            events=[deepcopy(h16_runner.EXPECTED_H17_ALLOCATION), wrong_trial],
            candidate_commit=candidate_commit,
            identities=identities,
        )


def test_h16_legacy_stage_summary_remains_usable_without_trial_id() -> None:
    candidate_commit = "c" * 40
    identities = {key: f"{key}-sha256" for key in (
        "simulator_policy_source_sha256",
        "candidate_evaluator_sha256",
        "simulator_binding_sha256",
        "armg_source_sha256",
        "armg_vocab_sha256",
        "g7_checkpoint_sha256",
    )}
    legacy_summary = {
        "record_type": "h16_stage_summary",
        "stage": "train",
        "status": "COMPLETE",
        "candidate_commit": candidate_commit,
        "advance_eligible": True,
        **identities,
    }

    assert h16_runner._stage_summary([legacy_summary], "train", "h16") == legacy_summary
    h16_runner._validate_transition(
        trial_id="h16",
        stage="probe",
        events=[legacy_summary],
        candidate_commit=candidate_commit,
        identities=identities,
    )


def test_stage_configuration_sets_paired_pool_sizes() -> None:
    assert h16_runner.STAGE_CONFIG["train"]["paired_seed_count"] == 10
    assert h16_runner.STAGE_CONFIG["probe"]["paired_seed_count"] == 10
    assert h16_runner.STAGE_CONFIG["dev"]["paired_seed_count"] == 30
    assert h16_runner.STAGE_CONFIG["confirmation_a"]["paired_seed_count"] == 100
    assert h16_runner.STAGE_CONFIG["confirmation_b"]["paired_seed_count"] == 100


def test_pinned_upstream_manifest_is_verified_without_project_git_ancestry(tmp_path: Path) -> None:
    manifest = tmp_path / "UPSTREAM.json"
    manifest.write_text(
        json.dumps({
            "repository": h16_runner.PINNED_GAMEPLAY_REPOSITORY_URL,
            "commit": h16_runner.PINNED_GAMEPLAY_COMMIT,
        }),
        encoding="utf-8",
    )
    assert h16_runner._validate_pinned_gameplay_manifest(manifest) == h16_runner.PINNED_GAMEPLAY_COMMIT

    manifest.write_text(
        json.dumps({
            "repository": h16_runner.PINNED_GAMEPLAY_REPOSITORY_URL,
            "commit": "0" * 40,
        }),
        encoding="utf-8",
    )
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_pinned_gameplay_manifest(manifest)


def test_stage_seed_roles_keep_probe_and_dev_out_of_training() -> None:
    seeds = tuple(range(10))
    assert h16_runner._stage_seed_roles("train", seeds) == (seeds, None)
    assert h16_runner._stage_seed_roles("probe", seeds) == (None, seeds)
    dev_seeds = tuple(range(30))
    assert h16_runner._stage_seed_roles("dev", dev_seeds) == (None, dev_seeds)
    confirmation_seeds = tuple(range(100))
    assert h16_runner._stage_seed_roles("confirmation_a", confirmation_seeds) == (
        None,
        confirmation_seeds,
    )
    assert h16_runner._stage_seed_roles("confirmation_b", confirmation_seeds) == (
        None,
        confirmation_seeds,
    )


def test_shared_paired_summary_uses_exact_one_sided_sign_test() -> None:
    tied = h16_runner.h3._paired_summary_for_stage(["defeat"] * 10, ["defeat"] * 10)
    assert tied["discordant_pairs"] == 0
    assert tied["exact_one_sided_sign_p_candidate_positive"] == 1.0

    candidate_positive = h16_runner.h3._paired_summary_for_stage(
        ["defeat"] * 10,
        ["victory"] * 3 + ["defeat"] * 7,
    )
    assert candidate_positive["candidate_only_wins"] == 3
    assert candidate_positive["parent_only_wins"] == 0
    assert candidate_positive["exact_one_sided_sign_p_candidate_positive"] == pytest.approx(0.125)


def test_train_probe_gates_use_distinct_effective_override_seed_coverage() -> None:
    nonnegative_train = {"net_wins": 0}
    assert h16_runner._stage_gate("train", nonnegative_train, 2, True)["advance_eligible"] is False
    assert h16_runner._stage_gate("train", nonnegative_train, 3, True)["advance_eligible"] is True
    assert h16_runner._stage_gate("probe", nonnegative_train, 3, False)["advance_eligible"] is False


def test_dev_positive_net_is_candidate_selection_signal_without_p_threshold() -> None:
    assert h16_runner._stage_gate(
        "dev",
        {"net_wins": 4, "exact_one_sided_sign_p_candidate_positive": 0.0625},
        0,
        True,
    )["advance_eligible"] is True
    assert h16_runner._stage_gate(
        "dev",
        {"net_wins": 0, "exact_one_sided_sign_p_candidate_positive": 1.0},
        0,
        True,
    )["advance_eligible"] is False
    assert h16_runner._stage_gate(
        "dev", {"net_wins": -1}, 0, True
    )["advance_eligible"] is False


def test_dev_hard_safety_or_integrity_failure_still_blocks_confirmation() -> None:
    assert h16_runner._stage_gate(
        "dev",
        {"net_wins": 4, "exact_one_sided_sign_p_candidate_positive": 0.0625},
        0,
        False,
    )["advance_eligible"] is False


def test_floor_hp_regressions_are_reported_but_do_not_block_exploration() -> None:
    diagnostic = h16_runner._terminal_floor_hp_diagnostic([
        {
            "parent_outcome": "defeat", "candidate_outcome": "defeat",
            "parent_floor": 10, "candidate_floor": 9, "parent_hp": 5, "candidate_hp": 3,
        },
        {
            "parent_outcome": "victory", "candidate_outcome": "victory",
            "parent_floor": 50, "candidate_floor": 50, "parent_hp": 80, "candidate_hp": 70,
        },
        {
            "parent_outcome": "defeat", "candidate_outcome": "victory",
            "parent_floor": 30, "candidate_floor": 50, "parent_hp": 0, "candidate_hp": 12,
        },
        {
            "parent_outcome": "victory", "candidate_outcome": "defeat",
            "parent_floor": 50, "candidate_floor": 20, "parent_hp": 4, "candidate_hp": 0,
        },
    ])

    assert diagnostic["blocking"] is False
    assert diagnostic["candidate_relation_counts"]["terminal_floor"] == {
        "equal": 1, "higher": 1, "lower": 2,
    }
    assert diagnostic["candidate_relation_counts"]["final_hp"]["lower"] == 3
    assert diagnostic["by_paired_outcome"]["both_defeat"]["pair_count"] == 1
    assert diagnostic["by_paired_outcome"]["both_victory"]["pair_count"] == 1
    assert diagnostic["by_paired_outcome"]["candidate_only"]["pair_count"] == 1
    assert diagnostic["by_paired_outcome"]["parent_only"]["pair_count"] == 1
    assert h16_runner._stage_gate("train", {"net_wins": 0}, 3, True)["advance_eligible"] is True


def _synthetic_confirmation_batch(stage: str, candidate_only: int, parent_only: int) -> dict[str, object]:
    pairs = []
    for index in range(100):
        if index < candidate_only:
            parent_outcome, candidate_outcome = "defeat", "victory"
        elif index < candidate_only + parent_only:
            parent_outcome, candidate_outcome = "victory", "defeat"
        else:
            parent_outcome = candidate_outcome = "defeat"
        pairs.append({
            "parent": {"outcome": parent_outcome},
            "candidate": {"outcome": candidate_outcome},
        })
    return {
        "stage": stage,
        "status": "COMPLETE",
        "seed_count": 100,
        "hard_guards_passed": True,
        "seed_disjointness_verified": True,
        "pool_manifest_sha256": "a" * 64 if stage == "confirmation_a" else "b" * 64,
        "candidate_commit": "c" * 40,
        "candidate_evaluator_sha256": "d" * 64,
        "simulator_policy_source_sha256": "e" * 64,
        "simulator_binding_sha256": "f" * 64,
        "simulator_gameplay_commit": "7476a81954020087da31d41d16fddf475746ec2d",
        "armg_source_sha256": "1" * 64,
        "armg_vocab_sha256": "2" * 64,
        "g7_checkpoint_sha256": "3" * 64,
        "pairs": pairs,
    }


def test_confirmation_gate_preserves_two_batch_net_and_trial_alpha() -> None:
    batch_a = _synthetic_confirmation_batch("confirmation_a", candidate_only=6, parent_only=0)
    batch_b = _synthetic_confirmation_batch("confirmation_b", candidate_only=6, parent_only=0)

    first = h16_runner._confirmation_gate(
        batch_a, batch_b, trial_k=1, prior_trial_ks=[]
    )
    second = h16_runner._confirmation_gate(
        batch_a, batch_b, trial_k=2, prior_trial_ks=[1]
    )

    assert first["alpha_k"] == pytest.approx(0.025)
    assert second["alpha_k"] == pytest.approx(0.0125)
    assert first["batch_a_net"] == first["batch_b_net"] == 6
    assert first["combined"]["net_wins"] == 12
    assert first["combined"]["candidate_only_wins"] == 12
    assert first["combined"]["parent_only_wins"] == 0
    assert first["accepted"] is True
    assert second["accepted"] is True
    with pytest.raises(h16_runner.EvaluationIntegrityError, match="reset"):
        h16_runner._confirmation_gate(
            batch_a, batch_b, trial_k=1, prior_trial_ks=[1]
        )


def test_confirmation_gate_rejects_nonpositive_batch_or_less_than_ten_net() -> None:
    batch_a = _synthetic_confirmation_batch("confirmation_a", candidate_only=4, parent_only=0)
    batch_b = _synthetic_confirmation_batch("confirmation_b", candidate_only=4, parent_only=0)
    below_minimum = h16_runner._confirmation_gate(
        batch_a, batch_b, trial_k=1, prior_trial_ks=[]
    )
    assert below_minimum["combined"]["net_wins"] == 8
    assert below_minimum["sign_test_passed"] is True
    assert below_minimum["accepted"] is False

    nonpositive_batch_b = _synthetic_confirmation_batch(
        "confirmation_b", candidate_only=4, parent_only=4
    )
    rejected = h16_runner._confirmation_gate(
        batch_a, nonpositive_batch_b, trial_k=1, prior_trial_ks=[]
    )
    assert rejected["batch_b_net"] == 0
    assert rejected["both_batches_positive_net"] is False
    assert rejected["accepted"] is False


def test_selected_pool_requires_exact_count_and_unique_integer_ids() -> None:
    h16_runner._validate_selected_seeds("train", tuple(range(10)))
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_selected_seeds("train", tuple(range(9)))
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_selected_seeds("train", (1, 2, 3, 4, 5, 6, 7, 8, 9, 9))
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_selected_seeds("train", (1, 2, 3, 4, 5, 6, 7, 8, 9, True))


@pytest.mark.parametrize(
    "missing",
    ("illegal_action_count", "timeout_count", "crash_count", "communication_error_count"),
)
def test_pair_integrity_fails_closed_when_canonical_counter_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    monkeypatch.setattr(h16_runner.h2, "_check_evidence", lambda *_args, **_kwargs: None)
    result = {
        "outcome": "defeat",
        "error": None,
        "illegal_action_count": 0,
        "timeout_count": 0,
        "crash_count": 0,
        "communication_error_count": 0,
        "potion_inventory_snapshot_complete": True,
    }
    del result[missing]

    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._check_pair_integrity(tmp_path / "evidence.ndjson", result)


def _decision(
    *, selected: dict[str, object], recommended: dict[str, object], before: str, override: bool
) -> dict[str, object]:
    return {
        "type": "combat_decision_trace_v1",
        "encounter_index": 1,
        "battle_step": 3,
        "state_before_signature_sha256": before,
        "selected_action": selected,
        "mcts_recommended_action": recommended,
        "lethal_intent_defend_override": override,
    }


def _applied(*, selected: dict[str, object], after: str) -> dict[str, object]:
    return {
        "type": "combat_action_applied_v1",
        "encounter_index": 1,
        "battle_step": 3,
        "selected_action": selected,
        "state_after_signature_sha256": after,
    }


def test_override_coverage_requires_matching_prefix_and_changed_afterstate() -> None:
    recommended = {"kind": "end_turn"}
    parent = {
        "decisions": [_decision(selected=recommended, recommended=recommended, before="a" * 64, override=False)],
        "applied": [_applied(selected=recommended, after="b" * 64)],
    }
    defend = {"kind": "play_card", "card": "DEFEND_RED", "hand_index": 1}
    candidate = {
        "decisions": [_decision(selected=defend, recommended=recommended, before="a" * 64, override=True)],
        "applied": [_applied(selected=defend, after="c" * 64)],
    }

    assert h16_runner._pair_has_effective_override(parent, candidate) is True
    candidate["applied"] = [_applied(selected=defend, after="b" * 64)]
    assert h16_runner._pair_has_effective_override(parent, candidate) is False
    candidate["applied"] = [_applied(selected=defend, after="c" * 64)]
    candidate["decisions"] = [
        _decision(selected=defend, recommended=recommended, before="d" * 64, override=True)
    ]
    assert h16_runner._pair_has_effective_override(parent, candidate) is False


def test_action_trace_completeness_fails_closed_on_missing_post_action() -> None:
    events = [
        _decision(
            selected={"kind": "end_turn"},
            recommended={"kind": "end_turn"},
            before="a" * 64,
            override=False,
        )
    ]

    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_action_trace_completeness(events)


def test_trace_action_signature_uses_selected_actions_in_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "actions.ndjson"
    evidence.write_text(
        '{"type":"simulator_noncombat","floor":1,"act":1,"choice":{"index":0}}\n'
        '{"type":"simulator_combat_action","floor":1,"chosen_bits":17,"mcts_sims":2000}\n',
        encoding="utf-8",
    )
    signature, action_count = h16_runner._evidence_action_signature(evidence)
    assert len(signature) == 64
    assert action_count == 2


def test_trace_replay_requires_matching_terminal_summary_and_action_signature() -> None:
    trace_on = {
        "outcome": "defeat",
        "result": "PASS_SIMULATOR_COMPLETE_RUN",
        "final_floor": 18,
        "final_hp": 0,
        "game_steps": 73,
        "lethal_intent_defend_override_count": 1,
    }
    trace_off = dict(trace_on)
    assert h16_runner._trace_replay_matches(
        trace_on, trace_off, "a" * 64, "a" * 64, 42, 42
    )
    trace_off["final_floor"] = 17
    assert not h16_runner._trace_replay_matches(
        trace_on, trace_off, "a" * 64, "a" * 64, 42, 42
    )
