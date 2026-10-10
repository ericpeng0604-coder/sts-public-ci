from copy import deepcopy
import json

import pytest

from roguelike_ai.sts1_phase3.inflame_end_turn import select_inflame_before_end_turn
from scripts.sts1.sts1_g7_h20_shadow_audit import AuditError, audit
from scripts.sts1 import sts1_g7_h16_lethal_defend_eval as runner


def test_h20_coverage_requires_matching_parent_and_changed_afterstate():
    _, actions, end = case()
    def decision(action, before, override):
        return {"encounter_index": 1, "battle_step": 2,
                "state_before_signature_sha256": before,
                "selected_action": action, "mcts_recommended_action": end,
                "inflame_override": override}
    def applied(action, after):
        return {"encounter_index": 1, "battle_step": 2,
                "selected_action": action, "state_after_signature_sha256": after}
    parent = {"decisions": [decision(end, "a" * 64, False)],
              "applied": [applied(end, "b" * 64)]}
    candidate = {"decisions": [decision(actions[1], "a" * 64, True)],
                 "applied": [applied(actions[1], "c" * 64)]}
    assert runner._pair_has_effective_override(parent, candidate, "h20")
    assert not runner._pair_has_effective_override(parent, candidate, "h19")
    candidate["decisions"][0]["state_before_signature_sha256"] = "d" * 64
    assert not runner._pair_has_effective_override(parent, candidate, "h20")


def test_h20_replay_checks_its_own_override_count():
    on = {"inflame_override_count": 1}
    off = {"inflame_override_count": 2}
    assert not runner._trace_replay_matches(on, off, "a", "a", 1, 1, "h20")


def test_h20_denominator_and_retention_are_unchanged():
    assert runner._stage_episode_counts("h20", "train", 10)["expected_episodes"] == 21
    assert runner.STAGE_CONFIG["train"]["minimum_override_seed_coverage"] == 3
    assert runner.TRIAL_ROUND_IDS["h20"] != runner.TRIAL_ROUND_IDS["h19"]


def case():
    state = {
        "combat_active": True, "energy": 1, "powers": [],
        "hand": [{"id": "INFLAME", "position": 0, "cost": 1,
                  "upgrades": 0, "is_playable": True, "has_target": False}],
        "enemies": [{"name": "THE_CHAMP", "hp": 50, "is_gone": False, "powers": []}],
    }
    end = {"kind": "end_turn"}
    actions = [end, {"kind": "play_card", "hand_index": 0, "action_id": "exact-native-action"}]
    return state, actions, end


def select(state, actions, end, complete=True):
    choice, reason = select_inflame_before_end_turn(state, actions, end, legal_actions_complete=complete)
    return (None if choice is None else choice.action), reason


def test_selects_exact_legal_action_and_preserves_input_evidence():
    state, actions, end = case()
    original = deepcopy((state, actions, end))
    action, reason = select(state, actions, end)
    assert action == actions[1]
    assert reason == "legal_inflame_before_end_turn"
    assert (state, actions, end) == original
    action["hand_index"] = 99
    assert actions[1]["hand_index"] == 0


@pytest.mark.parametrize("name", ["AWAKENED_ONE", "TIME_EATER", "CORRUPT_HEART"])
def test_excludes_enemy_card_play_penalties(name):
    state, actions, end = case()
    state["enemies"][0]["name"] = name
    assert select(state, actions, end)[0] is None


@pytest.mark.parametrize("name", ["PAIN", "NORMALITY"])
def test_excludes_hand_curses(name):
    state, actions, end = case()
    state["hand"].append({"id": name, "position": 1})
    assert select(state, actions, end)[0] is None


@pytest.mark.parametrize("target", ["powers", "enemy"])
def test_unknown_powers_fail_closed(target):
    state, actions, end = case()
    powers = [{"name": "UNKNOWN_TRIGGER", "amount": 1}]
    if target == "powers":
        state["powers"] = powers
    else:
        state["enemies"][0]["powers"] = powers
    assert select(state, actions, end)[0] is None


@pytest.mark.parametrize("energy", [0, -1, True, "1"])
def test_insufficient_or_unknown_energy_never_overrides(energy):
    state, actions, end = case()
    state["energy"] = energy
    assert select(state, actions, end)[0] is None


def test_legal_completeness_and_live_membership_are_required():
    state, actions, end = case()
    assert select(state, actions, end, False)[0] is None
    assert select(state, actions[1:], end)[0] is None
    assert select(state, [end], end)[0] is None
    different = dict(actions[1], action_id="different-action")
    assert select(state, actions + [different], end)[0] is None


def test_multiple_cards_use_lowest_hand_index_deterministically():
    state, actions, end = case()
    card = deepcopy(state["hand"][0])
    card["position"] = 2
    state["hand"].insert(0, card)
    actions.insert(1, {"kind": "play_card", "hand_index": 2})
    assert select(state, actions, end)[0] == actions[2]
    assert select(state, list(reversed(actions)), end)[0] == actions[2]


def test_duplicate_hand_positions_are_not_inferred():
    state, actions, end = case()
    state["hand"].append(deepcopy(state["hand"][0]))
    assert select(state, actions, end)[0] is None


def test_malformed_power_names_abstain_without_exception():
    state, actions, end = case()
    state["powers"] = [{"name": [], "amount": 1}]
    assert select(state, actions, end)[0] is None


def test_identical_projection_aliases_retain_the_first_exact_native_ordinal():
    state, actions, end = case()
    actions.append(dict(actions[1]))
    original = deepcopy(actions)
    choice, reason = select_inflame_before_end_turn(state, actions, end, legal_actions_complete=True)
    assert reason == "legal_inflame_before_end_turn"
    assert choice.native_action_index == 1
    assert choice.action == actions[choice.native_action_index]
    assert actions == original


def synthetic_traces(directory, *, stage="train", duplicate_identity=False):
    for index in range(10):
        state, actions, end = case()
        metadata = {
            "arm": "parent", "stage": stage, "round_id": "round-011-20261010",
            "trial_id": "h19", "mcts_sims": 2000, "seed_disjointness_verified": True,
            "seed_id": "synthetic-test-source" if duplicate_identity else f"synthetic-test-source-{index}",
            "simulator_gameplay_commit": "7476a81954020087da31d41d16fddf475746ec2d",
        }
        for key in ("g7_checkpoint_sha256", "simulator_binding_sha256", "armg_source_sha256",
                    "armg_vocab_sha256", "simulator_policy_source_sha256", "candidate_evaluator_sha256"):
            metadata[key] = "a" * 64  # Synthetic identity; not an experiment artifact.
        records = [
            {"type": "diagnostic_trace_header_v1", "run_metadata": metadata},
            {"type": "combat_decision_trace_v1", "mcts_sims": 2000,
             "legal_actions_complete": True, "mcts_recommended_action": end,
             "selected_action": end, "public_state": state, "canonical_native_legal_actions": actions},
            {"complete": True, "final_floor": 1, "illegal_action_count": 0,
             "crash_count": 0, "timeout_count": 0, "communication_error_count": 0},
        ]
        (directory / f"parent-trace-fixture-{index}.ndjson").write_text(
            "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
        )


def test_shadow_audit_emits_only_repeatable_aggregates_and_preserves_sources(tmp_path):
    synthetic_traces(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = audit(tmp_path)
    assert result == audit(tmp_path)
    assert result["shadow_action_changes"] == result["distinct_trace_coverage"] == 10
    assert result["new_episodes"] == 0
    assert "synthetic-test-source" not in json.dumps(result)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_shadow_audit_rejects_probe_evidence_before_selection(tmp_path):
    synthetic_traces(tmp_path, stage="probe")
    with pytest.raises(AuditError, match="registered parent Train evidence"):
        audit(tmp_path)


def test_shadow_audit_rejects_duplicate_source_identity(tmp_path):
    synthetic_traces(tmp_path, duplicate_identity=True)
    with pytest.raises(AuditError, match="duplicate source seed identity"):
        audit(tmp_path)
