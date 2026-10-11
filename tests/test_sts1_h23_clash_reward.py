from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import pytest

from roguelike_ai.sts1_phase3.clash_reward_skip import select_clash_reward_skip, validate_override, effective_override
from roguelike_ai.sts1_phase3.simulator import run_simulator_game, SimulatorRunError
from scripts.sts1 import sts1_g7_h16_lethal_defend_eval as runner
from scripts.sts1 import sts1_g7_h23_clash_reward_eval as h23

TYPES = {"Clash": "ATTACK", "Strike": "ATTACK", "Defend": "SKILL"}
CHOICES = [{"kind": "card", "choice": "card", "card_name": "Clash+"}, {"kind": "card", "choice": "skip"}]
DECK = [{"name": "Strike", "upgrades": 0}, {"name": "Defend", "upgrades": 0}]


def event(override=False):
    selected, detail = select_clash_reward_skip(0, CHOICES, DECK, TYPES)
    return {"type": "noncombat_decision_trace_v1", "game_step": 3, "screen_before": "REWARDS",
            "state_before": {"deck": deepcopy(DECK)},
            "state_after": {"deck": deepcopy(DECK) + ([] if override else [{"name": "Clash+", "upgrades": 1}])},
            "legal_choices_complete": True,
            "legal_choices": [{"index": i, "semantics": deepcopy(c)} for i, c in enumerate(CHOICES)],
            "recommended_legal_action_index": 0, "selected_legal_action_index": selected if override else 0,
            "clash_reward_override": override, "clash_reward_intervention": detail if override else None,
            "clash_reward_reason": detail["reason"] if override else "disabled_or_not_reward"}


def test_exact_skip_preserves_inputs_and_upgraded_identity():
    original = deepcopy((CHOICES, DECK, TYPES))
    selected, detail = select_clash_reward_skip(0, CHOICES, DECK, TYPES)
    assert selected == 1 and detail["overridden"] is True
    assert (CHOICES, DECK, TYPES) == original
    validate_override(event(True), TYPES)


@pytest.mark.parametrize("deck,choices,types,complete", [
    ([], CHOICES, TYPES, True), ([{"name": "Strike"}], CHOICES, TYPES, True),
    ([{"name": "unknown"}], CHOICES, TYPES, True), (DECK, CHOICES[:1], TYPES, True),
    (DECK, CHOICES + [CHOICES[1]], TYPES, True), (DECK, CHOICES, TYPES, False),
    (DECK, CHOICES, {"Defend": "UNKNOWN"}, True),
])
def test_unknown_attack_only_and_ambiguous_skip_abstain(deck, choices, types, complete):
    selected, detail = select_clash_reward_skip(0, choices, deck, types, legal_choices_complete=complete)
    assert selected == 0 and detail["overridden"] is False


def test_effective_override_requires_equal_recorded_state_and_prefix():
    p, c = event(False), event(True)
    prefix = {"type": "combat_decision_trace_v1", "game_step": 2,
              "state_before_signature_sha256": "a" * 64, "selected_action": {"kind": "end_turn"}}
    assert effective_override([prefix, p], [deepcopy(prefix), c])
    changed = deepcopy(prefix)
    changed["selected_action"] = {"kind": "other"}
    assert not effective_override([prefix, p], [changed, c])
    changed = deepcopy(c)
    changed["state_before"]["gold"] = 99
    assert not effective_override([p], [changed])
    changed = deepcopy(c)
    changed["state_after"] = deepcopy(p["state_after"])
    assert not effective_override([p], [changed])


@pytest.mark.parametrize("key,value", [("selected_legal_action_index", 0),
                                      ("legal_choices_complete", False), ("clash_reward_override", False)])
def test_override_trace_recomputes_frozen_selector(key, value):
    e = event(True)
    e[key] = value
    with pytest.raises(ValueError):
        validate_override(e, TYPES)


def test_h23_preserves_existing_stage_gates_and_requires_known_runtime():
    assert runner._stage_episode_counts("h23", "train", 10)["expected_episodes"] == 21
    assert runner.TRIAL_ROUND_IDS["h23"] == "round-018-20261011"
    assert runner._stage_gate("train", runner.h3._paired_summary_for_stage(["defeat"] * 10, ["defeat"] * 10), 2, True)["advance_eligible"] is False
    assert inspect.signature(run_simulator_game).parameters["skip_clash_reward_in_mixed_deck"].default is False
    with pytest.raises(SimulatorRunError, match="ArmG"):
        run_simulator_game(student=None, sts=None, seed=1, skip_clash_reward_in_mixed_deck=True)
    with pytest.raises(SimulatorRunError, match="boolean"):
        run_simulator_game(student=None, sts=None, seed=1, skip_clash_reward_in_mixed_deck=1)


def test_h23_pool_exclusion_and_freeze_before_first_episode(monkeypatch, tmp_path):
    base = h23.collection
    prior = tmp_path / "prior"
    seeds = prior / "round-017-seeds"
    (seeds / "pools").mkdir(parents=True)
    inventory = {"schema_version": base.ledger.INVENTORY_SCHEMA_VERSION, "complete": True,
        "inventory_id": "fixture", "source_manifests": [
            {"category": category, "source_ref": "fixture:" + category, "sha256": "a" * 64,
             "seed_ids": [i + 1]} for i, category in enumerate(base.ledger.REQUIRED_SOURCE_CATEGORIES)]}
    base.write_new(seeds / "exclusion-inventory.json", inventory)
    old = base.ledger.generate_exploration_round(inventory,
        inventory_sha256=runner._sha256(seeds / "exclusion-inventory.json"),
        round_id="fixture-prior", generation_key="private-fixture")
    for name, pool in old["pools"].items():
        base.write_new(seeds / "pools" / (name + ".json"), pool)
    keys = set(runner.PRIVATE_IDENTITY_KEYS) | {"candidate_selector_sha256"} | set(runner.h23_identity_paths(tmp_path))
    runtime = {}
    for key in keys:
        runtime[key] = tmp_path / ("slaythespire.cp312-win_amd64.pyd" if key == "simulator_binding_sha256" else key)
        runtime[key].write_text(key)
    base.write_new(prior / "identity-private.json", {"expected_sha256": {k: runner._sha256(p) for k, p in runtime.items()}})
    args = SimpleNamespace(prior_root=prior, module_dir=tmp_path, native_source=tmp_path,
                           g7_checkpoint=runtime["g7_checkpoint_sha256"])
    monkeypatch.setattr(base, "runtime_files", lambda args: runtime)
    monkeypatch.setattr(base, "head", lambda: "b" * 40)
    monkeypatch.setattr(runner, "_validate_pinned_gameplay_manifest", lambda p: None)
    monkeypatch.setattr(h23, "load_native_types", lambda p: TYPES)
    root = tmp_path / "round18"
    assert h23.prepare(args, root)["expected_episodes"] == 21
    lock, train, _ = base.validate_preflight(args, root, h23.ROUND, h23.SEEDS)
    excluded, _ = base.ledger.validate_inventory(json.loads((root / h23.SEEDS / "exclusion-inventory.json").read_text()))
    old_ids = {s for p in old["pools"].values() for s in p["seed_ids"]}
    assert old_ids <= excluded and not old_ids.intersection(train)
    assert lock["expected_episodes"] == 21
    runtime["candidate_selector_sha256"].write_text("drift")
    with pytest.raises(ValueError, match="runtime"):
        base.validate_preflight(args, root, h23.ROUND, h23.SEEDS)
