from __future__ import annotations

import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sts1"))

import sts1_g7_h16_lethal_defend_eval as h16_runner
from roguelike_ai.sts1_phase3.simulator import (
    SimulatorRunError,
    _apply_full_potion_shop_skip_guard,
    _apply_lethal_intent_defend_rescue,
)


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


def _h19_allocation() -> dict[str, object]:
    return {
        "record_type": "h19_pool_allocation",
        "schema_version": "sts1-g7-pool-allocation-v1",
        "trial_id": "h19",
        "round_id": h16_runner.ROUND_ID,
        "exclusion_inventory": {
            "inventory_id": "fixture-inventory",
            "sha256": "a" * 64,
            "source_count": 3,
            "unique_excluded_seed_count": 12,
        },
        "seed_ledger_sha256": "b" * 64,
        "pools": {
            name: {"pool_id": f"fixture-{name}", "manifest_sha256": "c" * 64}
            for name in h16_runner.POOL_FILES
        },
    }


def _h19_identities() -> dict[str, str]:
    return {
        "simulator_policy_source_sha256": "policy",
        "candidate_evaluator_sha256": "evaluator",
        "simulator_binding_sha256": "binding",
        "armg_source_sha256": "armg",
        "armg_vocab_sha256": "vocab",
        "g7_checkpoint_sha256": "parent",
        "simulator_gameplay_commit": h16_runner.PINNED_GAMEPLAY_COMMIT,
        "identity_lock_sha256": "lock",
        "stage_gate_protocol_version": h16_runner.STAGE_GATE_PROTOCOL_VERSION,
    }


def test_h19_private_paths_and_legacy_trials_fail_closed(tmp_path) -> None:
    pools_dir = tmp_path / "round-010-seeds" / "pools"
    output = tmp_path / "round-010-h19-train-20261010"
    usage = pools_dir.parent / "h19-usage-private.jsonl"
    inventory = pools_dir.parent / "exclusion-inventory.json"
    identity_lock = pools_dir.parent / "h19-identity-private.json"

    h16_runner._validate_private_paths(
        pools_dir, "h19", "train", output, usage, inventory, identity_lock
    )
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_private_paths(
            pools_dir, "h18", "train", output, usage, inventory, identity_lock
        )
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_private_paths(
            pools_dir, "h19", "train", h16_runner.REPO_ROOT / "private-output",
            usage, inventory, identity_lock
        )


def test_h19_train_transition_requires_exact_private_allocation() -> None:
    allocation = _h19_allocation()
    candidate_commit = "a" * 40
    identities = _h19_identities()

    h16_runner._validate_transition(
        trial_id="h19",
        stage="train",
        events=[allocation],
        allocation_event=allocation,
        candidate_commit=candidate_commit,
        identities=identities,
    )

    altered = deepcopy(allocation)
    altered["round_id"] = "wrong-round"
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_transition(
            trial_id="h19",
            stage="train",
            events=[altered],
            allocation_event=allocation,
            candidate_commit=candidate_commit,
            identities=identities,
        )


def test_h19_probe_requires_eligible_train_summary_and_same_identity() -> None:
    candidate_commit = "b" * 40
    allocation = _h19_allocation()
    identities = _h19_identities()
    summary = {
        "record_type": "h19_stage_summary",
        "trial_id": "h19",
        "stage": "train",
        "status": "COMPLETE",
        "candidate_commit": candidate_commit,
        "advance_eligible": True,
        **identities,
    }
    h16_runner._validate_transition(
        trial_id="h19",
        stage="probe",
        events=[allocation, summary],
        allocation_event=allocation,
        candidate_commit=candidate_commit,
        identities=identities,
    )

    changed = {**summary, "identity_lock_sha256": "different-lock"}
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_transition(
            trial_id="h19",
            stage="probe",
            events=[allocation, changed],
            allocation_event=allocation,
            candidate_commit=candidate_commit,
            identities=identities,
        )


def test_h19_identity_lock_is_mandatory_and_public_source_has_no_private_pins(tmp_path) -> None:
    lock = tmp_path / "private-identity.json"
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._validate_identity(
            tmp_path, tmp_path, tmp_path / "g7.pt", lock, "h19", "a" * 40
        )

    source = Path(h16_runner.__file__).read_text(encoding="utf-8")
    assert re.search(r"\b[0-9a-f]{64}\b", source) is None
    assert "EXPECTED_H17_ALLOCATION" not in source
    assert "EXPECTED_H18_ALLOCATION" not in source
    parser = h16_runner._parser()
    trial_action = next(action for action in parser._actions if action.dest == "trial_id")
    assert trial_action.choices == ("h19",)


def test_h19_identity_lock_pins_all_runtime_inputs(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    module_dir = tmp_path / "module"
    armg_root = tmp_path / "armg"
    checkpoint = tmp_path / "g7.pt"
    source = repo / "src" / "roguelike_ai" / "sts1_phase3" / "simulator.py"
    upstream = repo / "external" / "sts_lightspeed" / "UPSTREAM.json"
    binding = module_dir / "slaythespire.cp312-win_amd64.pyd"
    armg_source = armg_root / "armG_train.py"
    armg_vocab = armg_root / "armS_card_vocab.json"
    for path, content in (
        (source, b"simulator"),
        (upstream, json.dumps({
            "repository": h16_runner.PINNED_GAMEPLAY_REPOSITORY_URL,
            "commit": h16_runner.PINNED_GAMEPLAY_COMMIT,
        }).encode()),
        (binding, b"binding"),
        (armg_source, b"armg-source"),
        (armg_vocab, b"armg-vocab"),
        (checkpoint, b"frozen-g7"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(h16_runner, "REPO_ROOT", repo)
    candidate_commit = "d" * 40
    paths = {
        "simulator_binding_sha256": binding,
        "armg_source_sha256": armg_source,
        "armg_vocab_sha256": armg_vocab,
        "g7_checkpoint_sha256": checkpoint,
        "simulator_policy_source_sha256": source,
        "candidate_evaluator_sha256": Path(h16_runner.__file__).resolve(),
    }
    lock = {
        "schema_version": h16_runner.PRIVATE_IDENTITY_SCHEMA_VERSION,
        "trial_id": "h19",
        "candidate_commit": candidate_commit,
        "simulator_gameplay_commit": h16_runner.PINNED_GAMEPLAY_COMMIT,
        "mcts_sims": 2000,
        "expected_sha256": {key: h16_runner._sha256(path) for key, path in paths.items()},
    }
    lock_path = tmp_path / "private-identity.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    identities = h16_runner._validate_identity(
        module_dir, armg_root, checkpoint, lock_path, "h19", candidate_commit
    )
    assert identities["simulator_gameplay_commit"] == h16_runner.PINNED_GAMEPLAY_COMMIT
    assert identities["identity_lock_sha256"] == h16_runner._sha256(lock_path)
    assert set(h16_runner.PRIVATE_IDENTITY_KEYS) <= set(identities)


def test_round010_assets_validate_generated_pools_and_private_allocation(tmp_path) -> None:
    ledger_module = h16_runner.seed_ledger
    inventory = {
        "schema_version": ledger_module.INVENTORY_SCHEMA_VERSION,
        "inventory_id": "synthetic-inventory",
        "complete": True,
        "source_manifests": [
            {
                "category": category,
                "source_ref": f"fixture:{category}",
                "sha256": f"{index:064x}",
                "seed_ids": [],
            }
            for index, category in enumerate(ledger_module.REQUIRED_SOURCE_CATEGORIES, 1)
        ],
    }
    seed_dir = tmp_path / "round-010-seeds"
    pools_dir = seed_dir / "pools"
    pools_dir.mkdir(parents=True)
    inventory_path = seed_dir / "exclusion-inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    ledger = ledger_module.generate_exploration_round(
        inventory,
        inventory_sha256=h16_runner._sha256(inventory_path),
        round_id=h16_runner.ROUND_ID,
        generation_key="unit-test-only",
    )
    (pools_dir / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    for role, manifest in ledger["pools"].items():
        (pools_dir / h16_runner.POOL_FILES[role]).write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    pool, seeds, preflight, allocation = h16_runner._round010_assets(
        trial_id="h19",
        stage="train",
        pool_file=pools_dir / "train_hypothesis_1.json",
        pools_dir=pools_dir,
        inventory_path=inventory_path,
        allocation_event=None,
    )
    assert len(seeds) == 10
    assert pool["pool_id"] == allocation["pools"]["train_hypothesis_1"]["pool_id"]
    assert preflight["seed_disjointness_verified"] is True

    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._round010_assets(
            trial_id="h19",
            stage="train",
            pool_file=pools_dir / "train_hypothesis_1.json",
            pools_dir=pools_dir,
            inventory_path=inventory_path,
            allocation_event={"record_type": "wrong"},
        )


def _shop_choice(item_type: str | None, *, leave: bool = False) -> dict[str, object]:
    return {"kind": "shop", "item_type": item_type, "leave": leave}


def _shop_guard(
    *,
    screen_name: str = "SHOP_ROOM",
    kind: str = "shop",
    recommended_index: int = 0,
    selected_index: int = 0,
    choices: list[dict[str, object]] | None = None,
    potions: list[str] | None = None,
    gc: object | None = None,
) -> tuple[int, dict[str, object] | None]:
    if choices is None:
        choices = [_shop_choice("POTION"), _shop_choice(None, leave=True)]
    if gc is None:
        gc = SimpleNamespace(potions=potions or ["FIRE_POTION"] * 5)
    return _apply_full_potion_shop_skip_guard(
        screen_name=screen_name,
        shop_screen_name="SHOP_ROOM",
        kind=kind,
        recommended_index=recommended_index,
        selected_index=selected_index,
        choice_semantics=choices,
        gc=gc,
    )


def test_full_potion_shop_guard_executes_legal_skip_and_records_decision() -> None:
    selected, intervention = _shop_guard()

    assert selected == 1
    assert intervention == {
        "status": "overridden",
        "reason": "potion_capacity_full",
        "recommended_index": 0,
        "executed_index": 1,
        "recommended_choice": {"kind": "shop", "item_type": "POTION", "leave": False},
        "executed_choice": {"kind": "shop", "item_type": None, "leave": True},
    }


def test_full_potion_shop_guard_keeps_purchase_when_a_slot_is_empty() -> None:
    selected, intervention = _shop_guard(potions=["EMPTY_POTION_SLOT"] + ["FIRE_POTION"] * 4)

    assert selected == 0
    assert intervention is None


def test_full_potion_shop_guard_is_inactive_outside_shop_or_for_non_potions() -> None:
    selected, intervention = _shop_guard(
        screen_name="EVENT_ROOM",
        gc=SimpleNamespace(),
    )
    assert selected == 0
    assert intervention is None

    selected, intervention = _shop_guard(
        choices=[_shop_choice("RELIC"), _shop_choice(None, leave=True)],
        gc=SimpleNamespace(),
    )
    assert selected == 0
    assert intervention is None


@pytest.mark.parametrize(
    "gc",
    [
        SimpleNamespace(),
        SimpleNamespace(potions=["FIRE_POTION"] * 4 + ["UNKNOWN"]),
    ],
)
def test_full_potion_shop_guard_fails_closed_on_missing_or_unknown_inventory(gc: object) -> None:
    with pytest.raises(SimulatorRunError, match="potion inventory"):
        _shop_guard(gc=gc)


def test_full_potion_shop_guard_fails_closed_without_one_legal_skip() -> None:
    with pytest.raises(SimulatorRunError, match="legal shop skip"):
        _shop_guard(choices=[_shop_choice("POTION")])

    with pytest.raises(SimulatorRunError, match="legal shop skip"):
        _shop_guard(
            choices=[_shop_choice("POTION"), _shop_choice(None, leave=True), _shop_choice(None, leave=True)]
        )


def test_full_potion_shop_guard_fails_closed_on_unknown_recommendation_semantics() -> None:
    with pytest.raises(SimulatorRunError, match="shop action semantics"):
        _shop_guard(choices=[_shop_choice(None), _shop_choice(None, leave=True)])


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


def _registered_pair_summary(
    stage: str, *, candidate_only: int = 0, parent_only: int = 0, both_victories: int = 0
) -> dict[str, int | float]:
    count = h16_runner.STAGE_CONFIG[stage]["paired_seed_count"]
    both_defeats = count - candidate_only - parent_only - both_victories
    assert min(candidate_only, parent_only, both_victories, both_defeats) >= 0
    parent = (
        ["victory"] * both_victories
        + ["victory"] * parent_only
        + ["defeat"] * candidate_only
        + ["defeat"] * both_defeats
    )
    candidate = (
        ["victory"] * both_victories
        + ["defeat"] * parent_only
        + ["victory"] * candidate_only
        + ["defeat"] * both_defeats
    )
    return h16_runner.h3._paired_summary_for_stage(parent, candidate)


def test_stage_gate_protocol_version_and_registered_denominators_are_reported() -> None:
    assert {
        stage: config["paired_seed_count"]
        for stage, config in h16_runner.STAGE_CONFIG.items()
    } == {
        "train": 10,
        "probe": 10,
        "dev": 30,
        "confirmation_a": 100,
        "confirmation_b": 100,
    }
    assert h16_runner._stage_episode_counts("h19", "train", 10) == {
        "paired_episode_count": 20,
        "trace_invariance_replay_count": 1,
        "expected_episodes": 21,
    }
    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._stage_episode_counts("h18", "train", 10)
    gate = h16_runner._stage_gate("train", _registered_pair_summary("train"), 3, True)
    assert gate["stage_gate_protocol_version"] == h16_runner.STAGE_GATE_PROTOCOL_VERSION


def test_train_probe_gates_use_distinct_effective_override_seed_coverage() -> None:
    nonnegative_train = _registered_pair_summary("train")
    assert h16_runner._stage_gate("train", nonnegative_train, 2, True)["advance_eligible"] is False
    assert h16_runner._stage_gate("train", nonnegative_train, 3, True)["advance_eligible"] is True
    assert h16_runner._stage_gate("probe", nonnegative_train, 3, False)["advance_eligible"] is False
    assert h16_runner._stage_gate(
        "train", _registered_pair_summary("train", parent_only=1), 3, True
    )["advance_eligible"] is False


def test_stage_gate_rejects_missing_or_inconsistent_pair_counters() -> None:
    with pytest.raises(h16_runner.EvaluationIntegrityError, match="canonical integer counters"):
        h16_runner._stage_gate("dev", {"net_wins": 4}, 0, True)
    inconsistent = _registered_pair_summary("dev", candidate_only=4)
    inconsistent["net_wins"] = 3
    with pytest.raises(h16_runner.EvaluationIntegrityError, match="disagree"):
        h16_runner._stage_gate("dev", inconsistent, 0, True)


def test_dev_positive_net_is_candidate_selection_signal_without_p_threshold() -> None:
    paired = _registered_pair_summary("dev", candidate_only=4)
    assert paired["candidate_only_wins"] == 4
    assert paired["parent_only_wins"] == 0
    assert paired["exact_one_sided_sign_p_candidate_positive"] == pytest.approx(0.0625)
    assert h16_runner._stage_gate(
        "dev",
        paired,
        0,
        True,
    )["advance_eligible"] is True
    assert h16_runner._stage_gate(
        "dev",
        _registered_pair_summary("dev"),
        0,
        True,
    )["advance_eligible"] is False
    assert h16_runner._stage_gate(
        "dev", _registered_pair_summary("dev", parent_only=1), 0, True
    )["advance_eligible"] is False


def test_dev_hard_safety_or_integrity_failure_still_blocks_confirmation() -> None:
    assert h16_runner._stage_gate(
        "dev",
        _registered_pair_summary("dev", candidate_only=4),
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
    assert h16_runner._stage_gate(
        "train", _registered_pair_summary("train", candidate_only=1), 3, True
    )["advance_eligible"] is True


def _synthetic_confirmation_batch(stage: str, candidate_only: int, parent_only: int) -> dict[str, object]:
    pairs = []
    diagnostic_pairs = []
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
        diagnostic_pairs.append({
            "parent_outcome": parent_outcome,
            "candidate_outcome": candidate_outcome,
            "parent_floor": 50,
            "candidate_floor": 49 if index < 5 else 50,
            "parent_hp": 50,
            "candidate_hp": 49 if index < 5 else 50,
        })
    return {
        "stage": stage,
        "stage_gate_protocol_version": h16_runner.STAGE_GATE_PROTOCOL_VERSION,
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
        "terminal_floor_hp_diagnostic": h16_runner._terminal_floor_hp_diagnostic(diagnostic_pairs),
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
    assert first["terminal_floor_hp_diagnostics"]["confirmation_a"][
        "candidate_relation_counts"]["terminal_floor"]["lower"] == 5
    assert first["stage_gate_protocol_version"] == h16_runner.STAGE_GATE_PROTOCOL_VERSION
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


@pytest.mark.parametrize(
    "failure",
    (
        "missing_arm",
        "unknown_outcome",
        "hard_guard",
        "seed_provenance",
        "identity",
        "floor_hp",
        "floor_hp_count",
    ),
)
def test_confirmation_gate_fails_closed_on_incomplete_or_mismatched_evidence(failure: str) -> None:
    batch_a = _synthetic_confirmation_batch("confirmation_a", candidate_only=6, parent_only=0)
    batch_b = _synthetic_confirmation_batch("confirmation_b", candidate_only=6, parent_only=0)
    if failure == "missing_arm":
        del batch_a["pairs"][0]["candidate"]
    elif failure == "unknown_outcome":
        batch_a["pairs"][0]["candidate"]["outcome"] = "unknown"
    elif failure == "hard_guard":
        batch_b["hard_guards_passed"] = False
    elif failure == "seed_provenance":
        batch_b["seed_disjointness_verified"] = False
    elif failure == "identity":
        batch_b["candidate_evaluator_sha256"] = "9" * 64
    elif failure == "floor_hp":
        batch_b["terminal_floor_hp_diagnostic"] = None
    elif failure == "floor_hp_count":
        batch_b["terminal_floor_hp_diagnostic"]["paired_delta_summary"]["final_hp"]["count"] = 99

    with pytest.raises(h16_runner.EvaluationIntegrityError):
        h16_runner._confirmation_gate(batch_a, batch_b, trial_k=1, prior_trial_ks=[])


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


def test_shared_shop_guard_does_not_count_as_candidate_override_coverage() -> None:
    parent = {"decisions": [], "applied": []}
    candidate = {
        "decisions": [
            {
                "type": "noncombat_decision_trace_v1",
                "shop_policy_intervention": {
                    "status": "overridden",
                    "reason": "potion_capacity_full",
                },
            }
        ],
        "applied": [],
    }

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
