from __future__ import annotations

import json
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

from roguelike_ai.sts1_phase3.simulator import run_simulator_game


class _Screen(Enum):
    MAP_SCREEN = auto()
    BATTLE = auto()
    REWARDS = auto()
    REST_ROOM = auto()
    SHOP_ROOM = auto()
    EVENT_SCREEN = auto()


class _GameOutcome(Enum):
    UNDECIDED = auto()
    PLAYER_VICTORY = auto()
    PLAYER_LOSS = auto()


class _BattleOutcome(Enum):
    UNDECIDED = auto()
    PLAYER_VICTORY = auto()


class _Room(Enum):
    ELITE = auto()


class _MapAction:
    def __init__(self, target_x: int) -> None:
        self.idx1 = target_x
        self.bits = target_x + 100

    def execute(self, gc: _GameContext) -> None:
        gc.cur_map_node_x = self.idx1
        gc.cur_map_node_y += 1
        gc.screen_state = _Screen.BATTLE


class _Action:
    action_type = "END_TURN"
    source_idx = -1
    target_idx = -1
    bits = 1

    def execute(self, battle: _BattleContext) -> None:
        battle.outcome = _BattleOutcome.PLAYER_VICTORY


class _AlternateAction:
    action_type = "SIMULATOR_SPECIAL"
    source_idx = 1
    target_idx = -1
    bits = 2

    def execute(self, battle: _BattleContext) -> None:
        battle.outcome = _BattleOutcome.PLAYER_VICTORY


class _BattleContext:
    def init(self, gc: _GameContext) -> None:
        self.player = SimpleNamespace(
            cur_hp=gc.cur_hp,
            max_hp=gc.max_hp,
            block=0,
            energy=3,
            strength=0,
            dexterity=0,
            focus=0,
            artifact=0,
        )
        self.hand = []
        self.draw_pile = []
        self.discard_pile = []
        self.exhaust_pile = []
        self.monsters = [SimpleNamespace(name="Training Dummy", cur_hp=1, max_hp=1, block=0, intent="ATTACK")]
        self.turn = 1
        self.outcome = _BattleOutcome.UNDECIDED

    def exit_battle(self, gc: _GameContext) -> None:
        gc.outcome = _GameOutcome.PLAYER_VICTORY
        gc.screen_state = _Screen.MAP_SCREEN


class _GameContext:
    def __init__(self, _character: object, _seed: int, _ascension: int) -> None:
        self.outcome = _GameOutcome.UNDECIDED
        self.screen_state = _Screen.MAP_SCREEN
        self.floor_num = 1
        self.act = 1
        self.gold = 50
        self.cur_hp = 80
        self.max_hp = 80
        self.cur_map_node_x = 3
        self.cur_map_node_y = 1
        self.deck = []
        self.relics = ["Burning Blood"]
        self.potions = []

    def map_node_room(self, _x: int, _y: int) -> _Room:
        return _Room.ELITE


class _Agent:
    def playout(self, _gc: _GameContext) -> None:
        return None


class _TrainingPolicy:
    def decide(self, _gc: _GameContext, _sts: object):
        def enter_combat_at(target_x: int):
            def enter_combat(gc: _GameContext) -> None:
                gc.cur_map_node_x = target_x
                gc.cur_map_node_y = 2
                gc.screen_state = _Screen.BATTLE

            return enter_combat

        return "map", 1, [
            {"room": "ELITE", "lookahead_1": {"MONSTER": 1}},
            {"room": "ELITE", "lookahead_1": {"MONSTER": 1}},
        ], [enter_combat_at(2), enter_combat_at(5)], [0.0, 1.0]

    def describe_choice(self, _kind: str, value: object) -> dict[str, object]:
        return dict(value)

    def training_vector_snapshot(self, _gc: _GameContext, _descs: list[object]) -> dict[str, list[object]]:
        return {"obs_412": [], "candidate_desc_368": []}

    def deck_snapshot(self, gc: _GameContext) -> list[dict[str, object]]:
        return list(gc.deck)


class _PinnedSTS:
    CharacterClass = SimpleNamespace(IRONCLAD=object())
    ScreenState = _Screen
    GameOutcome = _GameOutcome
    Outcome = _BattleOutcome
    GameContext = _GameContext
    Agent = _Agent
    BattleContext = _BattleContext

    @staticmethod
    def get_seed_long(seed: str) -> int:
        if seed != "347001":
            raise ValueError(seed)
        return 163868251

    @staticmethod
    def get_seed_str(seed: int) -> str:
        if seed != 163868251:
            raise ValueError(seed)
        return "347001"

    @staticmethod
    def get_legal_actions(_battle: _BattleContext) -> list[_Action]:
        return [_Action()]

    @staticmethod
    def get_legal_game_actions(_gc: _GameContext) -> list[_MapAction]:
        return [_MapAction(2), _MapAction(5)]


class _DetachedRecommendationSTS(_PinnedSTS):
    @staticmethod
    def get_legal_actions(_battle: _BattleContext) -> list[object]:
        return [_Action(), _AlternateAction()]

    @staticmethod
    def mcts_recommend(_battle: _BattleContext, _budget: int) -> _Action:
        # Simulate a pybind wrapper returned separately from the legal-action list.
        return _Action()


def test_diagnostic_trace_preserves_fixed_seed_outcome_and_full_legal_actions(tmp_path: Path) -> None:
    common = {
        "student": None,
        "sts": _PinnedSTS,
        "seed": "347001",
        "armg_policy": _TrainingPolicy(),
        "combat_mcts_sims": 2000,
    }
    without_trace = run_simulator_game(**common)
    trace_path = tmp_path / "diagnostic.ndjson"
    with_trace = run_simulator_game(
        **common,
        diagnostic_trace_path=trace_path,
        diagnostic_metadata={"fixture": "trace-invariance"},
    )

    stable_fields = (
        "seed",
        "simulator_seed_long",
        "seed_contract",
        "outcome",
        "result",
        "final_floor",
        "final_hp",
        "illegal_action_count",
        "timeout_count",
        "crash_count",
        "game_steps",
        "armg_action_count",
        "mcts_action_count",
    )
    assert {key: without_trace[key] for key in stable_fields} == {
        key: with_trace[key] for key in stable_fields
    }
    assert with_trace["outcome"] == "victory"
    assert with_trace["illegal_action_count"] == 0
    assert with_trace["timeout_count"] == 0
    assert with_trace["crash_count"] == 0

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    by_type = {record["type"]: record for record in records}
    assert by_type["diagnostic_trace_header_v1"]["ui_seed"] == "347001"
    assert by_type["diagnostic_trace_header_v1"]["run_metadata"] == {"fixture": "trace-invariance"}
    assert by_type["noncombat_decision_trace_v1"]["legal_choices_complete"] is True
    route = by_type["noncombat_decision_trace_v1"]["route"]
    assert by_type["noncombat_decision_trace_v1"]["state_before"]["map_node"] == {"x": 3, "y": 1}
    assert route["choices_complete"] is True
    assert [item["target_node"] for item in route["choices"]] == [
        {"x": 2, "y": 2},
        {"x": 5, "y": 2},
    ]
    assert route["selected_route"] == {
        "from": {"x": 3, "y": 1},
        "to": {"x": 5, "y": 2},
        "room": "ELITE",
        "legal_action_index": 1,
    }
    assert by_type["encounter_started_v1"]["state"]["run"]["hp"] == 80
    combat = by_type["combat_decision_trace_v1"]
    assert combat["canonical_native_legal_actions"] == [{"kind": "end_turn"}]
    assert combat["policy_legal_actions"] == [{"kind": "end_turn"}]
    assert combat["selected_action"] == {"kind": "end_turn"}
    assert by_type["encounter_finished_v1"]["battle_outcome"] == "PLAYER_VICTORY"
    assert by_type["terminal_trace_v1"]["complete"] is True
    assert by_type["terminal_trace_v1"]["legal_actions_complete"] is True


def test_legacy_fallback_trace_records_native_legal_actions_without_changing_run(
    tmp_path: Path,
) -> None:
    common = {
        "student": None,
        "sts": _PinnedSTS,
        "seed": "347001",
        "combat_mcts_sims": 2000,
    }
    without_trace = run_simulator_game(**common)
    trace_path = tmp_path / "legacy-fallback-diagnostic.ndjson"
    with_trace = run_simulator_game(
        **common,
        diagnostic_trace_path=trace_path,
        diagnostic_metadata={"fixture": "legacy-fallback-trace-invariance"},
    )

    stable_fields = (
        "seed",
        "simulator_seed_long",
        "seed_contract",
        "outcome",
        "result",
        "final_floor",
        "final_hp",
        "illegal_action_count",
        "timeout_count",
        "crash_count",
        "game_steps",
        "fallback_count",
        "mcts_action_count",
    )
    assert {key: without_trace[key] for key in stable_fields} == {
        key: with_trace[key] for key in stable_fields
    }
    assert with_trace["outcome"] == "victory"
    assert with_trace["illegal_action_count"] == 0
    assert with_trace["timeout_count"] == 0
    assert with_trace["crash_count"] == 0

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    by_type = {record["type"]: record for record in records}
    noncombat = by_type["noncombat_decision_trace_v1"]
    assert noncombat["policy"] == "legacy_fallback"
    assert noncombat["legal_choices_complete"] is True
    assert noncombat["selected_legal_action_index"] == 0
    assert [choice["semantics"]["bits"] for choice in noncombat["legal_choices"]] == [102, 105]
    assert noncombat["route"]["choices_complete"] is True
    assert noncombat["route"]["selected_route"]["legal_action_index"] == 0
    assert by_type["terminal_trace_v1"]["complete"] is True
    assert by_type["terminal_trace_v1"]["legal_actions_complete"] is True


def test_combat_trace_maps_detached_native_action_wrapper_by_bits(tmp_path: Path) -> None:
    trace_path = tmp_path / "detached-native-action.ndjson"
    result = run_simulator_game(
        student=None,
        sts=_DetachedRecommendationSTS,
        seed="347001",
        combat_mcts_sims=2000,
        diagnostic_trace_path=trace_path,
    )

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    combat = next(record for record in records if record["type"] == "combat_decision_trace_v1")
    terminal = next(record for record in records if record["type"] == "terminal_trace_v1")
    assert result["outcome"] == "victory"
    assert result["illegal_action_count"] == 0
    assert result["timeout_count"] == 0
    assert result["crash_count"] == 0
    assert combat["selected_public_action_index"] == 0
    assert combat["selected_native_action_index"] == 0
    assert combat["legal_actions_complete"] is True
    assert terminal["legal_actions_complete"] is True
