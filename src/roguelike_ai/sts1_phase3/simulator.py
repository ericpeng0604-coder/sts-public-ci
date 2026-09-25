"""Pinned sts_lightspeed full-run harness for STS1 Phase 3 A0.

Combat uses the exact frozen Student v0. Non-combat decisions use a tiny
public deterministic fallback. Native action aliases are collapsed only when
we can prove they are the same targetless-card action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from .frozen_student import (
    FrozenStudentV0,
    canonical_json,
    normalize_action_payload,
    project_policy_observation,
    sha256_json,
)
from .protocol import A0_FROZEN_SEEDS_V1, A0_PROTOCOL_VERSION, frozen_a0_manifest


SIMULATOR_EVIDENCE_SCHEMA = "sts1-phase3-a0-simulator-v1"
MAX_GAME_STEPS = 600
MAX_BATTLE_STEPS = 800


class SimulatorRunError(RuntimeError):
    """The pinned simulator could not execute the formal Phase-3 policy safely."""


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


def _value(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _card(card: Any, *, position: int | None = None, playable: bool | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if position is not None:
        result["position"] = position
    card_id = _value(card, "id")
    if card_id is not None:
        result["id"] = _enum_name(card_id)
    name = _value(card, "name")
    if name is not None:
        result["name"] = str(name)
    card_type = _value(card, "type")
    if card_type is not None:
        result["type"] = _enum_name(card_type)
    cost = _value(card, "cost_for_turn", _value(card, "cost"))
    if cost is not None:
        result["cost"] = int(cost)
    upgraded = _value(card, "upgraded")
    if upgraded is not None:
        result["upgrades"] = 1 if bool(upgraded) else 0
    requires_target = _value(card, "requires_target")
    if requires_target is not None:
        result["has_target"] = bool(requires_target)
    if playable is not None:
        result["is_playable"] = playable
    return result


def _player_powers(player: Any) -> list[dict[str, Any]]:
    powers: list[dict[str, Any]] = []
    for attr, public_name in (
        ("strength", "Strength"),
        ("dexterity", "Dexterity"),
        ("focus", "Focus"),
        ("artifact", "Artifact"),
    ):
        amount = _value(player, attr, 0)
        if isinstance(amount, int) and not isinstance(amount, bool) and amount != 0:
            powers.append({"name": public_name, "amount": amount})
    return powers


def _enemy(enemy: Any, *, index: int, battle: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"index": index}
    for attr, public_name in (
        ("name", "name"),
        ("cur_hp", "hp"),
        ("max_hp", "max_hp"),
        ("block", "block"),
    ):
        value = _value(enemy, attr)
        if value is not None:
            result[public_name] = str(value) if public_name == "name" else value

    intent = _value(enemy, "intent")
    if intent is not None:
        result["intent"] = str(intent)
    intent_damage_api = getattr(enemy, "intent_damage", None)
    if callable(intent_damage_api):
        damage_info = intent_damage_api(battle)
        damage = _value(damage_info, "damage")
        attack_count = _value(damage_info, "attack_count")
        if damage is not None:
            result["intent_damage"] = damage
        if attack_count is not None:
            result["intent_hits"] = attack_count
    elif intent_damage_api is not None:
        result["intent_damage"] = intent_damage_api
        intent_hits = _value(enemy, "intent_hits")
        if intent_hits is not None:
            result["intent_hits"] = intent_hits

    alive = _value(enemy, "alive")
    if alive is not None:
        result["is_gone"] = not bool(alive)

    powers: list[dict[str, Any]] = []
    for attr, public_name in (
        ("strength", "Strength"),
        ("vulnerable", "Vulnerable"),
        ("weak", "Weak"),
        ("poison", "Poison"),
    ):
        amount = _value(enemy, attr, 0)
        if isinstance(amount, int) and not isinstance(amount, bool) and amount != 0:
            powers.append({"name": public_name, "amount": amount})
    result["powers"] = powers
    return result


def _action_type(action: Any) -> str:
    return _enum_name(_value(action, "action_type", "UNKNOWN")).upper()


def _public_action(action: Any, hand: Sequence[Any]) -> dict[str, Any]:
    kind = _action_type(action)
    source_idx = _value(action, "source_idx", -1)
    target_idx = _value(action, "target_idx", -1)

    if kind == "CARD":
        item: dict[str, Any] = {"kind": "play_card", "hand_index": int(source_idx) + 1}
        card = hand[int(source_idx)] if isinstance(source_idx, int) and 0 <= source_idx < len(hand) else None
        if card is not None and bool(_value(card, "requires_target", False)):
            item["target_index"] = int(target_idx)
        return item

    if kind == "POTION":
        if isinstance(target_idx, int) and target_idx > 5:
            return {"kind": "discard_potion", "potion_index": int(source_idx)}
        item = {"kind": "use_potion", "potion_index": int(source_idx)}
        if isinstance(target_idx, int) and target_idx >= 0:
            item["target_index"] = int(target_idx)
        return item

    if kind == "END_TURN":
        return {"kind": "end_turn"}

    if kind in {"SINGLE_CARD_SELECT", "MULTI_CARD_SELECT"}:
        item = {"kind": "choose", "choice_index": int(source_idx), "selection_type": kind}
        if isinstance(target_idx, int) and target_idx >= 0:
            item["target_index"] = int(target_idx)
        return item

    return {
        "kind": "simulator_public_action",
        "action_type": kind,
        "source_idx": source_idx,
        "target_idx": target_idx,
    }


def _safe_targetless_card_alias(first: Any, second: Any, hand: Sequence[Any]) -> bool:
    """Accept only the pinned simulator's proven target-index alias for targetless cards."""

    if _action_type(first) != "CARD" or _action_type(second) != "CARD":
        return False
    first_source = _value(first, "source_idx", -1)
    second_source = _value(second, "source_idx", -1)
    if not isinstance(first_source, int) or first_source != second_source:
        return False
    if first_source < 0 or first_source >= len(hand):
        return False
    card = hand[first_source]
    requires_target = _value(card, "requires_target", None)
    return isinstance(requires_target, bool) and requires_target is False


def _project_legal_actions(
    actions: Sequence[Any],
    hand: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Project native actions and preserve an exact public->native index map.

    The only permitted collapse is a duplicate public action for the same
    targetless card. That alias was verified on the pinned simulator by cloning
    the same BattleContext and observing identical post-action public states.
    """

    public_actions: list[dict[str, Any]] = []
    native_indices: list[int] = []
    first_by_identity: dict[str, tuple[int, Any]] = {}
    alias_count = 0

    for native_index, action in enumerate(actions):
        projected = _public_action(action, hand)
        identity = canonical_json(projected)
        prior = first_by_identity.get(identity)
        if prior is not None:
            _, prior_action = prior
            if not _safe_targetless_card_alias(prior_action, action, hand):
                raise SimulatorRunError(
                    "duplicate projected legal action without proven targetless-card equivalence"
                )
            alias_count += 1
            continue

        first_by_identity[identity] = (native_index, action)
        public_actions.append(projected)
        native_indices.append(native_index)

    if not public_actions and actions:
        raise SimulatorRunError("native legal actions projected to an empty public action set")
    return public_actions, native_indices, alias_count


def _legal_actions(actions: Sequence[Any], hand: Sequence[Any]) -> list[dict[str, Any]]:
    """Compatibility helper returning only public actions."""

    public_actions, _, _ = _project_legal_actions(actions, hand)
    return public_actions


def _hybrid_vote_choice_bits(
    mcts_votes: Sequence[tuple[int, int]],
    *,
    student_bits: int,
) -> tuple[int, bool]:
    """Choose among MCTS recommendations, using Student only to break MCTS ties."""
    if len(mcts_votes) < 2:
        raise SimulatorRunError("hybrid MCTS requires at least two MCTS budgets")
    counts: dict[int, int] = {}
    for budget, bits in mcts_votes:
        if budget < 1:
            raise SimulatorRunError("hybrid MCTS budget must be positive")
        counts[bits] = counts.get(bits, 0) + 1
    best_count = max(counts.values())
    tied = {bits for bits, count in counts.items() if count == best_count}
    if len(tied) == 1:
        return next(iter(tied)), False
    if student_bits in tied:
        return student_bits, True
    # Student chose outside the MCTS tie: prefer the highest-budget tied MCTS vote.
    for budget, bits in sorted(mcts_votes, reverse=True):
        if bits in tied:
            return bits, False
    raise SimulatorRunError("hybrid MCTS could not resolve tied recommendations")


class SimulatorCombatAdapter:
    """Project a pinned BattleContext into the frozen Student public contract."""

    def adapt(
        self,
        battle: Any,
        *,
        legal_actions: Sequence[Any],
        run_state: Mapping[str, Any] | None = None,
        projected_legal_actions: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        run = dict(run_state or {})
        player = _value(battle, "player")
        hand_raw = _sequence(_value(battle, "hand", []))
        action_list = _sequence(legal_actions)
        playable_slots = {
            int(_value(action, "source_idx"))
            for action in action_list
            if _action_type(action) == "CARD" and isinstance(_value(action, "source_idx"), int)
        }
        if projected_legal_actions is None:
            public_actions, _, _ = _project_legal_actions(action_list, hand_raw)
        else:
            public_actions = [dict(item) for item in projected_legal_actions]

        state: dict[str, Any] = {
            "schema_version": "sts1-public-state-v1",
            "source": "simulator",
            "hp": _value(player, "cur_hp"),
            "max_hp": _value(player, "max_hp"),
            "block": _value(player, "block"),
            "energy": _value(player, "energy"),
            "hand": [
                _card(card, position=index + 1, playable=index in playable_slots)
                for index, card in enumerate(hand_raw)
            ],
            "draw_pile": [_card(card) for card in _sequence(_value(battle, "draw_pile", []))],
            "discard_pile": [_card(card) for card in _sequence(_value(battle, "discard_pile", []))],
            "exhaust_pile": [_card(card) for card in _sequence(_value(battle, "exhaust_pile", []))],
            "powers": _player_powers(player),
            "enemies": [
                _enemy(enemy, index=index, battle=battle)
                for index, enemy in enumerate(_sequence(_value(battle, "monsters", [])))
            ],
            "turn": _value(battle, "turn"),
            "combat_active": _enum_name(_value(battle, "outcome", "UNDECIDED")).upper() == "UNDECIDED",
            "relics": list(run.get("relics", [])),
            "potions": list(run.get("potions", [])),
            "gold": run.get("gold"),
            "floor": run.get("floor"),
            "act": run.get("act"),
            "character": run.get("character", "IRONCLAD"),
            "ascension_level": run.get("ascension_level", 0),
            "room": run.get("room", "COMBAT"),
            "screen_type": run.get("screen_type", "NONE"),
            "screen_choices": list(run.get("screen_choices", [])),
            "rewards": list(run.get("rewards", [])),
            "map_choices": list(run.get("map_choices", [])),
            "legal_actions": public_actions,
        }
        projected = project_policy_observation(state)
        normalized_actions = [normalize_action_payload(item) for item in state["legal_actions"]]
        signature_payload = dict(projected)
        signature_payload["legal_actions"] = sorted(normalized_actions, key=canonical_json)
        state["decision_signature"] = sha256_json(signature_payload)
        return state


def public_run_state(gc: Any) -> dict[str, Any]:
    floor = _value(gc, "floor_num")
    act = _value(gc, "act")
    gold = _value(gc, "gold")
    return {
        "floor": int(floor) if isinstance(floor, int) and not isinstance(floor, bool) else floor,
        "act": int(act) if isinstance(act, int) and not isinstance(act, bool) else act,
        "gold": int(gold) if isinstance(gold, int) and not isinstance(gold, bool) else gold,
        "character": "IRONCLAD",
        "ascension_level": 0,
        "room": "COMBAT",
        "screen_type": "NONE",
    }


def deterministic_noncombat_step(gc: Any, sts: Any) -> str:
    if gc.screen_state == sts.ScreenState.REWARDS:
        offered = list(gc.get_card_reward())
        if offered:
            gc.pick_reward_card(offered[0])
            return "reward:first_card"
        gc.skip_reward_cards()
        return "reward:skip_empty"

    supported = {
        sts.ScreenState.MAP_SCREEN,
        sts.ScreenState.REST_ROOM,
        sts.ScreenState.SHOP_ROOM,
        sts.ScreenState.EVENT_SCREEN,
    }
    if gc.screen_state not in supported:
        raise SimulatorRunError(f"unsupported non-combat screen: {gc.screen_state}")
    actions = list(sts.get_legal_game_actions(gc))
    if not actions:
        raise SimulatorRunError(f"no legal non-combat action on screen: {gc.screen_state}")
    actions[0].execute(gc)
    return f"{_enum_name(gc.screen_state)}:first_legal"


class ArmGNoncombatPolicy:
    """Exact frozen upstream ArmG inference for public non-combat decisions."""

    def __init__(self, *, root: Path, weight_path: Path) -> None:
        if not root.is_dir():
            raise SimulatorRunError(f"ArmG compatibility root missing: {root}")
        if not weight_path.is_file():
            raise SimulatorRunError(f"ArmG weight missing: {weight_path}")

        os.environ["STS_BOT_DIR"] = str(root)
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        try:
            torch = importlib.import_module("torch")
            module = importlib.import_module("armG_train")
        except Exception as exc:
            raise SimulatorRunError(f"could not import frozen ArmG runtime: {exc}") from exc

        # Freeze vocabulary lookup exactly like the accepted baseline lifecycle fix.
        module.card_idx = lambda name: module._vocab.get(name, module.VOCAB_CAP - 1)
        try:
            net = module.Scorer((128, 128))
            net.load_state_dict(torch.load(weight_path, weights_only=True, map_location="cpu"))
            net.eval()
        except Exception as exc:
            raise SimulatorRunError(f"could not load frozen ArmG weight: {exc}") from exc

        self.module = module
        self.torch = torch
        self.net = net

    def step(self, gc: Any, sts: Any) -> str:
        kind, descs, execs = self.module.build_choices(gc)
        if not descs:
            if gc.screen_state == sts.ScreenState.REWARDS:
                gc.skip_reward_cards()
                return "armg:reward:skip_empty"
            raise SimulatorRunError(f"ArmG produced no legal choice on screen: {gc.screen_state}")
        if len(descs) == 1:
            index = 0
        else:
            with self.torch.no_grad():
                obs = self.torch.tensor(self.module.obs_vec(gc), dtype=self.torch.float32)
                scores = self.net.score(obs, descs)
                index = int(self.torch.argmax(scores).item())
        execs[index](gc)
        return f"armg:{kind}:{index}/{len(descs)}"


def _set_pauses(agent: Any) -> None:
    agent.pause_on_card_reward = True
    agent.pause_on_map = True
    agent.pause_on_rest = True
    agent.pause_on_shop = True
    agent.pause_on_event = True
    agent.pause_on_battle = True


def resolve_simulator_seed(
    sts: Any,
    seed: str | int,
    *,
    heldout_seeds: Sequence[int] | None = None,
    training_seeds: Sequence[int] | None = None,
) -> tuple[str, int]:
    """Resolve frozen UI, held-out evaluation, or training-only internal seeds."""

    if heldout_seeds is not None and training_seeds is not None:
        raise SimulatorRunError("held-out and training seed allowlists are mutually exclusive")

    ui_seed = str(seed)
    if training_seeds is not None:
        try:
            simulator_seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise SimulatorRunError(f"training simulator seed must be an integer: {seed!r}") from exc
        allowed = {int(value) for value in training_seeds}
        if simulator_seed not in allowed:
            raise SimulatorRunError("training simulator seed is not present in the training seed allowlist")
        return ui_seed, simulator_seed

    if heldout_seeds is not None:
        try:
            simulator_seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise SimulatorRunError(f"held-out simulator seed must be an integer: {seed!r}") from exc
        allowed = {int(value) for value in heldout_seeds}
        if simulator_seed not in allowed:
            raise SimulatorRunError("held-out simulator seed is not present in the pinned evaluation seed set")
        return ui_seed, simulator_seed

    if ui_seed not in A0_FROZEN_SEEDS_V1:
        raise SimulatorRunError("formal simulator run requires a seed from the frozen A0 manifest")
    get_seed_long = getattr(sts, "get_seed_long", None)
    get_seed_str = getattr(sts, "get_seed_str", None)
    if not callable(get_seed_long) or not callable(get_seed_str):
        raise SimulatorRunError("pinned simulator seed conversion API is unavailable")
    try:
        simulator_seed = int(get_seed_long(ui_seed))
        round_trip = str(get_seed_str(simulator_seed))
    except Exception as exc:
        raise SimulatorRunError(f"pinned simulator seed conversion failed: {exc}") from exc
    if round_trip != ui_seed:
        raise SimulatorRunError(
            f"pinned simulator seed round-trip mismatch: ui={ui_seed!r} round_trip={round_trip!r}"
        )
    return ui_seed, simulator_seed


def run_simulator_game(
    *,
    student: Any,
    sts: Any,
    seed: str | int,
    evidence_path: Path | None = None,
    max_game_steps: int = MAX_GAME_STEPS,
    max_battle_steps: int = MAX_BATTLE_STEPS,
    armg_policy: ArmGNoncombatPolicy | None = None,
    combat_mcts_sims: int | None = None,
    combat_mcts_budgets: Sequence[int] | None = None,
    hybrid_mcts_budgets: Sequence[int] | None = None,
    combat_mcts_late_sims: int | None = None,
    combat_mcts_late_floor: int = 50,
    combat_mcts_exploration: float | None = None,
    heldout_seeds: Sequence[int] | None = None,
    training_seeds: Sequence[int] | None = None,
    collect_ppo: bool = False,
    collect_teacher: bool = False,
) -> dict[str, Any]:
    ui_seed, simulator_seed = resolve_simulator_seed(
        sts,
        seed,
        heldout_seeds=heldout_seeds,
        training_seeds=training_seeds,
    )
    seed_contract = (
        "training_internal"
        if training_seeds is not None
        else ("heldout_internal" if heldout_seeds is not None else "frozen_a0_ui")
    )
    if max_game_steps < 1 or max_battle_steps < 1:
        raise SimulatorRunError("simulator step bounds must be positive")
    consensus_budgets = tuple(int(value) for value in (combat_mcts_budgets or ()))
    hybrid_budgets = tuple(int(value) for value in (hybrid_mcts_budgets or ()))
    active_mcts_modes = sum(bool(value) for value in (
        combat_mcts_sims is not None,
        consensus_budgets,
        hybrid_budgets,
    ))
    if active_mcts_modes > 1:
        raise SimulatorRunError("choose exactly one MCTS combat mode")
    if combat_mcts_sims is not None and consensus_budgets:
        raise SimulatorRunError("choose either one MCTS budget or consensus budgets, not both")
    if combat_mcts_late_sims is not None and consensus_budgets:
        raise SimulatorRunError("late MCTS schedule cannot be combined with consensus budgets")
    if combat_mcts_late_sims is not None and combat_mcts_sims is None:
        raise SimulatorRunError("late MCTS schedule requires --combat-mcts-sims as the base budget")
    if combat_mcts_sims is not None and combat_mcts_sims < 1:
        raise SimulatorRunError("combat MCTS simulations must be positive")
    if combat_mcts_late_sims is not None and combat_mcts_late_sims < 1:
        raise SimulatorRunError("late combat MCTS simulations must be positive")
    if combat_mcts_late_floor < 1:
        raise SimulatorRunError("late combat MCTS floor must be positive")
    if combat_mcts_exploration is not None and combat_mcts_exploration <= 0:
        raise SimulatorRunError("combat MCTS exploration parameter must be positive")
    if combat_mcts_exploration is not None and consensus_budgets:
        raise SimulatorRunError("tuned exploration cannot be combined with consensus budgets")
    if any(value < 1 for value in consensus_budgets):
        raise SimulatorRunError("all MCTS consensus budgets must be positive")
    if hybrid_budgets and (len(hybrid_budgets) < 2 or any(value < 1 for value in hybrid_budgets)):
        raise SimulatorRunError("hybrid MCTS requires at least two positive budgets")
    if collect_ppo and collect_teacher:
        raise SimulatorRunError("PPO and MCTS Teacher collection are mutually exclusive")
    if collect_ppo and (combat_mcts_sims is not None or consensus_budgets or hybrid_budgets):
        raise SimulatorRunError("PPO rollout collection cannot run with MCTS combat policy")
    if collect_ppo and not callable(getattr(student, "sample_action", None)):
        raise SimulatorRunError("PPO rollout collection requires a Student v1 sample_action policy")
    if collect_teacher and combat_mcts_sims is None and not consensus_budgets:
        raise SimulatorRunError("MCTS Teacher collection requires an MCTS combat policy; pure MCTS only")
    if collect_teacher and hybrid_budgets:
        raise SimulatorRunError("MCTS Teacher collection cannot use Student-influenced hybrid MCTS")

    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, simulator_seed, 0)
    agent = sts.Agent()
    _set_pauses(agent)
    adapter = SimulatorCombatAdapter()
    student_actions = 0
    fallback_count = 0
    armg_action_count = 0
    mcts_action_count = 0
    hybrid_student_vote_count = 0
    hybrid_student_tiebreak_count = 0
    illegal_actions = 0
    timeout_count = 0
    crash_count = 0
    equivalent_action_alias_count = 0
    latencies_ms: list[float] = []
    game_steps = 0
    max_floor = int(_value(gc, "floor_num", 0) or 0)
    max_act = int(_value(gc, "act", 0) or 0)
    error: str | None = None

    _record(evidence_path, {
        "type": "frozen_manifest",
        "manifest": frozen_a0_manifest(),
        "ui_seed": ui_seed,
        "simulator_seed_long": simulator_seed,
        "seed_contract": seed_contract,
    })

    try:
        while gc.outcome == sts.GameOutcome.UNDECIDED and game_steps < max_game_steps:
            game_steps += 1
            agent.playout(gc)
            max_floor = max(max_floor, int(_value(gc, "floor_num", 0) or 0))
            max_act = max(max_act, int(_value(gc, "act", 0) or 0))
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                break

            if gc.screen_state != sts.ScreenState.BATTLE:
                screen_before = _enum_name(gc.screen_state)
                if armg_policy is None:
                    choice = deterministic_noncombat_step(gc, sts)
                    fallback_count += 1
                    policy_name = "legacy_fallback"
                else:
                    choice = armg_policy.step(gc, sts)
                    armg_action_count += 1
                    policy_name = "armg"
                _record(evidence_path, {
                    "type": "simulator_noncombat",
                    "floor": int(_value(gc, "floor_num", 0) or 0),
                    "screen": screen_before,
                    "policy": policy_name,
                    "choice": choice,
                })
                continue

            battle = sts.BattleContext()
            battle.init(gc)
            battle_steps = 0
            while battle.outcome == sts.Outcome.UNDECIDED and battle_steps < max_battle_steps:
                battle_steps += 1
                native_actions = list(sts.get_legal_actions(battle))
                if not native_actions:
                    raise SimulatorRunError("no native legal action in undecided battle")
                hand_raw = _sequence(_value(battle, "hand", []))
                public_actions, native_index_map, alias_count = _project_legal_actions(native_actions, hand_raw)
                equivalent_action_alias_count += alias_count
                if alias_count:
                    _record(evidence_path, {
                        "type": "simulator_equivalent_action_alias",
                        "floor": int(_value(gc, "floor_num", 0) or 0),
                        "alias_count": alias_count,
                        "native_action_count": len(native_actions),
                        "public_action_count": len(public_actions),
                    })
                if combat_mcts_sims is not None or consensus_budgets or hybrid_budgets:
                    teacher_public_state = None
                    hybrid_public_state = None
                    if collect_teacher or hybrid_budgets:
                        hybrid_or_teacher_state = adapter.adapt(
                            battle,
                            legal_actions=native_actions,
                            run_state=public_run_state(gc),
                            projected_legal_actions=public_actions,
                        )
                        if collect_teacher:
                            teacher_public_state = hybrid_or_teacher_state
                        if hybrid_budgets:
                            hybrid_public_state = hybrid_or_teacher_state
                    if collect_teacher and teacher_public_state is None:
                        teacher_public_state = adapter.adapt(
                            battle,
                            legal_actions=native_actions,
                            run_state=public_run_state(gc),
                            projected_legal_actions=public_actions,
                        )
                    floor_now = int(_value(gc, "floor_num", 0) or 0)
                    active_mcts_sims = combat_mcts_sims
                    if (
                        combat_mcts_late_sims is not None
                        and floor_now >= combat_mcts_late_floor
                    ):
                        active_mcts_sims = combat_mcts_late_sims
                    started = time.perf_counter()
                    vote_bits: list[int] = []
                    vote_budgets: list[int] = []
                    hybrid_student_bits = None
                    hybrid_student_tiebreak_used = False
                    if len(native_actions) == 1:
                        chosen = native_actions[0]
                    elif hybrid_budgets:
                        if hybrid_public_state is None:
                            raise SimulatorRunError("hybrid MCTS public state was not captured")
                        recommendations: list[tuple[int, Any, int]] = []
                        for budget in hybrid_budgets:
                            recommendation = sts.mcts_recommend(battle, budget)
                            if recommendation is None:
                                raise SimulatorRunError(
                                    f"hybrid MCTS returned no recommendation for budget={budget}"
                                )
                            bits = _value(recommendation, "bits")
                            if not isinstance(bits, int):
                                raise SimulatorRunError(
                                    "hybrid MCTS recommendation did not expose integer action bits"
                                )
                            recommendations.append((budget, recommendation, bits))
                            vote_budgets.append(budget)
                            vote_bits.append(bits)

                        decision = student.select_action(
                            hybrid_public_state,
                            require_command=False,
                        )
                        public_index = int(decision.action_index)
                        if public_index < 0 or public_index >= len(native_index_map):
                            raise SimulatorRunError("hybrid Student vote index is outside legal actions")
                        student_native = native_actions[native_index_map[public_index]]
                        hybrid_student_bits = _value(student_native, "bits")
                        if not isinstance(hybrid_student_bits, int):
                            raise SimulatorRunError("hybrid Student vote has no integer action bits")
                        chosen_bits_vote, hybrid_student_tiebreak_used = _hybrid_vote_choice_bits(
                            [(budget, bits) for budget, _, bits in recommendations],
                            student_bits=hybrid_student_bits,
                        )
                        chosen = next(
                            recommendation
                            for _, recommendation, bits in reversed(recommendations)
                            if bits == chosen_bits_vote
                        )
                        hybrid_student_vote_count += 1
                        hybrid_student_tiebreak_count += int(hybrid_student_tiebreak_used)
                    elif consensus_budgets:
                        recommendations: list[tuple[int, Any, int]] = []
                        for budget in consensus_budgets:
                            recommendation = sts.mcts_recommend(battle, budget)
                            if recommendation is None:
                                raise SimulatorRunError(f"MCTS returned no recommendation for budget={budget}")
                            bits = _value(recommendation, "bits")
                            if not isinstance(bits, int):
                                raise SimulatorRunError("MCTS recommendation did not expose integer action bits")
                            recommendations.append((budget, recommendation, bits))
                            vote_budgets.append(budget)
                            vote_bits.append(bits)
                        counts: dict[int, int] = {}
                        for bits in vote_bits:
                            counts[bits] = counts.get(bits, 0) + 1
                        best_count = max(counts.values())
                        tied = {bits for bits, count in counts.items() if count == best_count}
                        # If all budgets disagree, prefer the highest-budget recommendation.
                        chosen = next(
                            recommendation
                            for _, recommendation, bits in reversed(recommendations)
                            if bits in tied
                        )
                    else:
                        if combat_mcts_exploration is None:
                            chosen = sts.mcts_recommend(battle, active_mcts_sims)
                        else:
                            tuned = getattr(sts, "mcts_recommend_tuned", None)
                            if not callable(tuned):
                                raise SimulatorRunError("experimental tuned MCTS binding is unavailable")
                            chosen = tuned(battle, active_mcts_sims, combat_mcts_exploration)
                        if chosen is None:
                            chosen = native_actions[0]
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    latencies_ms.append(latency_ms)
                    chosen_bits = _value(chosen, "bits")
                    if collect_teacher:
                        if teacher_public_state is None:
                            raise SimulatorRunError("MCTS Teacher public state was not captured")
                        chosen_public = _public_action(chosen, hand_raw)
                        chosen_identity = canonical_json(chosen_public)
                        matches = [
                            index
                            for index, action in enumerate(public_actions)
                            if canonical_json(action) == chosen_identity
                        ]
                        if len(matches) != 1:
                            raise SimulatorRunError(
                                f"MCTS Teacher action mapping is not unique: matches={matches}"
                            )
                        teacher_index = matches[0]
                        teacher_action_id = sha256_json(
                            normalize_action_payload(public_actions[teacher_index])
                        )
                        _record(evidence_path, {
                            "type": "mcts_teacher_decision",
                            "floor": floor_now,
                            "public_state": teacher_public_state,
                            "teacher_action_index": teacher_index,
                            "teacher_action_id": teacher_action_id,
                            "mcts_sims": active_mcts_sims,
                            "mcts_budgets": list(consensus_budgets),
                            "chosen_bits": chosen_bits,
                        })
                    chosen.execute(battle)
                    mcts_action_count += 1
                    _record(evidence_path, {
                        "type": "simulator_combat_action",
                        "floor": int(_value(gc, "floor_num", 0) or 0),
                        "policy": (
                            "hybrid_mcts_student_tiebreak"
                            if hybrid_budgets
                            else ("mcts_consensus" if consensus_budgets else "mcts")
                        ),
                        "mcts_sims": active_mcts_sims,
                        "mcts_base_sims": combat_mcts_sims,
                        "mcts_late_sims": combat_mcts_late_sims,
                        "mcts_late_floor": combat_mcts_late_floor if combat_mcts_late_sims is not None else None,
                        "mcts_exploration": combat_mcts_exploration,
                        "mcts_budgets": list(consensus_budgets),
                        "hybrid_mcts_budgets": list(hybrid_budgets),
                        "student_vote_bits": hybrid_student_bits,
                        "student_tiebreak_used": hybrid_student_tiebreak_used,
                        "vote_budgets": vote_budgets,
                        "vote_bits": vote_bits,
                        "chosen_bits": chosen_bits,
                        "inference_latency_ms": latency_ms,
                    })
                else:
                    public_state = adapter.adapt(
                        battle,
                        legal_actions=native_actions,
                        run_state=public_run_state(gc),
                        projected_legal_actions=public_actions,
                    )
                    started = time.perf_counter()
                    if collect_ppo:
                        decision = student.sample_action(
                            public_state,
                            deterministic=False,
                            require_command=False,
                        )
                    else:
                        decision = student.select_action(public_state, require_command=False)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    latencies_ms.append(latency_ms)
                    if not 0 <= decision.action_index < len(native_index_map):
                        illegal_actions += 1
                        raise SimulatorRunError("Student selected public action index outside legal range")
                    expected_id = sha256_json(
                        normalize_action_payload(public_state["legal_actions"][decision.action_index])
                    )
                    if decision.action_id != expected_id:
                        illegal_actions += 1
                        raise SimulatorRunError("Student/native action identity mapping drift")
                    native_action_index = native_index_map[decision.action_index]
                    native_actions[native_action_index].execute(battle)
                    student_actions += 1
                    record = {
                        "type": "simulator_combat_action",
                        "floor": int(_value(gc, "floor_num", 0) or 0),
                        "policy": "student_v1_ppo" if collect_ppo else "student_v0",
                        "decision_signature": public_state["decision_signature"],
                        "action_id": decision.action_id,
                        "action_index": decision.action_index,
                        "native_action_index": native_action_index,
                        "student_score": decision.score,
                        "inference_latency_ms": latency_ms,
                    }
                    if collect_ppo:
                        record.update({
                            "type": "ppo_decision",
                            "public_state": public_state,
                            "old_log_prob": float(decision.log_prob),
                            "old_value": float(decision.value),
                        })
                    _record(evidence_path, record)
            if battle.outcome == sts.Outcome.UNDECIDED:
                timeout_count += 1
                raise SimulatorRunError(f"battle step bound reached: {max_battle_steps}")
            battle.exit_battle(gc)

        if gc.outcome == sts.GameOutcome.UNDECIDED:
            timeout_count += 1
            raise SimulatorRunError(f"game step bound reached: {max_game_steps}")
    except Exception as exc:
        crash_count = 1
        error = f"{type(exc).__name__}: {exc}"

    if error is not None:
        outcome = "unknown"
        result = "BLOCKED_SIMULATOR"
    elif gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
        outcome = "victory"
        result = "PASS_SIMULATOR_COMPLETE_RUN"
    elif gc.outcome == sts.GameOutcome.PLAYER_LOSS:
        outcome = "defeat"
        result = "PASS_SIMULATOR_COMPLETE_RUN"
    else:
        outcome = "unknown"
        result = "BLOCKED_SIMULATOR_UNKNOWN_OUTCOME"

    summary = {
        "schema_version": SIMULATOR_EVIDENCE_SCHEMA,
        "phase_protocol": A0_PROTOCOL_VERSION,
        "result": result,
        "error": error,
        "seed": ui_seed,
        "simulator_seed_long": simulator_seed,
        "seed_contract": seed_contract,
        "outcome": outcome,
        "final_floor": int(_value(gc, "floor_num", 0) or 0),
        "max_floor": max_floor,
        "max_act": max_act,
        "final_hp": int(_value(gc, "cur_hp", 0) or 0),
        "student_action_count": student_actions,
        "fallback_count": fallback_count,
        "armg_action_count": armg_action_count,
        "mcts_action_count": mcts_action_count,
        "hybrid_student_vote_count": hybrid_student_vote_count,
        "hybrid_student_tiebreak_count": hybrid_student_tiebreak_count,
        "combat_policy": (
            "hybrid_mcts_" + "_".join(str(value) for value in hybrid_budgets) + "_student_tiebreak"
            if hybrid_budgets
            else (
                "mcts_consensus_" + "_".join(str(value) for value in consensus_budgets)
                if consensus_budgets
                else (
                f"mcts_schedule_{combat_mcts_sims}_to_{combat_mcts_late_sims}_floor_{combat_mcts_late_floor}"
                if combat_mcts_late_sims is not None
                else (
                    f"mcts_{combat_mcts_sims}_explore_{combat_mcts_exploration:g}"
                    if combat_mcts_exploration is not None
                    else (
                        f"mcts_{combat_mcts_sims}"
                        if combat_mcts_sims is not None
                        else (
                            "student_v1_ppo_rollout"
                            if collect_ppo
                            else (
                                "student_v1_ppo_deterministic"
                                if callable(getattr(student, "sample_action", None))
                                else "student_v0"
                            )
                        )
                    )
                )
            )
            )
        ),
        "noncombat_policy": "armg" if armg_policy is not None else "legacy_fallback",
        "fallback_rate": fallback_count / max(1, student_actions + fallback_count + armg_action_count),
        "equivalent_action_alias_count": equivalent_action_alias_count,
        "illegal_action_count": illegal_actions,
        "timeout_count": timeout_count,
        "crash_count": crash_count,
        "mean_inference_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
        "max_inference_latency_ms": max(latencies_ms) if latencies_ms else None,
        "game_steps": game_steps,
        "ppo_collection": collect_ppo,
        "teacher_collection": collect_teacher,
    }
    _record(evidence_path, {"type": "summary", **summary})
    return summary


def _record(path: Path | None, value: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")


def _load_sts(module_dir: Path) -> Any:
    if not module_dir.is_dir():
        raise SimulatorRunError(f"simulator module directory missing: {module_dir}")
    sys.path.insert(0, str(module_dir))
    try:
        return importlib.import_module("slaythespire")
    except Exception as exc:
        raise SimulatorRunError(f"could not import pinned slaythespire module: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument(
        "--heldout-seed-file",
        type=Path,
        help="explicit allowlist of exactly 50 numeric internal seeds for held-out evaluation",
    )
    parser.add_argument(
        "--training-seed-file",
        type=Path,
        help="training-only numeric seed allowlist; must not overlap the held-out evaluation set",
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--max-game-steps", type=int, default=MAX_GAME_STEPS)
    parser.add_argument("--max-battle-steps", type=int, default=MAX_BATTLE_STEPS)
    parser.add_argument("--armg-root", type=Path)
    parser.add_argument("--armg-weight", type=Path)
    parser.add_argument("--combat-mcts-sims", type=int)
    parser.add_argument("--combat-mcts-late-sims", type=int)
    parser.add_argument("--combat-mcts-late-floor", type=int, default=50)
    parser.add_argument("--combat-mcts-exploration", type=float)
    parser.add_argument("--student-v1-checkpoint", type=Path)
    parser.add_argument("--collect-ppo", action="store_true")
    parser.add_argument("--collect-teacher", action="store_true")
    parser.add_argument(
        "--hybrid-mcts-budgets",
        help="comma-separated MCTS budgets; Student breaks ties between MCTS recommendations",
    )
    parser.add_argument(
        "--combat-mcts-budgets",
        help="comma-separated MCTS budgets; majority vote with highest-budget tie-break",
    )
    args = parser.parse_args(argv)
    if (args.armg_root is None) != (args.armg_weight is None):
        parser.error("--armg-root and --armg-weight must be supplied together")
    hybrid_mcts_budgets: tuple[int, ...] = ()
    if args.hybrid_mcts_budgets:
        try:
            hybrid_mcts_budgets = tuple(
                int(value.strip())
                for value in args.hybrid_mcts_budgets.split(",")
                if value.strip()
            )
        except ValueError as exc:
            parser.error(f"invalid --hybrid-mcts-budgets: {exc}")
        if len(hybrid_mcts_budgets) < 2:
            parser.error("--hybrid-mcts-budgets requires at least two positive integers")

    combat_mcts_budgets: tuple[int, ...] = ()
    if args.combat_mcts_budgets:
        try:
            combat_mcts_budgets = tuple(
                int(value.strip()) for value in args.combat_mcts_budgets.split(",") if value.strip()
            )
        except ValueError as exc:
            parser.error(f"invalid --combat-mcts-budgets: {exc}")
        if not combat_mcts_budgets:
            parser.error("--combat-mcts-budgets must contain at least one positive integer")
    if args.heldout_seed_file is not None and args.training_seed_file is not None:
        parser.error("--heldout-seed-file and --training-seed-file are mutually exclusive")

    heldout_seeds: tuple[int, ...] | None = None
    if args.heldout_seed_file is not None:
        if not args.heldout_seed_file.is_file():
            parser.error(f"held-out seed file missing: {args.heldout_seed_file}")
        try:
            heldout_seeds = tuple(
                int(line.strip())
                for line in args.heldout_seed_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        except ValueError as exc:
            parser.error(f"invalid held-out seed file: {exc}")
        if len(heldout_seeds) != 50 or len(set(heldout_seeds)) != 50:
            parser.error("held-out seed file must contain exactly 50 unique numeric seeds")

    training_seeds: tuple[int, ...] | None = None
    if args.training_seed_file is not None:
        if not args.training_seed_file.is_file():
            parser.error(f"training seed file missing: {args.training_seed_file}")
        try:
            training_seeds = tuple(
                int(line.strip())
                for line in args.training_seed_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        except ValueError as exc:
            parser.error(f"invalid training seed file: {exc}")
        if not training_seeds or len(set(training_seeds)) != len(training_seeds):
            parser.error("training seed file must contain one or more unique numeric seeds")
        if any(seed < 1 or seed > 10**9 for seed in training_seeds):
            parser.error("training seeds must be in the upstream-compatible range 1..1e9")
        if not (args.collect_ppo or args.collect_teacher):
            parser.error("--training-seed-file requires --collect-ppo or --collect-teacher")

    baseline_student = FrozenStudentV0.from_path(args.model)
    if args.student_v1_checkpoint is not None:
        from .student_v1_ppo import StudentV1PPO
        student = StudentV1PPO.load(args.student_v1_checkpoint, baseline_student)
    else:
        student = baseline_student
    if args.collect_ppo and args.student_v1_checkpoint is None:
        parser.error("--collect-ppo requires --student-v1-checkpoint")
    sts = _load_sts(args.module_dir)
    armg_policy = None
    if args.armg_root is not None and args.armg_weight is not None:
        armg_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.armg_weight)
    summary = run_simulator_game(
        student=student,
        sts=sts,
        seed=args.seed,
        evidence_path=args.evidence,
        max_game_steps=args.max_game_steps,
        max_battle_steps=args.max_battle_steps,
        armg_policy=armg_policy,
        combat_mcts_sims=args.combat_mcts_sims,
        combat_mcts_budgets=combat_mcts_budgets,
        hybrid_mcts_budgets=hybrid_mcts_budgets,
        combat_mcts_late_sims=args.combat_mcts_late_sims,
        combat_mcts_late_floor=args.combat_mcts_late_floor,
        combat_mcts_exploration=args.combat_mcts_exploration,
        heldout_seeds=heldout_seeds,
        training_seeds=training_seeds,
        collect_ppo=args.collect_ppo,
        collect_teacher=args.collect_teacher,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["result"] == "PASS_SIMULATOR_COMPLETE_RUN" else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAX_BATTLE_STEPS", "MAX_GAME_STEPS", "SIMULATOR_EVIDENCE_SCHEMA",
    "ArmGNoncombatPolicy", "SimulatorCombatAdapter", "SimulatorRunError",
    "deterministic_noncombat_step", "public_run_state", "resolve_simulator_seed",
    "run_simulator_game",
]
