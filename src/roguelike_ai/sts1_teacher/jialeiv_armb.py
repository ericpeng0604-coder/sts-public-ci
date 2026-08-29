"""Public-state-compatible adapter for Jialeiv's Arm B combat policy.

Upstream Arm B was behavior-cloned from the hidden-RNG MCTS oracle, but its
*inference inputs* are public combat facts: player stats/statuses, visible enemy
intent damage/statuses, hand/draw/discard composition, turn, and the candidate
legal action.  This module reproduces that inference encoding from
``DecisionContext`` without exposing a raw ``BattleContext`` to the model.

The original weights remain external MIT-licensed artifacts.  We pin and verify
their Git blob identities rather than copying them into this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import ActionSpec, DecisionContext

VOCAB_CAP = 256
MAX_MON = 5

PLAYER_STATUSES = (
    "VULNERABLE",
    "WEAK",
    "FRAIL",
    "STRENGTH",
    "DEXTERITY",
    "FOCUS",
    "ARTIFACT",
    "METALLICIZE",
    "REGEN",
    "PLATED_ARMOR",
    "DEMON_FORM",
    "RITUAL",
    "THORNS",
    "BARRICADE",
    "INTANGIBLE",
    "NO_BLOCK",
    "ENTANGLED",
)
MONSTER_STATUSES = (
    "VULNERABLE",
    "WEAK",
    "POISON",
    "STRENGTH",
    "ARTIFACT",
    "METALLICIZE",
    "REGEN",
    "PLATED_ARMOR",
    "BLOCK_RETURN",
    "MARK",
    "LOCK_ON",
    "SHACKLED",
    "INTANGIBLE",
    "CHOKED",
)
ACTION_TYPES = (
    "CARD",
    "POTION",
    "SINGLE_CARD_SELECT",
    "MULTI_CARD_SELECT",
    "END_TURN",
)
ACTION_TYPE_INDEX = {name: index for index, name in enumerate(ACTION_TYPES)}

STATE_DIM = 7 + len(PLAYER_STATUSES) + MAX_MON * (6 + len(MONSTER_STATUSES)) + 3 * VOCAB_CAP + 1
CAND_DIM = len(ACTION_TYPES) + VOCAB_CAP + MAX_MON + 1
INPUT_DIM = STATE_DIM + CAND_DIM

JIALEIV_ARM_B_WEIGHT_PATH = "weights/armB_model_B256x256.pt"
JIALEIV_ARM_B_WEIGHT_GIT_BLOB_SHA = "7b9db5598b1863b117d1ab8b9a80f4ed9056722c"
JIALEIV_VOCAB_PATH = "agent/armS_card_vocab.json"
JIALEIV_VOCAB_GIT_BLOB_SHA = "30041ef13c199f7aa11847cef860777be52c537e"


class JialeivArmBContractError(ValueError):
    """Raised when a public state cannot be represented by frozen Arm B."""


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _norm_name(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def load_frozen_vocab(path: str | Path, *, expected_git_blob: str = JIALEIV_VOCAB_GIT_BLOB_SHA) -> dict[str, int]:
    raw = Path(path).read_bytes()
    actual = git_blob_sha(raw)
    if actual != expected_git_blob:
        raise JialeivArmBContractError(f"jialeiv_vocab_git_blob_mismatch:{actual}:{expected_git_blob}")
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise JialeivArmBContractError("jialeiv_vocab_must_be_object")
    vocab: dict[str, int] = {}
    for name, index in parsed.items():
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < VOCAB_CAP:
            raise JialeivArmBContractError(f"jialeiv_vocab_bad_index:{name}:{index!r}")
        vocab[str(name)] = index
    return vocab


def verify_weight_blob(path: str | Path, *, expected_git_blob: str = JIALEIV_ARM_B_WEIGHT_GIT_BLOB_SHA) -> str:
    raw = Path(path).read_bytes()
    actual = git_blob_sha(raw)
    if actual != expected_git_blob:
        raise JialeivArmBContractError(f"jialeiv_armb_weight_git_blob_mismatch:{actual}:{expected_git_blob}")
    return actual


def _power_map(values: Sequence[Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            continue
        name = _norm_name(raw.get("id") or raw.get("name"))
        if not name:
            continue
        amount = raw.get("amount", raw.get("damage", 0))
        if isinstance(amount, bool) or not isinstance(amount, int | float):
            amount = 0
        result[name] = float(amount)
    return result


def _status_amount(powers: Mapping[str, float], name: str) -> float:
    target = _norm_name(name)
    aliases = {
        "PLATED_ARMOR": ("PLATED_ARMOR", "PLATED ARMOR"),
        "DEMON_FORM": ("DEMON_FORM", "DEMON FORM"),
        "NO_BLOCK": ("NO_BLOCK", "NO BLOCK"),
        "BLOCK_RETURN": ("BLOCK_RETURN", "BLOCK RETURN", "SHARP_HIDE", "SHARP HIDE"),
        "LOCK_ON": ("LOCK_ON", "LOCK ON"),
    }.get(target, (target,))
    for alias in aliases:
        key = _norm_name(alias)
        if key in powers:
            return powers[key]
    return 0.0


def _card_index(card: Mapping[str, Any], vocab: Mapping[str, int]) -> int:
    name = str(card.get("name") or card.get("id") or "")
    return int(vocab.get(name, VOCAB_CAP - 1))


def _counts(cards: Sequence[Any], vocab: Mapping[str, int]) -> list[float]:
    vector = [0.0] * VOCAB_CAP
    for raw in cards:
        if not isinstance(raw, Mapping):
            continue
        vector[_card_index(raw, vocab)] += 1.0
    return vector


def _is_attacking(enemy: Mapping[str, Any]) -> bool:
    intent = _norm_name(enemy.get("intent"))
    return "ATTACK" in intent or "DAMAGE" in intent


def encode_public_battle(context: DecisionContext, vocab: Mapping[str, int]) -> list[float]:
    if not bool(context.state.get("combat_active")):
        raise JialeivArmBContractError("armb_requires_active_combat")

    state = context.state
    player_powers = _power_map(state.get("powers", []))
    vector: list[float] = [
        float(state.get("hp") or 0) / 80.0,
        float(state.get("max_hp") or 0) / 80.0,
        float(state.get("block") or 0) / 50.0,
        float(state.get("energy") or 0) / 10.0,
        _status_amount(player_powers, "STRENGTH") / 10.0,
        _status_amount(player_powers, "DEXTERITY") / 10.0,
        _status_amount(player_powers, "FOCUS") / 10.0,
    ]
    vector.extend(_status_amount(player_powers, name) / 10.0 for name in PLAYER_STATUSES)

    enemies = [item for item in state.get("enemies", []) if isinstance(item, Mapping)]
    if len(enemies) > MAX_MON:
        raise JialeivArmBContractError(f"armb_too_many_enemies:{len(enemies)}")
    for index in range(MAX_MON):
        if index >= len(enemies):
            vector.extend([0.0] * (6 + len(MONSTER_STATUSES)))
            continue
        enemy = enemies[index]
        powers = _power_map(enemy.get("powers", []))
        intent_damage = enemy.get("intent_damage", 0)
        intent_hits = enemy.get("intent_hits", 0)
        if isinstance(intent_damage, bool) or not isinstance(intent_damage, int | float):
            intent_damage = 0
        if isinstance(intent_hits, bool) or not isinstance(intent_hits, int | float):
            intent_hits = 0
        vector.extend(
            [
                1.0,
                float(enemy.get("hp") or 0) / 100.0,
                float(enemy.get("block") or 0) / 50.0,
                1.0 if _is_attacking(enemy) else 0.0,
                float(intent_damage) / 50.0,
                float(intent_hits) / 5.0,
            ]
        )
        vector.extend(_status_amount(powers, name) / 10.0 for name in MONSTER_STATUSES)

    vector.extend(_counts(state.get("hand", []), vocab))
    vector.extend(_counts(state.get("draw_pile", []), vocab))
    vector.extend(_counts(state.get("discard_pile", []), vocab))
    vector.append(float(state.get("turn") or 0) / 20.0)

    if len(vector) != STATE_DIM:
        raise AssertionError(f"armb_state_dim:{len(vector)}:{STATE_DIM}")
    return vector


def encode_candidate(context: DecisionContext, action: ActionSpec, vocab: Mapping[str, int]) -> list[float]:
    payload = action.payload
    kind = str(payload.get("kind", ""))
    if kind == "play_card":
        action_type = "CARD"
    elif kind in {"use_potion", "discard_potion"}:
        action_type = "POTION"
    elif kind == "end_turn":
        action_type = "END_TURN"
    elif kind == "choose":
        selection_type = str(payload.get("selection_type", "")).upper()
        if selection_type not in {"SINGLE_CARD_SELECT", "MULTI_CARD_SELECT"}:
            raise JialeivArmBContractError(f"armb_card_select_kind_missing:{selection_type or 'NONE'}")
        action_type = selection_type
    else:
        raise JialeivArmBContractError(f"armb_unsupported_action:{kind}")

    vector = [0.0] * CAND_DIM
    vector[ACTION_TYPE_INDEX[action_type]] = 1.0
    offset = len(ACTION_TYPES)

    if action_type == "CARD":
        hand_index = payload.get("hand_index")
        card = None
        if isinstance(hand_index, int) and not isinstance(hand_index, bool):
            for fallback, raw in enumerate(context.state.get("hand", []), start=1):
                if isinstance(raw, Mapping) and raw.get("position", fallback) == hand_index:
                    card = raw
                    break
        if card is None:
            raise JialeivArmBContractError(f"armb_missing_hand_card:{hand_index!r}")
        vector[offset + _card_index(card, vocab)] = 1.0
        cost = card.get("cost", 0)
        if isinstance(cost, bool) or not isinstance(cost, int | float):
            cost = 0
        vector[offset + VOCAB_CAP + MAX_MON] = float(cost) / 3.0

    target = payload.get("target_index")
    if isinstance(target, int) and not isinstance(target, bool) and 0 <= target < MAX_MON:
        vector[offset + VOCAB_CAP + target] = 1.0

    return vector


@dataclass(frozen=True)
class ArmBPolicyScores:
    action_ids: tuple[str, ...]
    scores: tuple[float, ...]
    tie_action_ids: tuple[str, ...]
    unique_best_action_id: str | None


class JialeivArmBPolicy:
    """Load and run the pinned Arm B MLP strictly from a DecisionContext."""

    uses_hidden_information = False
    policy_id = "jialeiv-armb-b256x256-public-adapter-v1"

    def __init__(self, *, vocab: Mapping[str, int], model: Any) -> None:
        self.vocab = dict(vocab)
        self.model = model

    @classmethod
    def from_files(cls, *, vocab_path: str | Path, weight_path: str | Path) -> "JialeivArmBPolicy":
        vocab = load_frozen_vocab(vocab_path)
        verify_weight_blob(weight_path)
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("torch is required to load the external Arm B policy") from exc

        layers: list[Any] = []
        previous = INPUT_DIM
        for hidden in (256, 256):
            layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        model = nn.Sequential(*layers)
        state_dict = torch.load(Path(weight_path), weights_only=True, map_location="cpu")
        normalized = {
            (key[4:] if str(key).startswith("net.") else str(key)): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(normalized, strict=True)
        model.eval()
        return cls(vocab=vocab, model=model)

    def score_actions(self, context: DecisionContext, *, tie_tolerance: float = 1e-9) -> ArmBPolicyScores:
        actions: list[ActionSpec] = []
        rows: list[list[float]] = []
        state = encode_public_battle(context, self.vocab)
        for action in context.legal_actions:
            try:
                candidate = encode_candidate(context, action, self.vocab)
            except JialeivArmBContractError:
                continue
            actions.append(action)
            rows.append(state + candidate)

        if not actions:
            return ArmBPolicyScores((), (), (), None)

        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required to run the external Arm B policy") from exc
        with torch.no_grad():
            raw = self.model(torch.tensor(rows, dtype=torch.float32)).squeeze(-1).tolist()
        if not isinstance(raw, list):
            raw = [float(raw)]
        scores = tuple(float(value) for value in raw)
        best = max(scores)
        ties = tuple(
            action.action_id
            for action, score in zip(actions, scores, strict=True)
            if abs(score - best) <= tie_tolerance
        )
        return ArmBPolicyScores(
            action_ids=tuple(action.action_id for action in actions),
            scores=scores,
            tie_action_ids=ties,
            unique_best_action_id=ties[0] if len(ties) == 1 else None,
        )


__all__ = [
    "ACTION_TYPES",
    "ArmBPolicyScores",
    "CAND_DIM",
    "INPUT_DIM",
    "JIALEIV_ARM_B_WEIGHT_GIT_BLOB_SHA",
    "JIALEIV_ARM_B_WEIGHT_PATH",
    "JIALEIV_VOCAB_GIT_BLOB_SHA",
    "JIALEIV_VOCAB_PATH",
    "JialeivArmBContractError",
    "JialeivArmBPolicy",
    "MAX_MON",
    "MONSTER_STATUSES",
    "PLAYER_STATUSES",
    "STATE_DIM",
    "VOCAB_CAP",
    "encode_candidate",
    "encode_public_battle",
    "git_blob_sha",
    "load_frozen_vocab",
    "verify_weight_blob",
]
