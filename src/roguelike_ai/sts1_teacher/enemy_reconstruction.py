"""Conservative public enemy admission for Phase-1 rollout reconstruction.

V1 intentionally supports only a tiny audited surface. It is better to mark a
combat unsupported than to reconstruct hidden monster state from guesses.
Hidden previous move history is never an input here; the rollout backend must
sample it from the candidate-independent redeterminization plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# V1 currently admits only audited single-enemy Act-1 opening surfaces.
# Jaw Worm has hidden previous-move history after the opening turn, so that
# history is sampled rather than copied. Cultist is especially suitable for the
# next slice because its opening move is deterministically INCANTATION and its
# later move is DARK_STRIKE; no hidden per-instance setup value is required.
_SUPPORTED_V1 = frozenset({"JAW_WORM", "CULTIST"})
_ALLOWED_POWER_NAMES = frozenset({"STRENGTH", "VULNERABLE", "WEAK", "POISON"})
_CULTIST_OPENING_INTENTS = frozenset({"INCANTATION", "CULTIST_INCANTATION"})


def _canonical_name(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True)
class PublicEnemyAdmission:
    allowed: bool
    reasons: tuple[str, ...]
    enemy_count: int


def assess_public_enemies(state: Mapping[str, Any]) -> PublicEnemyAdmission:
    raw_enemies = state.get("enemies")
    if not isinstance(raw_enemies, Sequence) or isinstance(raw_enemies, str | bytes | bytearray):
        return PublicEnemyAdmission(False, ("enemies_not_sequence",), 0)

    reasons: list[str] = []
    enemy_count = len(raw_enemies)
    if enemy_count != 1:
        reasons.append(f"enemy_count_unsupported_v1:{enemy_count}")

    turn = state.get("turn")
    for index, raw_enemy in enumerate(raw_enemies):
        path = f"enemies[{index}]"
        if not isinstance(raw_enemy, Mapping):
            reasons.append(f"enemy_not_mapping:{path}")
            continue

        name = _canonical_name(raw_enemy.get("name"))
        if name not in _SUPPORTED_V1:
            reasons.append(f"enemy_unsupported_v1:{name or 'MISSING'}")

        for field in ("hp", "max_hp", "block"):
            value = raw_enemy.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                reasons.append(f"invalid_enemy_{field}:{path}")

        intent = raw_enemy.get("intent")
        if intent is None or not str(intent).strip():
            reasons.append(f"missing_public_intent:{path}")
        else:
            normalized_intent = _canonical_name(intent)
            # Player reconstruction V1 admits only turn 0. On that exact public
            # surface Cultist's current move is fixed by the pinned simulator:
            # INVALID prior history -> INCANTATION. Reject impossible opening
            # states rather than silently constructing a different timeline.
            if name == "CULTIST" and turn == 0 and normalized_intent not in _CULTIST_OPENING_INTENTS:
                reasons.append(f"cultist_opening_intent_mismatch_v1:{normalized_intent}")

        if raw_enemy.get("is_gone") is True:
            reasons.append(f"gone_enemy_unsupported_v1:{path}")

        powers = raw_enemy.get("powers", [])
        if not isinstance(powers, Sequence) or isinstance(powers, str | bytes | bytearray):
            reasons.append(f"enemy_powers_not_sequence:{path}")
            continue
        for power_index, power in enumerate(powers):
            if not isinstance(power, Mapping):
                reasons.append(f"enemy_power_not_mapping:{path}.powers[{power_index}]")
                continue
            power_name = _canonical_name(power.get("name"))
            if power_name not in _ALLOWED_POWER_NAMES:
                reasons.append(f"enemy_power_unsupported_v1:{power_name or 'MISSING'}")
            amount = power.get("amount")
            if isinstance(amount, bool) or not isinstance(amount, int):
                reasons.append(f"invalid_enemy_power_amount:{path}.powers[{power_index}]")

    return PublicEnemyAdmission(
        allowed=not reasons,
        reasons=tuple(sorted(set(reasons))),
        enemy_count=enemy_count,
    )


__all__ = ["PublicEnemyAdmission", "assess_public_enemies"]
