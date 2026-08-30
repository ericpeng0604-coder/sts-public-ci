"""Conservative public enemy admission for Phase-1 rollout reconstruction.

V1 intentionally supports only tiny audited surfaces. It is better to mark a
combat unsupported than to reconstruct hidden monster state from guesses.
Hidden previous move history is never accepted as a source input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_SUPPORTED_V1 = frozenset({"JAW_WORM", "CULTIST", "GREMLIN_NOB", "BLUE_SLAVER", "RED_SLAVER", "LOOTER"})
_ALLOWED_POWER_NAMES = frozenset({"STRENGTH", "VULNERABLE", "WEAK", "POISON", "THIEVERY"})
_CULTIST_OPENING_INTENTS = frozenset({"INCANTATION", "CULTIST_INCANTATION"})
_CULTIST_TURN1_INTENTS = frozenset({"DARK_STRIKE", "CULTIST_DARK_STRIKE"})
_NOB_OPENING_INTENTS = frozenset({"BELLOW", "GREMLIN_NOB_BELLOW"})
_BLUE_SLAVER_OPENING_INTENTS = frozenset({"STAB", "BLUE_SLAVER_STAB", "RAKE", "BLUE_SLAVER_RAKE"})
_RED_SLAVER_OPENING_INTENTS = frozenset({"STAB", "RED_SLAVER_STAB"})
_LOOTER_OPENING_INTENTS = frozenset({"MUG", "LOOTER_MUG"})


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
    ascension = state.get("ascension_level")
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
            if name == "CULTIST":
                if turn == 0:
                    if normalized_intent not in _CULTIST_OPENING_INTENTS:
                        reasons.append(f"cultist_opening_intent_mismatch_v1:{normalized_intent}")
                elif turn == 1:
                    if normalized_intent not in _CULTIST_TURN1_INTENTS:
                        reasons.append(f"cultist_turn1_intent_mismatch_v1:{normalized_intent}")
                else:
                    reasons.append("cultist_later_turn_unsupported_v1")
            if name == "GREMLIN_NOB" and turn == 0 and normalized_intent not in _NOB_OPENING_INTENTS:
                reasons.append(f"gremlin_nob_opening_intent_mismatch_v1:{normalized_intent}")
            if name == "BLUE_SLAVER" and turn == 0 and normalized_intent not in _BLUE_SLAVER_OPENING_INTENTS:
                reasons.append(f"blue_slaver_opening_intent_mismatch_v1:{normalized_intent}")
            if name == "RED_SLAVER":
                if turn != 0:
                    reasons.append("red_slaver_later_turn_unsupported_v1")
                elif normalized_intent not in _RED_SLAVER_OPENING_INTENTS:
                    reasons.append(f"red_slaver_opening_intent_mismatch_v1:{normalized_intent}")
            if name == "LOOTER":
                if turn == 0:
                    if normalized_intent not in _LOOTER_OPENING_INTENTS:
                        reasons.append(f"looter_opening_intent_mismatch_v1:{normalized_intent}")
                elif turn == 1:
                    if normalized_intent not in _LOOTER_OPENING_INTENTS:
                        reasons.append(f"looter_turn1_intent_mismatch_v1:{normalized_intent}")
                    gold = state.get("gold")
                    if isinstance(gold, bool) or not isinstance(gold, int) or gold <= 0:
                        reasons.append("looter_turn1_positive_gold_required_v1")
                else:
                    reasons.append("looter_later_turn_unsupported_v1")

        if raw_enemy.get("is_gone") is True:
            reasons.append(f"gone_enemy_unsupported_v1:{path}")

        powers = raw_enemy.get("powers", [])
        if not isinstance(powers, Sequence) or isinstance(powers, str | bytes | bytearray):
            reasons.append(f"enemy_powers_not_sequence:{path}")
            continue
        normalized_powers: list[tuple[str, int | None]] = []
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
                normalized_powers.append((power_name, None))
            else:
                normalized_powers.append((power_name, amount))

        if name == "CULTIST" and turn == 1:
            if isinstance(ascension, bool) or not isinstance(ascension, int):
                reasons.append("cultist_turn1_missing_public_ascension_v1")
            # The current public adapter intentionally does not expose Ritual.
            # For this pristine boundary, no player card was played on turn 0,
            # so all projected enemy powers must be empty. Ritual itself is
            # reconstructed from public ascension + the source-proven fixed
            # Incantation opener, with no new contract field.
            if normalized_powers:
                reasons.append("cultist_turn1_requires_no_projected_enemy_powers_v1")
            hp = raw_enemy.get("hp")
            max_hp = raw_enemy.get("max_hp")
            block = raw_enemy.get("block")
            if type(hp) is int and type(max_hp) is int and hp != max_hp:
                reasons.append("cultist_turn1_requires_pristine_enemy_hp_v1")
            if type(block) is int and block != 0:
                reasons.append("cultist_turn1_requires_zero_enemy_block_v1")

        if name == "LOOTER" and turn in (0, 1):
            if isinstance(ascension, bool) or not isinstance(ascension, int):
                reasons.append("looter_missing_public_ascension_v1")
            else:
                expected_thievery = 20 if ascension >= 17 else 15
                # The pinned simulator binding currently omits Thievery from its
                # public monster projection. This is still safely reconstructible:
                # preBattleAction derives it solely from public ascension level.
                # If a source does expose the public power, it must match exactly.
                if normalized_powers not in ([], [("THIEVERY", expected_thievery)]):
                    reasons.append(
                        f"looter_opening_thievery_mismatch_v1:expected_{expected_thievery}"
                    )

    return PublicEnemyAdmission(
        allowed=not reasons,
        reasons=tuple(sorted(set(reasons))),
        enemy_count=enemy_count,
    )


__all__ = ["PublicEnemyAdmission", "assess_public_enemies"]
