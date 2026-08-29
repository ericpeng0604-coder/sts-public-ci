#!/usr/bin/env python3
"""Make V1 monster reconstruction turn-aware and add audited Cultist opening support.

This overlay is applied after the base public BattleContext constructor. It
keeps Jaw Worm's sampled prior history fail-closed and extends the same
public-only constructor to Cultist. Cultist is admitted only on the opening
player turn, where its public intent must be INCANTATION and previous history
is necessarily INVALID.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
MARKER = "phase1_public_history_turn_guard_v1"
CULTIST_MARKER = "phase1_public_cultist_opening_v1"

HISTORY_ANCHOR = '''              const auto historyChoice = get_u64(seeds, "previous_history") % 3ULL;
              mo.moveHistory[1] = historyChoice == 0 ? MMID::JAW_WORM_CHOMP
                                : historyChoice == 1 ? MMID::JAW_WORM_THRASH
                                                     : MMID::JAW_WORM_BELLOW;'''

HISTORY_REPLACEMENT = '''              // phase1_public_history_turn_guard_v1:
              // The pinned simulator is zero-based: turn 0 is the first player turn,
              // so no older monster move exists there. Later Jaw Worm turns sample it.
              if (bc.turn <= 0) {
                  mo.moveHistory[1] = MMID::INVALID;
              } else if (enemyName == "JAW_WORM") {
                  const auto historyChoice = get_u64(seeds, "previous_history") % 3ULL;
                  mo.moveHistory[1] = historyChoice == 0 ? MMID::JAW_WORM_CHOMP
                                    : historyChoice == 1 ? MMID::JAW_WORM_THRASH
                                                         : MMID::JAW_WORM_BELLOW;
              } else {
                  throw std::runtime_error("cultist_later_turn_unsupported_v1");
              }'''

ENEMY_NAME_ANCHOR = '''              if (normalized(get_s(enemy, "name")) != "JAW_WORM") throw std::runtime_error("enemy_not_jaw_worm_v1");'''

ENEMY_NAME_REPLACEMENT = '''              // phase1_public_cultist_opening_v1: second audited single-enemy surface.
              const auto enemyName = normalized(get_s(enemy, "name"));
              if (enemyName != "JAW_WORM" && enemyName != "CULTIST") {
                  throw std::runtime_error("enemy_unsupported_single_v1:" + enemyName);
              }
              bc.encounter = enemyName == "CULTIST" ? MonsterEncounter::CULTIST : MonsterEncounter::JAW_WORM;'''

ENEMY_ID_ANCHOR = '''              mo.id = MonsterId::JAW_WORM;'''
ENEMY_ID_REPLACEMENT = '''              mo.id = enemyName == "CULTIST" ? MonsterId::CULTIST : MonsterId::JAW_WORM;'''

ENEMY_MOVE_ANCHOR = '''              mo.moveHistory[0] = jaw_worm_move(get_s(enemy, "intent"));'''
ENEMY_MOVE_REPLACEMENT = '''              const auto publicIntent = normalized(get_s(enemy, "intent"));
              if (enemyName == "JAW_WORM") {
                  mo.moveHistory[0] = jaw_worm_move(publicIntent);
              } else {
                  // Cultist V1 is opening-turn only. The pinned simulator maps
                  // INVALID prior history deterministically to INCANTATION.
                  if (bc.turn != 0) throw std::runtime_error("cultist_later_turn_unsupported_v1");
                  if (publicIntent != "INCANTATION" && publicIntent != "CULTIST_INCANTATION") {
                      throw std::runtime_error("cultist_opening_intent_mismatch_v1:" + publicIntent);
                  }
                  mo.moveHistory[0] = MMID::CULTIST_INCANTATION;
              }'''


def replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise SystemExit(f"unexpected {label} anchor count: {count}")
    return text.replace(anchor, replacement, 1)


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"missing hydrated binding: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text or CULTIST_MARKER in text:
        raise SystemExit("public history/cultist overlay already applied")

    text = replace_once(text, ENEMY_NAME_ANCHOR, ENEMY_NAME_REPLACEMENT, "enemy-name")
    text = replace_once(text, ENEMY_ID_ANCHOR, ENEMY_ID_REPLACEMENT, "enemy-id")
    text = replace_once(text, ENEMY_MOVE_ANCHOR, ENEMY_MOVE_REPLACEMENT, "enemy-move")
    text = replace_once(text, HISTORY_ANCHOR, HISTORY_REPLACEMENT, "history-guard")

    TARGET.write_text(text, encoding="utf-8")
    print(f"patched={TARGET}")
    print("opening_turn_zero_previous_history=INVALID")
    print("later_jaw_worm_previous_history=PUBLIC_SAMPLE")
    print("cultist_opening_incantation_only=1")
    print("cultist_later_turn=FAIL_CLOSED")
    print("source_move_history_access=0")


if __name__ == "__main__":
    main()
