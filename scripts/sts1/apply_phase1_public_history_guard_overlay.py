#!/usr/bin/env python3
"""Make V1 monster reconstruction turn-aware for audited opening enemies.

Applied after the base public BattleContext constructor. Jaw Worm later history
is sampled; Cultist, Gremlin Nob, and Blue Slaver are admitted only from opening
public observations. No source move history is read.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
MARKER = "phase1_public_history_turn_guard_v1"
CULTIST_MARKER = "phase1_public_cultist_opening_v1"
NOB_MARKER = "phase1_public_gremlin_nob_opening_v1"
BLUE_SLAVER_MARKER = "phase1_public_blue_slaver_opening_v1"

HISTORY_ANCHOR = '''              const auto historyChoice = get_u64(seeds, "previous_history") % 3ULL;
              mo.moveHistory[1] = historyChoice == 0 ? MMID::JAW_WORM_CHOMP
                                : historyChoice == 1 ? MMID::JAW_WORM_THRASH
                                                     : MMID::JAW_WORM_BELLOW;'''

HISTORY_REPLACEMENT = '''              // phase1_public_history_turn_guard_v1:
              // The pinned simulator is zero-based: turn 0 is the first player turn.
              if (bc.turn <= 0) {
                  mo.moveHistory[1] = MMID::INVALID;
              } else if (enemyName == "JAW_WORM") {
                  const auto historyChoice = get_u64(seeds, "previous_history") % 3ULL;
                  mo.moveHistory[1] = historyChoice == 0 ? MMID::JAW_WORM_CHOMP
                                    : historyChoice == 1 ? MMID::JAW_WORM_THRASH
                                                         : MMID::JAW_WORM_BELLOW;
              } else if (enemyName == "CULTIST") {
                  throw std::runtime_error("cultist_later_turn_unsupported_v1");
              } else if (enemyName == "GREMLIN_NOB") {
                  throw std::runtime_error("gremlin_nob_later_turn_unsupported_v1");
              } else {
                  throw std::runtime_error("blue_slaver_later_turn_unsupported_v1");
              }'''

ENEMY_NAME_ANCHOR = '''              if (normalized(get_s(enemy, "name")) != "JAW_WORM") throw std::runtime_error("enemy_not_jaw_worm_v1");'''

ENEMY_NAME_REPLACEMENT = '''              // audited single-enemy opening surfaces only.
              const auto enemyName = normalized(get_s(enemy, "name"));
              if (enemyName != "JAW_WORM" && enemyName != "CULTIST" && enemyName != "GREMLIN_NOB" && enemyName != "BLUE_SLAVER") {
                  throw std::runtime_error("enemy_unsupported_single_v1:" + enemyName);
              }
              bc.encounter = enemyName == "CULTIST" ? MonsterEncounter::CULTIST
                           : enemyName == "GREMLIN_NOB" ? MonsterEncounter::GREMLIN_NOB
                           : enemyName == "BLUE_SLAVER" ? MonsterEncounter::BLUE_SLAVER
                                                         : MonsterEncounter::JAW_WORM;'''

ENEMY_ID_ANCHOR = '''              mo.id = MonsterId::JAW_WORM;'''
ENEMY_ID_REPLACEMENT = '''              mo.id = enemyName == "CULTIST" ? MonsterId::CULTIST
                    : enemyName == "GREMLIN_NOB" ? MonsterId::GREMLIN_NOB
                    : enemyName == "BLUE_SLAVER" ? MonsterId::BLUE_SLAVER
                                                   : MonsterId::JAW_WORM;'''

ENEMY_MOVE_ANCHOR = '''              mo.moveHistory[0] = jaw_worm_move(get_s(enemy, "intent"));'''
ENEMY_MOVE_REPLACEMENT = '''              const auto publicIntent = normalized(get_s(enemy, "intent"));
              if (enemyName == "JAW_WORM") {
                  mo.moveHistory[0] = jaw_worm_move(publicIntent);
              } else if (enemyName == "CULTIST") {
                  if (bc.turn != 0) throw std::runtime_error("cultist_later_turn_unsupported_v1");
                  if (publicIntent != "INCANTATION" && publicIntent != "CULTIST_INCANTATION") {
                      throw std::runtime_error("cultist_opening_intent_mismatch_v1:" + publicIntent);
                  }
                  mo.moveHistory[0] = MMID::CULTIST_INCANTATION;
              } else if (enemyName == "GREMLIN_NOB") {
                  if (bc.turn != 0) throw std::runtime_error("gremlin_nob_later_turn_unsupported_v1");
                  if (publicIntent != "BELLOW" && publicIntent != "GREMLIN_NOB_BELLOW") {
                      throw std::runtime_error("gremlin_nob_opening_intent_mismatch_v1:" + publicIntent);
                  }
                  mo.moveHistory[0] = MMID::GREMLIN_NOB_BELLOW;
              } else {
                  // phase1_public_blue_slaver_opening_v1: the source opening move
                  // may be STAB or RAKE, but the chosen current intent is public.
                  // Reconstruct that intent exactly; do not copy the source roll.
                  if (bc.turn != 0) throw std::runtime_error("blue_slaver_later_turn_unsupported_v1");
                  if (publicIntent == "STAB" || publicIntent == "BLUE_SLAVER_STAB") {
                      mo.moveHistory[0] = MMID::BLUE_SLAVER_STAB;
                  } else if (publicIntent == "RAKE" || publicIntent == "BLUE_SLAVER_RAKE") {
                      mo.moveHistory[0] = MMID::BLUE_SLAVER_RAKE;
                  } else {
                      throw std::runtime_error("blue_slaver_opening_intent_mismatch_v1:" + publicIntent);
                  }
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
    if MARKER in text or CULTIST_MARKER in text or NOB_MARKER in text or BLUE_SLAVER_MARKER in text:
        raise SystemExit("public history/opening-enemy overlay already applied")

    text = replace_once(text, ENEMY_NAME_ANCHOR, ENEMY_NAME_REPLACEMENT, "enemy-name")
    text = replace_once(text, ENEMY_ID_ANCHOR, ENEMY_ID_REPLACEMENT, "enemy-id")
    text = replace_once(text, ENEMY_MOVE_ANCHOR, ENEMY_MOVE_REPLACEMENT, "enemy-move")
    text = replace_once(text, HISTORY_ANCHOR, HISTORY_REPLACEMENT, "history-guard")

    # Marker is intentionally embedded in the patched binding for CI scope proof.
    text = text.replace(
        "// audited single-enemy opening surfaces only.",
        "// audited single-enemy opening surfaces only. phase1_public_blue_slaver_opening_v1",
        1,
    )
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched={TARGET}")
    print("opening_turn_zero_previous_history=INVALID")
    print("later_jaw_worm_previous_history=PUBLIC_SAMPLE")
    print("cultist_opening_incantation_only=1")
    print("cultist_later_turn=FAIL_CLOSED")
    print("gremlin_nob_opening_bellow_only=1")
    print("gremlin_nob_later_turn=FAIL_CLOSED")
    print("blue_slaver_opening_public_intent_only=1")
    print("blue_slaver_later_turn=FAIL_CLOSED")
    print("source_move_history_access=0")
    print("source_opening_roll_access=0")


if __name__ == "__main__":
    main()
