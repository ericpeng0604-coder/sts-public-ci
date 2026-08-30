#!/usr/bin/env python3
"""Extend the hydrated public constructor with the audited Cultist turn-1 boundary.

This overlay runs after the existing public history guard. It changes only
Cultist turn 1: previous move is source-proven Incantation, current move is
source-proven Dark Strike, and Ritual is derived solely from public ascension.
No source BattleContext, RNG, move history, counter, or miscInfo is read.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
MARKER = "phase1_public_cultist_turn1_v1"

HISTORY_ANCHOR = '''              } else if (enemyName == "CULTIST") {
                  throw std::runtime_error("cultist_later_turn_unsupported_v1");
              } else if (enemyName == "GREMLIN_NOB") {'''

HISTORY_REPLACEMENT = '''              } else if (enemyName == "CULTIST") {
                  if (bc.turn == 1) {
                      // phase1_public_cultist_turn1_v1: the opening Cultist move
                      // is unconditionally INCANTATION in the pinned source.
                      mo.moveHistory[1] = MMID::CULTIST_INCANTATION;
                  } else {
                      throw std::runtime_error("cultist_later_turn_unsupported_v1");
                  }
              } else if (enemyName == "GREMLIN_NOB") {'''

MOVE_ANCHOR = '''              } else if (enemyName == "CULTIST") {
                  if (bc.turn != 0) throw std::runtime_error("cultist_later_turn_unsupported_v1");
                  if (publicIntent != "INCANTATION" && publicIntent != "CULTIST_INCANTATION") {
                      throw std::runtime_error("cultist_opening_intent_mismatch_v1:" + publicIntent);
                  }
                  mo.moveHistory[0] = MMID::CULTIST_INCANTATION;
              } else if (enemyName == "GREMLIN_NOB") {'''

MOVE_REPLACEMENT = '''              } else if (enemyName == "CULTIST") {
                  if (bc.turn == 0) {
                      if (publicIntent != "INCANTATION" && publicIntent != "CULTIST_INCANTATION") {
                          throw std::runtime_error("cultist_opening_intent_mismatch_v1:" + publicIntent);
                      }
                      mo.moveHistory[0] = MMID::CULTIST_INCANTATION;
                  } else if (bc.turn == 1) {
                      if (publicIntent != "DARK_STRIKE" && publicIntent != "CULTIST_DARK_STRIKE") {
                          throw std::runtime_error("cultist_turn1_intent_mismatch_v1:" + publicIntent);
                      }
                      mo.moveHistory[0] = MMID::CULTIST_DARK_STRIKE;
                      // Pinned source Incantation applies hallway Ritual 3/4/5
                      // for A0-1 / A2-16 / A17+. At the first end-of-round the
                      // source clears Ritual's justApplied bit, so public turn 1
                      // has Ritual active with justApplied=false and Strength=0.
                      const int expectedRitual = bc.ascension >= 17 ? 5 : (bc.ascension >= 2 ? 4 : 3);
                      mo.setHasStatus<MonsterStatus::RITUAL>(true);
                      mo.setStatus<MonsterStatus::RITUAL>(expectedRitual);
                      mo.setJustApplied<MonsterStatus::RITUAL>(false);
                  } else {
                      throw std::runtime_error("cultist_later_turn_unsupported_v1");
                  }
              } else if (enemyName == "GREMLIN_NOB") {'''

PROOF_ANCHOR = '''    m.def("get_legal_actions", &sts::py::getLegalActions,
          "enumerate all legal actions in the current battle state for forward search");'''

PROOF_INSERT = '''    // phase1_public_cultist_turn1_v1: read-only proof of a public-derived
    // gameplay power. These helpers expose no RNG, history, counter, or miscInfo.
    m.def("get_public_monster_ritual_v1",
          [](const BattleContext &bc, int idx) {
              if (idx < 0 || idx >= bc.monsters.monsterCount) throw std::runtime_error("monster_index_out_of_range");
              return bc.monsters.arr[idx].getStatus<MonsterStatus::RITUAL>();
          },
          pybind11::arg("battle"), pybind11::arg("monster_index"),
          "read reconstructed Ritual amount for native verification");
    m.def("get_public_monster_ritual_just_applied_v1",
          [](const BattleContext &bc, int idx) {
              if (idx < 0 || idx >= bc.monsters.monsterCount) throw std::runtime_error("monster_index_out_of_range");
              return bc.monsters.arr[idx].wasJustApplied<MonsterStatus::RITUAL>();
          },
          pybind11::arg("battle"), pybind11::arg("monster_index"),
          "read reconstructed Ritual justApplied flag for native verification");

'''


def replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise SystemExit(f"unexpected {label} anchor count: {count}")
    return text.replace(anchor, replacement, 1)


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"missing hydrated binding: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        raise SystemExit("cultist turn1 overlay already applied")
    text = replace_once(text, HISTORY_ANCHOR, HISTORY_REPLACEMENT, "cultist-history")
    text = replace_once(text, MOVE_ANCHOR, MOVE_REPLACEMENT, "cultist-move")
    text = replace_once(text, PROOF_ANCHOR, PROOF_INSERT + PROOF_ANCHOR, "cultist-proof")
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched={TARGET}")
    print("cultist_turn1_previous_move=PUBLIC_DERIVED_INCANTATION")
    print("cultist_turn1_current_move=PUBLIC_DERIVED_DARK_STRIKE")
    print("cultist_turn1_ritual=PUBLIC_ASCENSION_DERIVED")
    print("cultist_turn1_ritual_just_applied=PUBLIC_TURN_DERIVED_FALSE")
    print("source_raw_battle_context_input=0")
    print("source_hidden_rng_access=0")
    print("source_move_history_access=0")
    print("source_counter_history_access=0")
    print("source_hidden_miscinfo_access=0")


if __name__ == "__main__":
    main()
