#!/usr/bin/env python3
"""Add fail-closed public-derived per-turn counters for the Jaw Worm turn-1 slice.

This overlay runs after ``apply_phase1_public_reconstruction_overlay.py``.  It
never reads a source BattleContext.  For the audited starter-only second-player-
turn slice it derives simulator bookkeeping from current public state:

cards played = 5 - hand size
skills played = block / 5
attacks played = cards played - skills played
cards discarded = 0

The overlay also mirrors the Python admission reachability checks at the native
constructor boundary.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
MARKER = "phase1_public_jaw_worm_turn_one_midturn_counters_v1"

ANCHOR = '''              add_cards("hand");
              add_cards("draw_pile");
              add_cards("discard_pile");
              add_cards("exhaust_pile");
'''

INSERT = r'''              add_cards("hand");
              add_cards("draw_pile");
              add_cards("discard_pile");
              add_cards("exhaust_pile");

              // phase1_public_jaw_worm_turn_one_midturn_counters_v1:
              // derive turn-local bookkeeping from public state only.
              if (bc.turn == 1) {
                  if (bc.cards.drawPile.size() != 0 || bc.cards.exhaustPile.size() != 0) {
                      throw std::runtime_error("turn1_midturn_nonempty_draw_or_exhaust");
                  }
                  if (bc.cards.cardsInHand < 2 || bc.cards.cardsInHand > 5) {
                      throw std::runtime_error("turn1_midturn_hand_size_unreachable");
                  }
                  if (bc.cards.discardPile.size() != static_cast<std::size_t>(10 - bc.cards.cardsInHand)) {
                      throw std::runtime_error("turn1_midturn_discard_size_unreachable");
                  }

                  int strikes = 0;
                  int defends = 0;
                  int bashes = 0;
                  auto count_card = [&](const CardInstance &c) {
                      if (c.id == CardId::STRIKE_RED) ++strikes;
                      else if (c.id == CardId::DEFEND_RED) ++defends;
                      else if (c.id == CardId::BASH) ++bashes;
                  };
                  for (int i = 0; i < bc.cards.cardsInHand; ++i) count_card(bc.cards.hand[i]);
                  for (const auto &c : bc.cards.discardPile) count_card(c);
                  if (strikes != 5 || defends != 4 || bashes != 1) {
                      throw std::runtime_error("turn1_midturn_starter_composition_mismatch");
                  }

                  if (bc.player.energy < 0 || bc.player.energy > 3) {
                      throw std::runtime_error("turn1_midturn_energy_unreachable");
                  }
                  if (bc.player.block < 0 || bc.player.block % 5 != 0) {
                      throw std::runtime_error("turn1_midturn_block_unreachable");
                  }

                  const int played = 5 - bc.cards.cardsInHand;
                  const int skills = bc.player.block / 5;
                  const int attacks = played - skills;
                  const int spent = 3 - bc.player.energy;
                  if (skills < 0 || attacks < 0 || spent > 3 || (spent != played && spent != played + 1)) {
                      throw std::runtime_error("turn1_midturn_counter_mix_unreachable");
                  }

                  int discardStrikes = 0;
                  int discardDefends = 0;
                  int discardBashes = 0;
                  for (const auto &c : bc.cards.discardPile) {
                      if (c.id == CardId::STRIKE_RED) ++discardStrikes;
                      else if (c.id == CardId::DEFEND_RED) ++discardDefends;
                      else if (c.id == CardId::BASH) ++discardBashes;
                  }
                  if (skills > discardDefends) {
                      throw std::runtime_error("turn1_midturn_skill_history_unreachable");
                  }
                  if (spent == played + 1) {
                      if (attacks < 1 || discardBashes < 1 || attacks - 1 > discardStrikes) {
                          throw std::runtime_error("turn1_midturn_bash_history_unreachable");
                      }
                  } else if (attacks > discardStrikes) {
                      throw std::runtime_error("turn1_midturn_attack_history_unreachable");
                  }

                  bc.player.cardsPlayedThisTurn = played;
                  bc.player.skillsPlayedThisTurn = skills;
                  bc.player.attacksPlayedThisTurn = attacks;
                  bc.player.cardsDiscardedThisTurn = 0;
              } else if (bc.turn != 0) {
                  throw std::runtime_error("turn_unsupported_public_reconstruction_v1");
              }
'''


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"missing hydrated binding: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        raise SystemExit("midturn counter overlay already applied")
    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(f"unexpected midturn counter anchor count: {count}")
    TARGET.write_text(text.replace(ANCHOR, INSERT, 1), encoding="utf-8")
    print(f"patched={TARGET}")
    print("turn1_midturn_counters=PUBLIC_DERIVED")
    print("source_cards_played_counter_access=0")
    print("source_attack_counter_access=0")
    print("source_skill_counter_access=0")
    print("public_contract_fields_added=0")


if __name__ == "__main__":
    main()
