#!/usr/bin/env python3
"""Layer audited draw-card midturn reconstruction over frozen V1/V2.

This overlay is intentionally additive and runs *after* the existing public
reconstruction + history + starter-midturn overlays. Existing two-argument
calls keep the old behavior. A third reconstruction-aux argument unlocks only
audited Jaw Worm turn=1 slices: one normal Pommel Strike, or one Shrug It Off
at upgrade count 0 or 1.

The auxiliary data contains controller-observed turn counters only. It is not
part of sts1-public-state-v1 and never contains RNG state, draw order, UUIDs,
or a replayable action history. Shrug upgrade count is read from the current
public card instance, not from the auxiliary trace.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
MARKER = "phase1_public_draw_midturn_aux_v2"

SIG_OLD = '''    m.def("build_public_jaw_worm_context_v1",
          [](pybind11::dict state, pybind11::dict seeds) {'''
SIG_NEW = '''    m.def("build_public_jaw_worm_context_v1",
          [](pybind11::dict state, pybind11::dict seeds, pybind11::object reconstructionAuxObj) {'''

CARD_OLD = '''                  else if (idText == "BASH") { id = CardId::BASH; expectedCost = 2; }
                  else throw std::runtime_error("unsupported_card_v1:" + idText);'''
CARD_NEW = '''                  else if (idText == "BASH") { id = CardId::BASH; expectedCost = 2; }
                  else if (idText == "POMMEL_STRIKE") { id = CardId::POMMEL_STRIKE; expectedCost = 1; }
                  else if (idText == "SHRUG_IT_OFF") { id = CardId::SHRUG_IT_OFF; expectedCost = 1; }
                  else throw std::runtime_error("unsupported_card_v1:" + idText);'''

MIDTURN_OLD = '''              // phase1_public_jaw_worm_turn_one_midturn_counters_v1:
              // derive turn-local bookkeeping from public state only.
              if (bc.turn == 1) {'''

MIDTURN_NEW = r'''              // phase1_public_jaw_worm_turn_one_midturn_counters_v1:
              // derive turn-local bookkeeping from public state only.
              // phase1_public_pommel_midturn_aux_v1:
              // phase1_public_shrug_midturn_aux_v1:
              // phase1_public_shrug_upgraded_midturn_aux_v1:
              // phase1_public_draw_midturn_aux_v2: an optional, separate
              // controller trace can prove counters that snapshot arithmetic
              // cannot recover after a draw effect.
              const bool hasReconstructionAux = !reconstructionAuxObj.is_none();
              if (hasReconstructionAux && bc.turn == 1) {
                  if (!pybind11::isinstance<pybind11::dict>(reconstructionAuxObj)) {
                      throw std::runtime_error("draw_aux_not_mapping");
                  }
                  const auto aux = pybind11::cast<pybind11::dict>(reconstructionAuxObj);
                  auto aux_int = [&](const char *key) -> int {
                      if (!aux.contains(key)
                          || !pybind11::isinstance<pybind11::int_>(aux[key])
                          || pybind11::isinstance<pybind11::bool_>(aux[key])) {
                          throw std::runtime_error(std::string("draw_aux_invalid_int:") + key);
                      }
                      return pybind11::cast<int>(aux[key]);
                  };
                  auto aux_string = [&](const char *key) -> std::string {
                      if (!aux.contains(key) || !pybind11::isinstance<pybind11::str>(aux[key])) {
                          throw std::runtime_error(std::string("draw_aux_invalid_string:") + key);
                      }
                      return pybind11::cast<std::string>(aux[key]);
                  };

                  const int played = aux_int("cards_played_this_turn");
                  const int attacks = aux_int("attacks_played_this_turn");
                  const int skills = aux_int("skills_played_this_turn");
                  const int discarded = aux_int("cards_discarded_this_turn");
                  const bool pommelSlice = played == 1 && attacks == 1 && skills == 0 && discarded == 0;
                  const bool shrugSlice = played == 1 && attacks == 0 && skills == 1 && discarded == 0;
                  if (!pommelSlice && !shrugSlice) {
                      throw std::runtime_error("draw_aux_counter_slice_unsupported");
                  }
                  const char *prefix = pommelSlice ? "pommel" : "shrug";
                  auto slice_error = [&](const char *suffix) -> std::runtime_error {
                      return std::runtime_error(std::string(prefix) + suffix);
                  };

                  if (!aux.contains("complete")
                      || !pybind11::isinstance<pybind11::bool_>(aux["complete"])
                      || !pybind11::cast<bool>(aux["complete"])) {
                      throw slice_error("_aux_incomplete");
                  }
                  if (aux_string("schema_version") != "sts1-public-reconstruction-aux-v1") {
                      throw slice_error("_aux_schema_mismatch");
                  }
                  if (aux_string("source") != "communicationmod_command_trace_v1") {
                      throw slice_error("_aux_source_mismatch");
                  }
                  if (aux_int("turn") != bc.turn) {
                      throw slice_error("_aux_turn_mismatch");
                  }

                  if (bc.cards.cardsInHand != 5
                      || !bc.cards.drawPile.empty()
                      || bc.cards.discardPile.size() != 6
                      || !bc.cards.exhaustPile.empty()) {
                      throw slice_error("_midturn_pile_shape_mismatch");
                  }
                  if (bc.player.energy != 2) {
                      throw slice_error("_midturn_player_shape_mismatch");
                  }

                  int strikes = 0;
                  int defends = 0;
                  int bashes = 0;
                  int pommels = 0;
                  int shrugs = 0;
                  int unsupported = 0;
                  int upgradedShrugs = 0;
                  int upgradedOther = 0;
                  auto count_draw_slice_card = [&](const CardInstance &c) {
                      if (c.upgraded) {
                          if (c.id == CardId::SHRUG_IT_OFF) ++upgradedShrugs;
                          else ++upgradedOther;
                      }
                      if (c.id == CardId::STRIKE_RED) ++strikes;
                      else if (c.id == CardId::DEFEND_RED) ++defends;
                      else if (c.id == CardId::BASH) ++bashes;
                      else if (c.id == CardId::POMMEL_STRIKE) ++pommels;
                      else if (c.id == CardId::SHRUG_IT_OFF) ++shrugs;
                      else ++unsupported;
                  };
                  for (int i = 0; i < bc.cards.cardsInHand; ++i) count_draw_slice_card(bc.cards.hand[i]);
                  for (const auto &c : bc.cards.drawPile) count_draw_slice_card(c);
                  for (const auto &c : bc.cards.discardPile) count_draw_slice_card(c);
                  for (const auto &c : bc.cards.exhaustPile) count_draw_slice_card(c);

                  if (strikes != 5 || defends != 4 || bashes != 1
                      || pommels != (pommelSlice ? 1 : 0)
                      || shrugs != (shrugSlice ? 1 : 0)
                      || unsupported != 0 || upgradedOther != 0
                      || (pommelSlice && upgradedShrugs != 0)
                      || (shrugSlice && (upgradedShrugs < 0 || upgradedShrugs > 1))) {
                      throw slice_error("_midturn_deck_composition_mismatch");
                  }

                  const int expectedBlock = pommelSlice ? 0 : (upgradedShrugs == 1 ? 11 : 8);
                  if (bc.player.block != expectedBlock) {
                      throw slice_error("_midturn_player_shape_mismatch");
                  }

                  int discardPlayedCard = 0;
                  int discardUpgradedShrugs = 0;
                  for (const auto &c : bc.cards.discardPile) {
                      if ((pommelSlice && c.id == CardId::POMMEL_STRIKE)
                          || (shrugSlice && c.id == CardId::SHRUG_IT_OFF)) {
                          ++discardPlayedCard;
                      }
                      if (c.id == CardId::SHRUG_IT_OFF && c.upgraded) {
                          ++discardUpgradedShrugs;
                      }
                  }
                  if (discardPlayedCard != 1
                      || (shrugSlice && discardUpgradedShrugs != upgradedShrugs)) {
                      throw slice_error("_midturn_play_not_publicly_reachable");
                  }

                  bc.player.cardsPlayedThisTurn = 1;
                  bc.player.attacksPlayedThisTurn = pommelSlice ? 1 : 0;
                  bc.player.skillsPlayedThisTurn = shrugSlice ? 1 : 0;
                  bc.player.cardsDiscardedThisTurn = 0;
              } else if (bc.turn == 1) {'''

ARGS_OLD = '''          pybind11::arg("public_state"), pybind11::arg("sample_seeds"),
          "construct a new Jaw Worm BattleContext from public state and fresh sample seeds only");'''
ARGS_NEW = '''          pybind11::arg("public_state"), pybind11::arg("sample_seeds"),
          pybind11::arg("reconstruction_aux") = pybind11::none(),
          "construct a new Jaw Worm BattleContext from public state and fresh sample seeds only");'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"unexpected {label} anchor count: {count}")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"missing hydrated binding: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        raise SystemExit("draw-midturn aux overlay already applied")
    text = replace_once(text, SIG_OLD, SIG_NEW, "signature")
    text = replace_once(text, CARD_OLD, CARD_NEW, "card parser")
    text = replace_once(text, MIDTURN_OLD, MIDTURN_NEW, "midturn")
    text = replace_once(text, ARGS_OLD, ARGS_NEW, "argument")
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched={TARGET}")
    print("draw_midturn_aux=POMMEL_STRIKE_V1+SHRUG_IT_OFF_V1+SHRUG_IT_OFF_UPGRADED_V1")
    print("existing_two_argument_behavior=PRESERVED")
    print("public_contract_fields_added=0")
    print("hidden_rng_access_added=0")


if __name__ == "__main__":
    main()
