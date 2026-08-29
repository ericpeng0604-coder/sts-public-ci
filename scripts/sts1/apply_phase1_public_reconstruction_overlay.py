#!/usr/bin/env python3
"""Add the first fail-closed public-state -> BattleContext constructor.

V1 is intentionally tiny: Ironclad, one Jaw Worm, starter Strike/Defend/Bash,
empty potion slots, and no relic other than Burning Blood.  The constructor
accepts only a public-state dict plus caller-supplied re-determinization seeds;
it never accepts or clones an existing BattleContext.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
MARKER = "phase1_public_reconstruction_v1"

ANCHOR = '''    m.def("get_legal_actions", &sts::py::getLegalActions,
          "enumerate all legal actions in the current battle state for forward search");'''

INSERT = r'''    // phase1_public_reconstruction_v1: build from public observation + fresh sample seeds only.
    m.def("build_public_jaw_worm_context_v1",
          [](pybind11::dict state, pybind11::dict seeds) {
              auto get_i = [](const pybind11::dict &d, const char *key) -> int {
                  if (!d.contains(key)) throw std::runtime_error(std::string("missing_int:") + key);
                  return pybind11::cast<int>(d[key]);
              };
              auto get_u64 = [](const pybind11::dict &d, const char *key) -> std::uint64_t {
                  if (!d.contains(key)) throw std::runtime_error(std::string("missing_seed:") + key);
                  return pybind11::cast<std::uint64_t>(d[key]);
              };
              auto get_s = [](const pybind11::dict &d, const char *key) -> std::string {
                  if (!d.contains(key)) throw std::runtime_error(std::string("missing_string:") + key);
                  return pybind11::cast<std::string>(d[key]);
              };
              auto require_bool = [](const pybind11::dict &d, const char *key, bool expected) {
                  if (!d.contains(key) || pybind11::cast<bool>(d[key]) != expected) {
                      throw std::runtime_error(std::string("unexpected_bool:") + key);
                  }
              };
              auto normalized = [](std::string value) {
                  for (auto &c : value) {
                      if (c == ' ' || c == '-') c = '_';
                      else c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                  }
                  return value;
              };
              auto card_from_public = [&](const pybind11::dict &d) {
                  const auto idText = normalized(get_s(d, "id"));
                  const int upgrades = get_i(d, "upgrades");
                  const int publicCost = get_i(d, "cost");
                  if (upgrades < 0 || upgrades > 1) throw std::runtime_error("unsupported_upgrade_count");
                  CardId id = CardId::INVALID;
                  int expectedCost = -999;
                  if (idText == "STRIKE_RED") { id = CardId::STRIKE_RED; expectedCost = 1; }
                  else if (idText == "DEFEND_RED") { id = CardId::DEFEND_RED; expectedCost = 1; }
                  else if (idText == "BASH") { id = CardId::BASH; expectedCost = 2; }
                  else throw std::runtime_error("unsupported_card_v1:" + idText);
                  if (publicCost != expectedCost) throw std::runtime_error("temporary_or_unknown_card_cost_v1:" + idText);
                  CardInstance card(id, upgrades == 1);
                  if (card.cost != expectedCost) throw std::runtime_error("native_card_cost_mismatch:" + idText);
                  card.costForTurn = static_cast<std::int8_t>(expectedCost);
                  return card;
              };
              auto jaw_worm_move = [&](std::string text) {
                  text = normalized(text);
                  if (text == "CHOMP" || text == "JAW_WORM_CHOMP") return MMID::JAW_WORM_CHOMP;
                  if (text == "THRASH" || text == "JAW_WORM_THRASH") return MMID::JAW_WORM_THRASH;
                  if (text == "BELLOW" || text == "JAW_WORM_BELLOW") return MMID::JAW_WORM_BELLOW;
                  throw std::runtime_error("unsupported_jaw_worm_intent_v1:" + text);
              };

              if (normalized(get_s(state, "character")) != "IRONCLAD") throw std::runtime_error("character_not_ironclad_v1");
              if (normalized(get_s(state, "room")) != "COMBAT") throw std::runtime_error("room_not_combat_v1");
              require_bool(state, "combat_active", true);

              BattleContext bc;
              bc.seed = 0;  // Never copied from the source simulator; formal sampling uses the explicit RNGs below.
              bc.floorNum = get_i(state, "floor");
              bc.encounter = MonsterEncounter::JAW_WORM;
              bc.ascension = get_i(state, "ascension_level");
              bc.outcome = Outcome::UNDECIDED;
              bc.inputState = InputState::PLAYER_NORMAL;
              bc.turn = get_i(state, "turn");
              bc.monsterTurnIdx = 6;
              bc.isBattleOver = false;
              bc.endTurnQueued = false;
              bc.turnHasEnded = false;
              bc.skipMonsterTurn = false;
              bc.miscBits.reset();

              bc.aiRng = Random(get_u64(seeds, "ai"));
              bc.cardRandomRng = Random(get_u64(seeds, "card_random"));
              bc.miscRng = Random(get_u64(seeds, "misc"));
              bc.monsterHpRng = Random(get_u64(seeds, "monster_hp"));
              bc.potionRng = Random(get_u64(seeds, "potion"));
              bc.shuffleRng = Random(get_u64(seeds, "shuffle"));

              bc.player.cc = CharacterClass::IRONCLAD;
              bc.player.curHp = get_i(state, "hp");
              bc.player.maxHp = get_i(state, "max_hp");
              bc.player.block = get_i(state, "block");
              bc.player.energy = get_i(state, "energy");
              bc.player.gold = get_i(state, "gold");
              bc.player.energyPerTurn = 3;
              bc.player.cardDrawPerTurn = 5;

              auto playerPowers = pybind11::cast<pybind11::list>(state["powers"]);
              for (auto handle : playerPowers) {
                  auto power = pybind11::cast<pybind11::dict>(handle);
                  const auto name = normalized(get_s(power, "name"));
                  const int amount = get_i(power, "amount");
                  if (name == "STRENGTH") bc.player.strength = amount;
                  else if (name == "DEXTERITY") bc.player.dexterity = amount;
                  else if (name == "FOCUS") bc.player.focus = amount;
                  else if (name == "ARTIFACT") bc.player.artifact = amount;
                  else throw std::runtime_error("unsupported_player_power_v1:" + name);
              }

              auto relics = pybind11::cast<pybind11::list>(state["relics"]);
              for (auto handle : relics) {
                  auto relic = pybind11::cast<pybind11::dict>(handle);
                  const auto id = normalized(get_s(relic, "id"));
                  if (id == "BURNING_BLOOD") bc.player.setHasRelic<RelicId::BURNING_BLOOD>(true);
                  else throw std::runtime_error("unsupported_relic_v1:" + id);
              }

              auto potions = pybind11::cast<pybind11::list>(state["potions"]);
              if (pybind11::len(potions) > 5) throw std::runtime_error("potion_capacity_too_large_v1");
              bc.potionCapacity = static_cast<int>(pybind11::len(potions));
              bc.potionCount = 0;
              bc.potions.fill(Potion::EMPTY_POTION_SLOT);
              for (auto handle : potions) {
                  auto potion = pybind11::cast<pybind11::dict>(handle);
                  const auto id = normalized(get_s(potion, "id"));
                  if (id != "EMPTY_POTION_SLOT") throw std::runtime_error("nonempty_potion_unsupported_v1:" + id);
              }

              bc.cards = CardManager{};
              auto add_cards = [&](const char *pileName) {
                  auto pile = pybind11::cast<pybind11::list>(state[pileName]);
                  for (auto handle : pile) {
                      auto cardDict = pybind11::cast<pybind11::dict>(handle);
                      auto card = card_from_public(cardDict);
                      if (std::string(pileName) == "hand") {
                          if (bc.cards.cardsInHand >= CardManager::MAX_HAND_SIZE) throw std::runtime_error("hand_too_large_v1");
                          bc.cards.createTempCardInHand(card);
                      } else if (std::string(pileName) == "draw_pile") {
                          bc.cards.createTempCardInDrawPile(static_cast<int>(bc.cards.drawPile.size()), card);
                      } else if (std::string(pileName) == "discard_pile") {
                          bc.cards.createTempCardInDiscard(card);
                      } else if (std::string(pileName) == "exhaust_pile") {
                          card.setUniqueId(bc.cards.nextUniqueCardId++);
                          bc.cards.exhaustPile.push_back(card);
                      }
                  }
              };
              add_cards("hand");
              add_cards("draw_pile");
              add_cards("discard_pile");
              add_cards("exhaust_pile");

              auto enemies = pybind11::cast<pybind11::list>(state["enemies"]);
              if (pybind11::len(enemies) != 1) throw std::runtime_error("enemy_count_not_one_v1");
              auto enemy = pybind11::cast<pybind11::dict>(enemies[0]);
              if (normalized(get_s(enemy, "name")) != "JAW_WORM") throw std::runtime_error("enemy_not_jaw_worm_v1");

              bc.monsters = MonsterGroup{};
              bc.monsters.monsterCount = 1;
              bc.monsters.monstersAlive = 1;
              auto &mo = bc.monsters.arr[0];
              mo = Monster{};
              mo.idx = 0;
              mo.id = MonsterId::JAW_WORM;
              mo.curHp = get_i(enemy, "hp");
              mo.maxHp = get_i(enemy, "max_hp");
              mo.block = get_i(enemy, "block");
              mo.moveHistory[0] = jaw_worm_move(get_s(enemy, "intent"));
              const auto historyChoice = get_u64(seeds, "previous_history") % 3ULL;
              mo.moveHistory[1] = historyChoice == 0 ? MMID::JAW_WORM_CHOMP
                                : historyChoice == 1 ? MMID::JAW_WORM_THRASH
                                                     : MMID::JAW_WORM_BELLOW;

              auto enemyPowers = pybind11::cast<pybind11::list>(enemy["powers"]);
              for (auto handle : enemyPowers) {
                  auto power = pybind11::cast<pybind11::dict>(handle);
                  const auto name = normalized(get_s(power, "name"));
                  const int amount = get_i(power, "amount");
                  if (name == "STRENGTH") mo.setStatus<MonsterStatus::STRENGTH>(amount);
                  else if (name == "VULNERABLE") mo.setStatus<MonsterStatus::VULNERABLE>(amount);
                  else if (name == "WEAK") mo.setStatus<MonsterStatus::WEAK>(amount);
                  else if (name == "POISON") mo.setStatus<MonsterStatus::POISON>(amount);
                  else throw std::runtime_error("unsupported_enemy_power_v1:" + name);
              }

              return bc;
          },
          pybind11::arg("public_state"), pybind11::arg("sample_seeds"),
          "construct a new Jaw Worm BattleContext from public state and fresh sample seeds only");

'''


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"missing hydrated binding: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        raise SystemExit("public reconstruction overlay already applied")
    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(f"unexpected public reconstruction anchor count: {count}")
    TARGET.write_text(text.replace(ANCHOR, INSERT + ANCHOR, 1), encoding="utf-8")
    print(f"patched={TARGET}")
    print("constructor=build_public_jaw_worm_context_v1")
    print("accepts_existing_battle_context=0")
    print("source_hidden_rng_access=0")


if __name__ == "__main__":
    main()
