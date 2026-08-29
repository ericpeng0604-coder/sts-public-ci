#!/usr/bin/env python3
"""Add one audited normal-Finesse midturn slice after the Anger-rich draw overlay."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
BASE_MARKER = "phase1_public_draw_midturn_aux_v2"
ANGER_MARKER = "phase1_public_anger_midturn_aux_v1"
MARKER = "phase1_public_finesse_midturn_aux_v2"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"unexpected {label} anchor count: {count}")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"missing hydrated binding: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if BASE_MARKER not in text or ANGER_MARKER not in text:
        raise SystemExit("latest draw/Anger overlay must be applied before Finesse overlay")
    if MARKER in text:
        raise SystemExit("Finesse overlay already applied")

    text = replace_once(
        text,
        '''                  else if (idText == "ANGER") { id = CardId::ANGER; expectedCost = 0; }
                  else throw std::runtime_error("unsupported_card_v1:" + idText);''',
        '''                  else if (idText == "ANGER") { id = CardId::ANGER; expectedCost = 0; }
                  else if (idText == "FINESSE") { id = CardId::FINESSE; expectedCost = 0; }
                  else throw std::runtime_error("unsupported_card_v1:" + idText);''',
        "card parser",
    )

    text = replace_once(
        text,
        '''                  int pommels = 0;
                  int shrugs = 0;
                  int angers = 0;
                  int unsupported = 0;''',
        '''                  int pommels = 0;
                  int shrugs = 0;
                  int angers = 0;
                  int finesses = 0;
                  int unsupported = 0;''',
        "Finesse counter declaration",
    )

    text = replace_once(
        text,
        '''                      else if (c.id == CardId::SHRUG_IT_OFF) ++shrugs;
                      else if (c.id == CardId::ANGER) ++angers;
                      else ++unsupported;''',
        '''                      else if (c.id == CardId::SHRUG_IT_OFF) ++shrugs;
                      else if (c.id == CardId::ANGER) ++angers;
                      else if (c.id == CardId::FINESSE) ++finesses;
                      else ++unsupported;''',
        "Finesse card counter",
    )

    text = replace_once(
        text,
        '''                  const bool pommelSlice = attackCounterSlice && pommels == 1 && shrugs == 0 && angers == 0;
                  const bool angerSlice = attackCounterSlice && pommels == 0 && shrugs == 0 && angers == 2;
                  const bool shrugSlice = skillCounterSlice && pommels == 0 && shrugs == 1 && angers == 0;
                  const bool angerPommelSlice = attackCompositionSlice && pommels == 1 && shrugs == 0 && angers == 2;
                  if (!pommelSlice && !shrugSlice && !angerSlice && !angerPommelSlice) {
                      throw std::runtime_error("draw_aux_public_card_identity_unsupported");
                  }
                  const char *prefix = pommelSlice ? "pommel"
                      : (shrugSlice ? "shrug" : (angerSlice ? "anger" : "anger_pommel"));''',
        '''                  // phase1_public_finesse_midturn_aux_v2: identity stays
                  // public-state-derived and cannot overlap the Anger/Pommel/Shrug/Anger+Pommel slices.
                  const bool pommelSlice = attackCounterSlice && pommels == 1 && shrugs == 0 && angers == 0 && finesses == 0;
                  const bool angerSlice = attackCounterSlice && pommels == 0 && shrugs == 0 && angers == 2 && finesses == 0;
                  const bool shrugSlice = skillCounterSlice && pommels == 0 && shrugs == 1 && angers == 0 && finesses == 0;
                  const bool angerPommelSlice = attackCompositionSlice && pommels == 1 && shrugs == 0 && angers == 2 && finesses == 0;
                  const bool finesseSlice = skillCounterSlice && pommels == 0 && shrugs == 0 && angers == 0 && finesses == 1;
                  if (!pommelSlice && !shrugSlice && !angerSlice && !angerPommelSlice && !finesseSlice) {
                      throw std::runtime_error("draw_aux_public_card_identity_unsupported");
                  }
                  const char *prefix = pommelSlice ? "pommel"
                      : (shrugSlice ? "shrug" : (angerSlice ? "anger" : (angerPommelSlice ? "anger_pommel" : "finesse")));''',
        "identity slices",
    )

    text = replace_once(
        text,
        '''                  const int expectedEnergy = angerSlice ? 3 : 2;
                  if (bc.player.energy != expectedEnergy) {''',
        '''                  const int expectedEnergy = (angerSlice || finesseSlice) ? 3 : 2;
                  if (bc.player.energy != expectedEnergy) {''',
        "energy shape",
    )

    text = replace_once(
        text,
        '''                      || shrugs != (shrugSlice ? 1 : 0)
                      || angers != ((angerSlice || angerPommelSlice) ? 2 : 0)
                      || unsupported != 0 || upgradedOther != 0''',
        '''                      || shrugs != (shrugSlice ? 1 : 0)
                      || angers != ((angerSlice || angerPommelSlice) ? 2 : 0)
                      || finesses != (finesseSlice ? 1 : 0)
                      || unsupported != 0 || upgradedOther != 0''',
        "deck composition",
    )

    text = replace_once(
        text,
        '''                  const int expectedBlock = (pommelSlice || angerSlice || angerPommelSlice)
                      ? 0
                      : (upgradedShrugs == 1 ? 11 : 8);
                  if (bc.player.block != expectedBlock) {''',
        '''                  const int expectedBlock = (pommelSlice || angerSlice || angerPommelSlice)
                      ? 0
                      : (finesseSlice ? 2 : (upgradedShrugs == 1 ? 11 : 8));
                  if (bc.player.block != expectedBlock) {''',
        "block shape",
    )

    text = replace_once(
        text,
        '''                      if (((pommelSlice || angerPommelSlice) && c.id == CardId::POMMEL_STRIKE)
                          || (shrugSlice && c.id == CardId::SHRUG_IT_OFF)) {
                          ++discardPlayedCard;
                      }''',
        '''                      if (((pommelSlice || angerPommelSlice) && c.id == CardId::POMMEL_STRIKE)
                          || (shrugSlice && c.id == CardId::SHRUG_IT_OFF)
                          || (finesseSlice && c.id == CardId::FINESSE)) {
                          ++discardPlayedCard;
                      }''',
        "discard played identity",
    )

    text += f"\n// {MARKER}\n"
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched={TARGET}")
    print("finesse_midturn=NORMAL_ONLY")
    print("composes_with_anger=1")
    print("composes_with_anger_pommel=1")
    print("expected_energy=3")
    print("expected_block=2")
    print("expected_draw=1")
    print("public_contract_fields_added=0")
    print("hidden_rng_access_added=0")


if __name__ == "__main__":
    main()
