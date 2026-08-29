#!/usr/bin/env python3
"""Add the exact Anger + Finesse same-turn native reconstruction slice.

This overlay runs after the existing Finesse overlay and widens only the
already-audited rich-midturn auxiliary gate from the two-attack Anger+Pommel
composition to one additional exact counter mix: two cards, one attack, one
skill, with public piles proving two Angers and one normal Finesse.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
BASE_MARKER = "phase1_public_finesse_midturn_aux_v2"
MARKER = "phase1_public_anger_finesse_composition_v1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"unexpected {label} anchor count: {count}")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"missing hydrated binding: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if BASE_MARKER not in text:
        raise SystemExit("Finesse overlay must be applied before Anger+Finesse composition overlay")
    if MARKER in text:
        raise SystemExit("Anger+Finesse composition overlay already applied")

    text = replace_once(
        text,
        '''                  const bool attackCounterSlice = played == 1 && attacks == 1 && skills == 0 && discarded == 0;
                  const bool skillCounterSlice = played == 1 && attacks == 0 && skills == 1 && discarded == 0;
                  const bool attackCompositionSlice = played == 2 && attacks == 2 && skills == 0 && discarded == 0;
                  if (!attackCounterSlice && !skillCounterSlice && !attackCompositionSlice) {
                      throw std::runtime_error("draw_aux_counter_slice_unsupported");
                  }''',
        '''                  const bool attackCounterSlice = played == 1 && attacks == 1 && skills == 0 && discarded == 0;
                  const bool skillCounterSlice = played == 1 && attacks == 0 && skills == 1 && discarded == 0;
                  const bool attackCompositionSlice = played == 2 && attacks == 2 && skills == 0 && discarded == 0;
                  // phase1_public_anger_finesse_composition_v1: one exact mixed composition only.
                  const bool mixedCompositionSlice = played == 2 && attacks == 1 && skills == 1 && discarded == 0;
                  if (!attackCounterSlice && !skillCounterSlice && !attackCompositionSlice && !mixedCompositionSlice) {
                      throw std::runtime_error("draw_aux_counter_slice_unsupported");
                  }''',
        "mixed counter slice",
    )

    text = replace_once(
        text,
        '''                  const bool pommelSlice = attackCounterSlice && pommels == 1 && shrugs == 0 && angers == 0 && finesses == 0;
                  const bool angerSlice = attackCounterSlice && pommels == 0 && shrugs == 0 && angers == 2 && finesses == 0;
                  const bool shrugSlice = skillCounterSlice && pommels == 0 && shrugs == 1 && angers == 0 && finesses == 0;
                  const bool angerPommelSlice = attackCompositionSlice && pommels == 1 && shrugs == 0 && angers == 2 && finesses == 0;
                  const bool finesseSlice = skillCounterSlice && pommels == 0 && shrugs == 0 && angers == 0 && finesses == 1;
                  if (!pommelSlice && !shrugSlice && !angerSlice && !angerPommelSlice && !finesseSlice) {
                      throw std::runtime_error("draw_aux_public_card_identity_unsupported");
                  }
                  const char *prefix = pommelSlice ? "pommel"
                      : (shrugSlice ? "shrug" : (angerSlice ? "anger" : (angerPommelSlice ? "anger_pommel" : "finesse")));''',
        '''                  const bool pommelSlice = attackCounterSlice && pommels == 1 && shrugs == 0 && angers == 0 && finesses == 0;
                  const bool angerSlice = attackCounterSlice && pommels == 0 && shrugs == 0 && angers == 2 && finesses == 0;
                  const bool shrugSlice = skillCounterSlice && pommels == 0 && shrugs == 1 && angers == 0 && finesses == 0;
                  const bool angerPommelSlice = attackCompositionSlice && pommels == 1 && shrugs == 0 && angers == 2 && finesses == 0;
                  const bool finesseSlice = skillCounterSlice && pommels == 0 && shrugs == 0 && angers == 0 && finesses == 1;
                  const bool angerFinesseSlice = mixedCompositionSlice && pommels == 0 && shrugs == 0 && angers == 2 && finesses == 1;
                  if (!pommelSlice && !shrugSlice && !angerSlice && !angerPommelSlice && !finesseSlice && !angerFinesseSlice) {
                      throw std::runtime_error("draw_aux_public_card_identity_unsupported");
                  }
                  const char *prefix = pommelSlice ? "pommel"
                      : (shrugSlice ? "shrug" : (angerSlice ? "anger" : (angerPommelSlice ? "anger_pommel" : (finesseSlice ? "finesse" : "anger_finesse"))));''',
        "Anger Finesse identity slice",
    )

    text = replace_once(
        text,
        '''                  const int expectedHand = (angerSlice || angerPommelSlice) ? 4 : 5;
                  const int expectedDraw = (angerSlice || angerPommelSlice) ? 1 : 0;
                  const int expectedDiscard = angerPommelSlice ? 8 : (angerSlice ? 7 : 6);''',
        '''                  const int expectedHand = (angerSlice || angerPommelSlice || angerFinesseSlice) ? 4 : 5;
                  const int expectedDraw = (angerSlice || angerPommelSlice || angerFinesseSlice) ? 1 : 0;
                  const int expectedDiscard = (angerPommelSlice || angerFinesseSlice) ? 8 : (angerSlice ? 7 : 6);''',
        "pile shape",
    )

    text = replace_once(
        text,
        '''                  const int expectedEnergy = (angerSlice || finesseSlice) ? 3 : 2;''',
        '''                  const int expectedEnergy = (angerSlice || finesseSlice || angerFinesseSlice) ? 3 : 2;''',
        "energy shape",
    )

    text = replace_once(
        text,
        '''                      || pommels != ((pommelSlice || angerPommelSlice) ? 1 : 0)
                      || shrugs != (shrugSlice ? 1 : 0)
                      || angers != ((angerSlice || angerPommelSlice) ? 2 : 0)
                      || finesses != (finesseSlice ? 1 : 0)''',
        '''                      || pommels != ((pommelSlice || angerPommelSlice) ? 1 : 0)
                      || shrugs != (shrugSlice ? 1 : 0)
                      || angers != ((angerSlice || angerPommelSlice || angerFinesseSlice) ? 2 : 0)
                      || finesses != ((finesseSlice || angerFinesseSlice) ? 1 : 0)''',
        "deck composition",
    )

    text = replace_once(
        text,
        '''                  const int expectedBlock = (pommelSlice || angerSlice || angerPommelSlice)
                      ? 0
                      : (finesseSlice ? 2 : (upgradedShrugs == 1 ? 11 : 8));''',
        '''                  const int expectedBlock = (pommelSlice || angerSlice || angerPommelSlice)
                      ? 0
                      : ((finesseSlice || angerFinesseSlice) ? 2 : (upgradedShrugs == 1 ? 11 : 8));''',
        "block shape",
    )

    text = replace_once(
        text,
        '''                          || (shrugSlice && c.id == CardId::SHRUG_IT_OFF)
                          || (finesseSlice && c.id == CardId::FINESSE)) {''',
        '''                          || (shrugSlice && c.id == CardId::SHRUG_IT_OFF)
                          || ((finesseSlice || angerFinesseSlice) && c.id == CardId::FINESSE)) {''',
        "discard Finesse proof",
    )

    text = replace_once(
        text,
        '''                      || ((angerSlice || angerPommelSlice) && discardAngers != 2)) {''',
        '''                      || ((angerSlice || angerPommelSlice || angerFinesseSlice) && discardAngers != 2)) {''',
        "discard generated Anger proof",
    )

    text += f"\n// {MARKER}\n"
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched={TARGET}")
    print("anger_finesse_composition=NORMAL_ANGER_PLUS_NORMAL_FINESSE_ONLY")
    print("expected_cards_played=2")
    print("expected_attacks=1")
    print("expected_skills=1")
    print("expected_energy=3")
    print("expected_block=2")
    print("public_contract_fields_added=0")
    print("hidden_rng_access_added=0")


if __name__ == "__main__":
    main()
