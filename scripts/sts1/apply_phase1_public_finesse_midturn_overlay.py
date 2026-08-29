#!/usr/bin/env python3
"""Add one audited normal-Finesse midturn slice after draw-midturn aux v2."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
BASE_MARKER = "phase1_public_draw_midturn_aux_v2"
MARKER = "phase1_public_finesse_midturn_aux_v1"


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
        raise SystemExit("draw-midturn aux v2 must be applied before Finesse overlay")
    if MARKER in text:
        raise SystemExit("Finesse overlay already applied")

    text = replace_once(
        text,
        '''                  else if (idText == "SHRUG_IT_OFF") { id = CardId::SHRUG_IT_OFF; expectedCost = 1; }
                  else throw std::runtime_error("unsupported_card_v1:" + idText);''',
        '''                  else if (idText == "SHRUG_IT_OFF") { id = CardId::SHRUG_IT_OFF; expectedCost = 1; }
                  else if (idText == "FINESSE") { id = CardId::FINESSE; expectedCost = 0; }
                  else throw std::runtime_error("unsupported_card_v1:" + idText);''',
        "card parser",
    )

    text = replace_once(
        text,
        '''                  const bool pommelSlice = played == 1 && attacks == 1 && skills == 0 && discarded == 0;
                  const bool shrugSlice = played == 1 && attacks == 0 && skills == 1 && discarded == 0;
                  if (!pommelSlice && !shrugSlice) {
                      throw std::runtime_error("draw_aux_counter_slice_unsupported");
                  }
                  const char *prefix = pommelSlice ? "pommel" : "shrug";''',
        '''                  // phase1_public_finesse_midturn_aux_v1:
                  // counters prove the card type only; exact identity is resolved
                  // from the current public deck composition below. Keep the
                  // established Shrug error prefix for the shared skill trace so
                  // existing native negative proofs remain byte-for-byte stable.
                  const bool pommelTrace = played == 1 && attacks == 1 && skills == 0 && discarded == 0;
                  const bool skillTrace = played == 1 && attacks == 0 && skills == 1 && discarded == 0;
                  if (!pommelTrace && !skillTrace) {
                      throw std::runtime_error("draw_aux_counter_slice_unsupported");
                  }
                  const char *prefix = pommelTrace ? "pommel" : "shrug";''',
        "counter classifier",
    )

    text = replace_once(
        text,
        '''                  if (bc.player.energy != 2) {
                      throw slice_error("_midturn_player_shape_mismatch");
                  }''',
        '''                  if (bc.player.energy < 0 || bc.player.energy > 3) {
                      throw slice_error("_midturn_player_shape_mismatch");
                  }''',
        "early energy guard",
    )

    text = replace_once(
        text,
        '''                  int pommels = 0;
                  int shrugs = 0;
                  int unsupported = 0;''',
        '''                  int pommels = 0;
                  int shrugs = 0;
                  int finesses = 0;
                  int unsupported = 0;''',
        "finesse counter declaration",
    )

    text = replace_once(
        text,
        '''                      else if (c.id == CardId::POMMEL_STRIKE) ++pommels;
                      else if (c.id == CardId::SHRUG_IT_OFF) ++shrugs;
                      else ++unsupported;''',
        '''                      else if (c.id == CardId::POMMEL_STRIKE) ++pommels;
                      else if (c.id == CardId::SHRUG_IT_OFF) ++shrugs;
                      else if (c.id == CardId::FINESSE) ++finesses;
                      else ++unsupported;''',
        "finesse card counter",
    )

    text = replace_once(
        text,
        '''                  if (strikes != 5 || defends != 4 || bashes != 1
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
                  }''',
        '''                  const bool pommelSlice = pommelTrace && pommels == 1 && shrugs == 0 && finesses == 0;
                  const bool shrugSlice = skillTrace && pommels == 0 && shrugs == 1 && finesses == 0;
                  const bool finesseSlice = skillTrace && pommels == 0 && shrugs == 0 && finesses == 1;
                  if (!pommelSlice && !shrugSlice && !finesseSlice) {
                      throw slice_error("_identity_slice_unsupported");
                  }
                  if (strikes != 5 || defends != 4 || bashes != 1
                      || pommels != (pommelSlice ? 1 : 0)
                      || shrugs != (shrugSlice ? 1 : 0)
                      || finesses != (finesseSlice ? 1 : 0)
                      || unsupported != 0 || upgradedOther != 0
                      || (pommelSlice && upgradedShrugs != 0)
                      || (shrugSlice && (upgradedShrugs < 0 || upgradedShrugs > 1))
                      || (finesseSlice && upgradedShrugs != 0)) {
                      throw slice_error("_midturn_deck_composition_mismatch");
                  }

                  const int expectedEnergy = finesseSlice ? 3 : 2;
                  const int expectedBlock = pommelSlice ? 0 : (finesseSlice ? 2 : (upgradedShrugs == 1 ? 11 : 8));
                  if (bc.player.energy != expectedEnergy || bc.player.block != expectedBlock) {
                      throw slice_error("_midturn_player_shape_mismatch");
                  }''',
        "exact identity and player shape",
    )

    text = replace_once(
        text,
        '''                      if ((pommelSlice && c.id == CardId::POMMEL_STRIKE)
                          || (shrugSlice && c.id == CardId::SHRUG_IT_OFF)) {
                          ++discardPlayedCard;
                      }''',
        '''                      if ((pommelSlice && c.id == CardId::POMMEL_STRIKE)
                          || (shrugSlice && c.id == CardId::SHRUG_IT_OFF)
                          || (finesseSlice && c.id == CardId::FINESSE)) {
                          ++discardPlayedCard;
                      }''',
        "played card discard identity",
    )

    text = replace_once(
        text,
        '''                  bc.player.cardsPlayedThisTurn = 1;
                  bc.player.attacksPlayedThisTurn = pommelSlice ? 1 : 0;
                  bc.player.skillsPlayedThisTurn = shrugSlice ? 1 : 0;
                  bc.player.cardsDiscardedThisTurn = 0;''',
        '''                  bc.player.cardsPlayedThisTurn = 1;
                  bc.player.attacksPlayedThisTurn = pommelSlice ? 1 : 0;
                  bc.player.skillsPlayedThisTurn = (shrugSlice || finesseSlice) ? 1 : 0;
                  bc.player.cardsDiscardedThisTurn = 0;''',
        "turn counters",
    )

    text += f"\n// {MARKER}\n"
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched={TARGET}")
    print("finesse_midturn=NORMAL_ONLY")
    print("expected_energy=3")
    print("expected_block=2")
    print("expected_draw=1")
    print("public_contract_fields_added=0")
    print("hidden_rng_access_added=0")


if __name__ == "__main__":
    main()
