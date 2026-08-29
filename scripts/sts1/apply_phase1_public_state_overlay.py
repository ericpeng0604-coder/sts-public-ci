#!/usr/bin/env python3
"""Expose missing *public* STS1 runtime state through the hydrated binding.

This overlay is intentionally read-only.  It exposes the player's visible
potion belt (including empty slots) and does not expose seed/RNG/future state.
The script is idempotent only in the fail-closed sense: an already-patched or
unexpected upstream file is rejected instead of guessed.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
MARKER = "phase1_public_potion_belt_v1"

ANCHOR = '''        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        .def_property_readonly("cur_event",'''

REPLACEMENT = '''        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        // phase1_public_potion_belt_v1: player-visible, read-only state only.
        .def_property_readonly("potions",
               [](const GameContext &gc) {
                   pybind11::list out;
                   for (int idx = 0; idx < gc.potionCapacity; ++idx) {
                       const auto potion = gc.potions[idx];
                       pybind11::dict item;
                       item["index"] = idx;
                       item["id"] = std::string(potionEnumNames[static_cast<int>(potion)]);
                       item["name"] = std::string(getPotionName(potion));
                       item["empty"] = potion == Potion::EMPTY_POTION_SLOT;
                       out.append(item);
                   }
                   return out;
               },
               "player-visible potion belt including empty slots; no RNG/seed state"
        )
        .def_property_readonly("cur_event",'''


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"missing hydrated binding: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        raise SystemExit("public-state overlay already applied")
    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(f"unexpected public-state overlay anchor count: {count}")
    TARGET.write_text(text.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    print(f"patched={TARGET}")
    print("exposed=GameContext.potions")
    print("hidden_rng_exposed=0")


if __name__ == "__main__":
    main()
