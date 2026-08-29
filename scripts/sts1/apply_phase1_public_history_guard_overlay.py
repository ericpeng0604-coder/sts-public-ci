#!/usr/bin/env python3
"""Make V1 monster prior-history reconstruction turn-aware and fail closed."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
MARKER = "phase1_public_history_turn_guard_v1"

ANCHOR = '''              const auto historyChoice = get_u64(seeds, "previous_history") % 3ULL;
              mo.moveHistory[1] = historyChoice == 0 ? MMID::JAW_WORM_CHOMP
                                : historyChoice == 1 ? MMID::JAW_WORM_THRASH
                                                     : MMID::JAW_WORM_BELLOW;'''

REPLACEMENT = '''              // phase1_public_history_turn_guard_v1:
              // The pinned simulator is zero-based: turn 0 is the first player turn,
              // so no older monster move exists there. Later turns sample it.
              if (bc.turn <= 0) {
                  mo.moveHistory[1] = MMID::INVALID;
              } else {
                  const auto historyChoice = get_u64(seeds, "previous_history") % 3ULL;
                  mo.moveHistory[1] = historyChoice == 0 ? MMID::JAW_WORM_CHOMP
                                    : historyChoice == 1 ? MMID::JAW_WORM_THRASH
                                                         : MMID::JAW_WORM_BELLOW;
              }'''


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"missing hydrated binding: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        raise SystemExit("public history guard already applied")
    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(f"unexpected history guard anchor count: {count}")
    TARGET.write_text(text.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    print(f"patched={TARGET}")
    print("opening_turn_zero_previous_history=INVALID")
    print("later_turn_previous_history=PUBLIC_SAMPLE")
    print("source_move_history_access=0")


if __name__ == "__main__":
    main()
