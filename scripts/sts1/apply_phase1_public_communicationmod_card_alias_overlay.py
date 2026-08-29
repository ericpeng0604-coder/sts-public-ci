#!/usr/bin/env python3
"""Accept CommunicationMod's public starter-card id spellings at native boundary.

CommunicationMod uses Strike_R / Defend_R while sts_lightspeed uses
STRIKE_RED / DEFEND_RED. They are the same visible cards. This overlay only
adds those two explicit aliases; it does not rewrite state, expose UUIDs, or
broaden to unknown card ids.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "external" / "sts_lightspeed" / "source" / "bindings" / "slaythespire.cpp"
MARKER = "phase1_public_communicationmod_card_alias_v1"

STRIKE_OLD = '''                  if (idText == "STRIKE_RED") { id = CardId::STRIKE_RED; expectedCost = 1; }'''
STRIKE_NEW = '''                  // phase1_public_communicationmod_card_alias_v1: explicit public-name alias only.
                  if (idText == "STRIKE_RED" || idText == "STRIKE_R") { id = CardId::STRIKE_RED; expectedCost = 1; }'''
DEFEND_OLD = '''                  else if (idText == "DEFEND_RED") { id = CardId::DEFEND_RED; expectedCost = 1; }'''
DEFEND_NEW = '''                  else if (idText == "DEFEND_RED" || idText == "DEFEND_R") { id = CardId::DEFEND_RED; expectedCost = 1; }'''


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
        raise SystemExit("CommunicationMod card alias overlay already applied")
    text = replace_once(text, STRIKE_OLD, STRIKE_NEW, "Strike alias")
    text = replace_once(text, DEFEND_OLD, DEFEND_NEW, "Defend alias")
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched={TARGET}")
    print("aliases=STRIKE_R->STRIKE_RED,DEFEND_R->DEFEND_RED")
    print("public_contract_fields_added=0")
    print("hidden_state_access_added=0")


if __name__ == "__main__":
    main()
