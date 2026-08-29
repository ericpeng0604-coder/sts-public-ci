# STS1 Phase 1 — Anger + Pommel Composition V1

Status: implementation candidate; requires native CI before admission is trusted.

Base formal reconstruction HEAD: `5a4dded2fb8ee7a07a737dab81a95d9bdcbd6d94`.

## Exact admitted slice

- Ironclad vs Jaw Worm.
- Zero-based `turn == 1` only.
- Pre-generation combat composition: starter 10 cards + one normal Anger + one normal Pommel Strike.
- Exactly two cards played this turn.
- Exactly two attacks, zero skills, zero explicit discards.
- Current public piles contain exactly two normal Angers, both in discard, proving the generated copy inside this frozen deck slice.
- Current public piles contain the normal Pommel Strike in discard.
- Current pile shape is hand 4 / draw 1 / discard 8 / exhaust 0, proving the one-card draw effect inside this frozen deck slice.
- Energy is 2, block is 0, player powers are empty.

## Reconstruction auxiliary contract

The auxiliary object remains `sts1-public-reconstruction-aux-v1` from `communicationmod_command_trace_v1` and contains only bounded turn-local aggregate counters. It is reconstruction metadata, not policy observation state.

Expected counters:

- `turn = 1`
- `cards_played_this_turn = 2`
- `attacks_played_this_turn = 2`
- `skills_played_this_turn = 0`
- `cards_discarded_this_turn = 0`

## Safety invariants

- `sts1-public-state-v1` is unchanged.
- No formal public field is added.
- No source `BattleContext` is accepted.
- No hidden RNG or draw order is read from the source battle.
- No source move history or source simulator counters are read.
- No general hand-size shortcut is introduced.
- Existing Pommel, Shrug, Shrug+, Anger, starter-only, opening, and enemy quality slices must continue to pass unchanged.
- Any third play, extra rich card, upgrade, power, discard effect, unexpected pile shape, or incomplete auxiliary trace fails closed.

## Gate

This slice is not trusted until focused Python tests, native build/proof, executable legal action, and 5-state x2 deterministic Search smoke all pass on the candidate HEAD. It does not claim Phase 1 completion and does not unlock Phase 2.
