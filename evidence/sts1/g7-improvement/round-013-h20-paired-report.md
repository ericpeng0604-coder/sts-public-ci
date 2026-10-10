# STS1 G7 H20: no measured win-rate gain

When MCTS recommends End Turn with an eligible legal Inflame in hand, the frozen
H20 candidate plays the lowest-position Inflame using its exact native action
ordinal. Unknown powers, card-play penalty enemies, relevant curses and
incomplete inputs cause abstention. G7's weights and baseline policy remain
unchanged. H20 uses independent trace fields; H19 retains its registered behavior.

Issue #19; PR #30. Frozen runtime/evaluator commit:
`bae6a0be59c5371ea9e7052a1770087dd03dff28`.
Preregistration: https://github.com/ericpeng0604-coder/sts-public-ci/issues/19#issuecomment-6097026203

| Stage | Pairs | G7 wins | H20 wins | H20-only / G7-only | Net | One-sided sign p | Effective seed coverage |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Train | 10 | 2 | 2 | 0 / 0 | 0 | 1 | 5 / required 3 |
| Probe | 10 | 1 | 1 | 0 / 0 | 0 | 1 | 2 / required 3 |

Both stages completed, with 41 episodes total: 40 paired episodes plus one
candidate trace-off replay. Train replay outcome and selected-action signature
matched trace-on. Illegal, crash, timeout and communication counters were all
zero. Every paired episode passed the complete canonical legal-action trace and
retention checks. Train recorded 6 H20 overrides; Probe recorded 3. Coverage
counts require matching parent prefixes and changed afterstates, not just actions.

Train passed its gate. Probe failed its prospective effective-coverage gate;
H20 is rejected for advancement. Dev and confirmation are NOT_RUN. G7 remains
Champion. Descriptive combined wins are 3/20 for both arms, net 0; combining
exploration stages does not provide formal confirmation evidence.

Fresh pools excluded 22,756 existing/reserved seed IDs from 61 source manifests,
including all five Round011 pools. Private IDs, provenance fingerprints, raw
traces, pool manifests, identity lock and usage records remain outside Git. No
formal outcomes were read, and no consumed pool was reused.
All 81 raw NDJSON artifacts were retained (207,359,398 bytes).

Validation: 72 H20/H19 selector and evaluator tests passed; 25 simulator, trace
and seed-ledger tests passed with 2 subtests. Scope and whitespace checks passed.
The existing two independent 100-pair confirmation batches remain the larger
sample plan for a candidate that passes selection. This candidate is ineligible;
the gate was not relaxed after observing results.

Win-rate improvement: NOT_VERIFIED. Real-game acceptance: NOT_RUN. Provider
token usage: NOT_MEASURED. No promotion, deployment, main commit or merge.

Branch: `codex/sts1-g7-h20-inflame-20261010`. This runtime/evaluation task changed:

- `src/roguelike_ai/sts1_phase3/simulator.py`
- `scripts/sts1/sts1_g7_h16_lethal_defend_eval.py`
- `tests/test_sts1_h16_lethal_defend.py`
- `tests/test_sts1_h20_inflame.py`
- `evidence/sts1/g7-improvement/round-013-h20-runtime-preregistration.md`
- `evidence/sts1/g7-improvement/round-013-h20-paired-report.md`
