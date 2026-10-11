# Round018 H23: exact Clash reward Skip in a mixed deck

Issue #19; prerequisite Draft PR #37 at `c44f23329aa8200b7cc8707a104959d9295fda01`.
This registration precedes every H23 episode. G7 remains unchanged and retained.

## Fixed mechanism

On a card reward only, when G7 recommends Clash (base or upgraded), require a complete choice list, a fully identified native-type deck containing at least one nonattack, and exactly one legal card Skip. Select that existing Skip index. Unknown identity/type/deck, attack-only deck or missing/ambiguous Skip preserves G7. No deck-size or duplicate threshold, reward reranking, shop/campfire/map change, combat intervention, training or paid model/API is introduced. Clash's native restriction is on the hand; a mixed deck is a hypothesis cue, not proof that Clash is unplayable or skipping helps.

Shadow evidence from the completed Round017 Train10: ten opportunities across seven baseline runs. This is exposure, not an executed counterfactual. H12/H13/H14 generic selectivity results remain unchanged. Do not tune H23 on Probe/Dev/Formal.

## Prospective allocation and gates

Extend the complete Round017 exclusion inventory with all five Round017 USED/RESERVED pools, including every untouched pool. Generate new mutually disjoint Round018 pools; never execute or reopen old reserved seeds. Freeze code commit, selector, card-fact/type parser, reviewed native sources, corrected native binding, G7/ArmG source/vocabulary/checkpoint, evaluator/trace/ledger helpers and private pool identities before outcomes.

Both arms use the same corrected pinned simulator gameplay `7476a81954020087da31d41d16fddf475746ec2d`, Ironclad A0, G7 and MCTS2000. Train10: 20 paired episodes plus one first-candidate trace-off replay, exactly 21. Conditional Probe10: 20 episodes; conditional Dev30: 60. This single-hypothesis round has a fixed maximum of 101 episodes inside the ordinary 140 cap; no retries, extra samples or parameter sweep.

Every override is recomputed from the frozen selector and exact legal Skip. Effective coverage requires the first changed reward to have equal recorded pre-action state, legal choices and G7 recommendation, identical preceding encounter/combat/noncombat recorded state/action prefix, and different applied state/deck count (parent accepts one card, candidate skips). Recorded-state/prefix matching is not a full serialization of hidden native RNG state; same initial seed/runtime and deterministic matched prefix are prerequisites, not permission to infer unsupported causes.

Preserve full raw traces/actions/terminal/provenance and failed starts privately. Existing legality, completeness, safety, artifact limits and trace-on/off invariance remain blocking. Train and Probe require at least three seeds with effective overrides and nonnegative paired net; Dev requires positive net and complete guards. Exact one-sided sign p is reported. Paired terminal-floor/final-HP decreases remain diagnostics per Issue #19, not independent vetoes. Confirmation and actual-game acceptance require separate existing gates and are not executed here.

Report all wins, C-only/P-only/net/discordant/p, overrides/effective coverage, safety, complete episodes and paired terminal deltas, including nulls and regressions. Any integrity failure stops this round as NOT_VERIFIED, retaining partial evidence. No promotion or Champion replacement.

At registration: runtime episodes, paired win gain, confirmation and real-game acceptance NOT_RUN/NOT_VERIFIED. Provider token usage NOT_MEASURED.

Pre-episode verification: 110 tests and 2 subtests passed; corrected native combat-resource smoke passed with zero episodes started. Independent source review found no proven blocking bug and verified all 364 native type entries against the selector whitelist. Runtime paired outcomes and trace-on/off invariance remain pending.
