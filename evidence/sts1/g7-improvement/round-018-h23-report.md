# Round018 H23 — mixed-deck Clash reward Skip

H23 stopped after Probe because effective override coverage was 2/10, below the preregistered minimum of three seeds. Probe produced one candidate-only victory, but this does not establish a win-rate improvement. G7 remains the champion. No Dev, confirmation or real-game episodes were started.

## Frozen scope and execution

Issue #19; Draft PR #38, based on unmerged PR #37. Executed code commit: `f0db784896964b620115473b60cc5893c080c4b3`. Both arms used the same corrected native simulator gameplay `7476a81954020087da31d41d16fddf475746ec2d`, G7/ArmG, Ironclad A0 and MCTS2000. The selector, card parser, five reviewed native source files, binding, checkpoint, evaluators and complete pool inventory were frozen before outcomes. No model/API call, training or checkpoint replacement occurred.

The sole intervention selects the exact unique existing legal card Skip when G7 recommends Clash and the fully identified native-type deck includes a nonattack. Unknown or ambiguous inputs preserve G7. Clash requires the current hand to contain only attacks; a mixed deck is merely the registered cue. Combat, routing, shops and campfires were unchanged.

Fresh Round018 pools excluded 23,106 prior USED/RESERVED/unknown-purpose seeds across 86 source manifests, including all five Round017 pools. Seventy new mutually disjoint seeds were reserved. Only Train10 and Probe10 were used. Untouched Dev/Formal pools remain reserved and excluded from future allocations; they must not be reopened to tune this hypothesis.

## Complete paired results

| Stage | Pairs | Episodes | G7 wins | H23 wins | C-only | P-only | Net | Discordant | Exact one-sided sign p | Overrides | Effective seeds | Advancement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Train | 10 | 21 | 2 | 2 | 0 | 0 | 0 | 0 | 1.0 | 7 | 6 | Eligible |
| Probe | 10 | 20 | 0 | 1 | 1 | 0 | +1 | 1 | 0.5 | 3 | 2 | Stop: coverage below 3 |

Total: 20 pairs, 41 episodes including one trace-off replay; G7 2/20 wins and H23 3/20. This descriptive aggregate combines different stage pools and is not a confirmation result. No extra episodes, retries, parameter adjustment or alternate hypothesis were executed. Both stage summaries are COMPLETE with hard guards passed. Train met the existing coverage and nonnegative-net gates; Probe failed coverage despite positive net. The positive Probe signal does not waive that gate.

An effective override requires an identical recorded encounter/combat/noncombat state/action prefix through the first changed reward, matching pre-action deck, canonical choices and G7 recommendation, then parent accepting one card while candidate skips. Every override was independently recomputed by the frozen selector in the runner. This verifies the recorded prefix, not serialization of hidden native RNG state.

## Integrity and secondary diagnostics

All 41 terminal records completed; all 40 paired episodes passed legal-action trace validation. Illegal actions, crashes, timeouts and communication errors were each zero. The first candidate trace-off replay matched outcome, override count and all 466 action signatures; it is excluded from the paired denominator. Trace evidence includes exact legal actions, raw states, terminal records and provenance and remains private. Artifact payloads recorded by the evaluator were 118,379,306 bytes for Train and 109,155,445 for Probe, within the existing limits.

| Stage | Mean candidate-minus-G7 terminal floor | Floor higher/equal/lower | Mean final HP delta | HP higher/equal/lower |
| --- | ---: | --- | ---: | --- |
| Train | +1.7 | 1 / 9 / 0 | -0.6 | 0 / 9 / 1 |
| Probe | +1.8 | 2 / 8 / 0 | +0.1 | 1 / 9 / 0 |

These are secondary diagnostics under Issue #19, not independent win-rate evidence or gate overrides. Most paired outcomes remained unchanged.

## Verification and next work

Pre-episode validation: 110 tests plus 2 subtests passed, covering exact Skip identity, unknown/ambiguous abstention, prefix integrity, flag guards, fixed gates, fresh exclusion and source-freeze mutation rejection, plus relevant existing H22/collector/trace/seed-ledger regressions. Corrected native resource smoke passed with zero episodes. Independent source review found no proven blocking bug and verified all 364 native card types against the selector whitelist. Independent completion review re-aggregated both stage summaries from their paired records and confirmed wins, net, p, overrides, coverage and terminal diagnostics. It did not re-read full raw traces or independently replay native actions.

Do not extend H23 after this coverage failure or promote it from a single discordant win. The next separate research task should use already collected Train preparation evidence to characterize draw/scaling consistency and exact reward/removal/upgrade semantics, then prospectively register one fresh intervention. Probe outcomes must not be used to tune H23. Existing research sources and preparation analysis remain in PRs #35–#37.

Formal confirmation, actual STS1 game smoke/pilot/acceptance/Victory and win-rate improvement remain NOT_RUN/NOT_VERIFIED. Complete hidden RNG equivalence and independent raw-trace replay are NOT_VERIFIED. Provider usage is NOT_MEASURED. Original dirty parent workspace was preserved. This report commit follows all runtime stages and was not used as the frozen gameplay code identity.
