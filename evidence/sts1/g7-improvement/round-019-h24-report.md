# Round019 H24 completion — no improvement after 50 pairs

H24 completed the fixed Train10, Probe10 and Dev30 stages: 50 matched pairs and 101 episodes including one trace-off replay. Dev candidate wins were 4/30 versus G7 6/30, net -2. The frozen draw-reward rule does not qualify for confirmation. Keep G7; stop H24 without additional samples, tuning or retries.

Issue #19; Draft PR #39 based on unmerged PR #38. Executed frozen code: `0b3c4a356c61ba2e2564e29624d12b279b48d707`. Preregistration: `round-019-h24-preregistration.md`. Execution completed on 2026-10-12; the Round019 identifier was assigned during preparation and remains `round-019-20261011`.

## Scope and prospective conditions

The single intervention favors a unique existing base-name Pommel Strike, Shrug It Off or Battle Trance reward when G7 accepts another card and the fully typed deck contains fewer than two cards in that fixed target set. Unknown/ambiguous inputs, inconsistent reviewed target upgrades, G7 Skip and an already recommended target preserve G7. There is no H23 combination, combat/route/shop/campfire intervention, training, paid API or checkpoint replacement. The set is a partial capability count, not proof that a deck lacks draw or is ready for a Boss.

Both arms used corrected native gameplay `7476a81954020087da31d41d16fddf475746ec2d`, Ironclad A0, frozen G7/ArmG and MCTS2000. Fresh allocation excluded 23,176 prior USED/RESERVED/unknown-purpose seeds from 91 manifests, including all five Round018 pools. Seventy new mutually disjoint seeds were reserved; only the registered Train10/Probe10/Dev30 pools were executed. Unused pools remain reserved and must be excluded from future allocation.

The selector, native type/effect parser and reviewed source files, prefix helper, binding/checkpoint, runtime and evaluators, all manifests and exclusion inventory were frozen before outcomes and reverified after all stages without starting episodes. No code or rule change occurred between stages. H23 remained stopped; no old seeds were reopened.

## Complete paired results

| Stage | Pairs | Episodes | G7 wins | H24 wins | C-only | P-only | Net | Discordant | Exact one-sided sign p | Overrides | Effective seeds | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Train | 10 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 9 | 7 | Advance |
| Probe | 10 | 20 | 1 | 2 | 1 | 0 | +1 | 1 | 0.5 | 6 | 4 | Advance |
| Dev | 30 | 60 | 6 | 4 | 0 | 2 | -2 | 2 | 1.0 | 17 | 15 | Stop: negative net |

Across all three different stage pools, descriptive totals are G7 7/50 (14%) and H24 6/50 (12%), net -1; C-only 1, P-only 2, discordant 3, one-sided sign p 0.875. This aggregate is not a confirmation result. In Dev, four pairs were both victories, 24 both defeats and two G7-only victories. Zero discordance in Train is not evidence of equivalence. The positive Probe signal did not persist in Dev.

All three stages are COMPLETE and their hard guards passed. Train and Probe reached the preregistered effective-coverage minimum of three and nonnegative-net requirement. Dev completed all 30 pairs but failed the positive-net selection requirement. No formal confirmation is warranted from this result.

## Integrity, diagnostics and verification

All 101 terminal records completed; all 100 paired trace episodes passed canonical legal-action validation. Illegal action, crash, timeout and communication error counters were each zero throughout. First-candidate trace-off replay matched outcome, override count and all 488 action signatures; it does not enter the paired denominator. The evaluator recorded artifact payloads of 80,034,163 bytes for Train, 108,253,674 for Probe and 300,105,644 for Dev, within the existing limits. Full raw states, exact actions, traces, terminal records and provenance remain private.

Every override was recomputed from the frozen selector. Effective coverage required matching recorded encounter/combat/noncombat prefix, pre-action state, choices and recommendation through the first changed reward, followed by exact accepted-card multiset checks preserving all existing cards. Before freeze an independent reviewer found that differing non-deck state alone could incorrectly count as effective; that proof gap was fixed and regression-tested before any episode. Recorded prefix matching is not serialization of hidden native RNG state.

| Stage | Mean terminal floor delta | Floor higher/equal/lower | Mean final HP delta | HP higher/equal/lower |
| --- | ---: | --- | ---: | --- |
| Train | -0.8 | 2 / 5 / 3 | 0 | 0 / 10 / 0 |
| Probe | +1.8 | 1 / 8 / 1 | +2.8 | 1 / 9 / 0 |
| Dev | -1.2 | 3 / 22 / 5 | -1.3667 | 1 / 27 / 2 |

Deltas are candidate minus G7, and are secondary diagnostics under Issue #19. No outcomes or regressions were omitted. The result does not establish which specific combat interaction explains the two G7-only wins or that draw cards are generally harmful.

Validation: 128 tests plus 2 subtests passed; corrected native resource smoke passed with zero episodes. A separate completion calculation, without importing the stage-summary calculation, re-aggregated every paired record's wins/C-only/P-only/net/exact p, override/effective counts, pair completeness and terminal distributions; all matched. Post-run source/runtime/pool freeze verification passed. The independent review agent hit its usage limit after reporting the pre-freeze proof gap, so final independent-agent source/completion review was not obtained. No independent full raw-trace replay was performed.

Next research should diagnose the recorded preparation/combat interactions on Train evidence and form a new prospectively registered hypothesis. Do not tune H24 from Probe/Dev outcomes or append samples to reverse this result. G7 remains Champion. Formal confirmation, actual STS1 smoke/pilot/acceptance/Victory and win-rate improvement remain NOT_RUN/NOT_VERIFIED; complete hidden RNG equivalence and independent raw-trace replay remain NOT_VERIFIED. Provider usage NOT_MEASURED. Original dirty parent workspace was preserved. The report commit follows runtime completion and is not the gameplay freeze identity.
