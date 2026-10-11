# Round017 Boss preparation diagnostic completion

Issue #19; Draft PR #36. Frozen executable source: `0a2e5d275b2df0bbf046014fe0eb7930784e8611`.
The preregistration was committed and posted prospectively before collection. G7 remains unchanged and retained; this round does not select a candidate or claim improved win rate.

## Completed evidence

- Fresh Train10 plus one trace-off invariance replay: 11 episode starts and 11 completions, with no retries. Ten complete private preparation profiles and full raw episode/trace evidence retained outside Git. Other new pools remain reserved and unused.
- Baseline: 1 victory and 9 defeats. Terminal floors: 33 x3, 41 x1, 42 x1, 46 x1, 50 x3, 51 x1. These are unpaired observations, not an improvement estimate.
- Full-episode trace-on/off invariance PASS. Illegal actions, crashes, timeouts and communication errors: all zero. Frozen runtime/source/pool preflight PASS. Existing exclusion inventory: 23,036 USED/RESERVED seeds across 81 manifests.
- 871 noncombat decisions: map 411, rewards 214, shop 102, rest 80, events 64. Unknown decision-time Boss and unknown card-capability/readiness information are explicitly retained for all 871 decisions. Live potion inventory was available at every recorded encounter entry.
- Recognized Boss-name entries: 24. Final encounter HP bands: above 80% x6, 60-80% x1, below 60% x3. Name matching is not a proven complete Boss-conversion denominator.

## Preparation observations

These aggregates cover matched Boss entry names only; they are descriptive and do not establish causality.

| Act | Matched entries | Mean HP / max HP | Mean deck size | Mean upgraded cards | Mean held potions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 92.6% | 20.5 | 2.3 | 0.7 |
| 2 | 10 | 83.4% | 31.6 | 6.3 | 0.5 |
| 3 | 4 | 95.1% | 43.75 | 11.75 | 1.25 |

All four matched Act3 entries had at least 80.2% HP; their decks ranged from 38 to 47 cards. Three ended in defeat and one in victory. This makes deck content, draw consistency and reward selection useful next research targets. It does not prove that deck size caused defeat or that additional healing would be useless. Potion utility, scaling, draw, block, energy and burst remain unclassified.

The current rest-choice trace exposes numeric option indexes; it does not establish rest-versus-upgrade semantics. Do not label these indexes as actions without native enum verification. Shop screen counts likewise are decision counts, not unique visits or purchase counts.

## Next implementable stage

1. Add read-only native exposure of the currently visible Boss identity and explicit campfire/shop/reward semantics, retaining exact legal action identity. Verify against the native source and observed state; never derive earlier Boss knowledge from later combat.
2. Build an Ironclad card-capability mapping from verified game rules, including upgrades and conditional effects. Preserve unknown cards and interactions. Characterize the existing G7's reward acceptance, card removal and upgrade selection using the Train-only evidence already collected.
3. Choose one preparation intervention only after that characterization, register its trigger and exclusions prospectively, and collect fresh disjoint paired Train/Probe/Dev evidence. Do not combine reward, shop and campfire changes into one first hypothesis.
4. Apply Issue #19's existing advancement, frozen confirmation and separate real-game gates. No promotion follows from this diagnostic round.

## Verification and limits

Focused tests: 71 passed plus 2 subtests passed. Native resource smoke: PASS, zero episodes; live slots/copy semantics/native state/legal actions/MCTS2000 recommendation preserved. Independent pre-collection scope/schema/exclusion review: PASS. Independent completion review: PASS; all public aggregates match ten bounded private profiles, usage has eleven starts/completions and ten trace-on/one trace-off episodes, seven current source hashes match the freeze, and prior/current runtime identity locks agree. The reviewer did not re-read raw traces, independently replay actions or re-hash native/checkpoint binaries. Private seed IDs and private provenance hashes are not published.

No policy change, paired candidate evaluation, formal confirmation or actual-game acceptance was performed. Win-rate gain, complete Boss alias coverage, decision-time Boss visibility and card readiness are NOT_VERIFIED. Provider token usage is NOT_MEASURED; no model/API call was introduced. Original dirty work was preserved in the parent workspace.
