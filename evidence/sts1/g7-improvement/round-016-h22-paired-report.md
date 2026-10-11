# H22 Train10 / Probe10 / Dev30: no paired win gain

Issue #19, Draft PR #34. Frozen runtime code
`5a90499068cfb785510c24acd5d2acf20c1c913f`. Preregistered before episodes:
https://github.com/ericpeng0604-coder/sts-public-ci/issues/19#issuecomment-6106465259

| Stage | Pairs | G7 wins | H22 wins | C-only | P-only | Net | Exact one-sided p | Effective changed seeds | Overrides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 10 | 2 | 2 | 0 | 0 | 0 | 1 | 4 | 5 |
| Probe | 10 | 3 | 3 | 0 | 0 | 0 | 1 | 5 | 15 |
| Dev | 30 | 4 | 4 | 0 | 0 | 0 | 1 | 19 | 32 |

All 50 pairs completed. Descriptive combined wins are **9/50 (18%) in each arm**;
this combines exploratory stages and is not confirmation or an equivalence
claim. Train and Probe passed their effective coverage, nonnegative net and hard
guard gates. Dev failed its positive-net gate: **confirmation NOT_RUN**.
G7 retained; win-rate improvement **NOT_VERIFIED**. Do not extend the sample,
relax the rule or rerun the same hypothesis to search for a favorable result.

Exactly **101 episodes**: Train21, Probe20, Dev60. The one prospectively scheduled
first-candidate trace-off replay passed full action signature/count and outcome
invariance. All illegal, crash, timeout and communication counts are zero;
all 101 terminals complete, all 100 paired traces legal. Each override was
revalidated by the frozen selector and native ordinal. Effective coverage required
matching prefix/state and a changed afterstate. Independent summary recount
passed for wins, C-only/P-only/net/p, effective changes, overrides, floor/HP,
episode counts, safety, identity consistency and disjoint stage seeds. Both arms
used the unchanged corrected native binding, frozen G7/ArmG and MCTS2000.

| Stage | Terminal floor lower/equal/higher | Final HP lower/equal/higher |
| --- | --- | --- |
| Train | 0/9/1 | 0/10/0 |
| Probe | 0/10/0 | 0/9/1 |
| Dev | 3/26/1 | 1/29/0 |

Dev floor deltas range -17 to +17, mean -0.9; HP deltas range -23 to 0,
mean -0.7667. These are secondary diagnostics, not additional selection vetoes.
Artifact-manifest byte totals: Train119,814,584; Probe115,046,486;
Dev326,049,143. These totals cover retained stage artifacts, not just traces.

The strongest conclusion is that the fixed rule changed actions often enough
to test, yet none of the changed outcomes became a win. It does not establish
that all early defense is ineffective. Three cause assessments remain bounded:

1. H21's free-defense intervention was sparse (1 effective Train seed); H22
   addressed coverage, reaching 19/30 Dev seeds, but coverage alone did not
   create wins. This is directly measured.
2. Earlier Train-only diagnosis found repeated Boss deaths and heterogeneous
   encounter-start HP. Isolated combat defense may be insufficient compared
   with deck preparation and resource planning; this remains a hypothesis.
3. The stale run-context potion observation was corrected and the live battle
   inventory validated before this round. Its causal share of losses remains
   unknown; no potion-rescue or model-quality claim follows from the fix.

The next distinct research step should use fresh Train-only evidence to assess
Boss/deck or resource preparation, then preregister a new mechanism and new
disjoint pools. Probe/Dev decision traces must not be mined for the next policy.

Private inventory:76 sources /22,966 excluded IDs. Five new disjoint Round016
pools reserved70 IDs before collection; Train1, Probe and Dev consumed50 IDs.
Train2/3 remain reserved and excluded from future rounds along with all other
Round016 pools. Raw states, exact legal actions, terminal/provenance evidence,
private identities and keys retained outside Git; no IDs or private hashes
published. No held-out tuning, formal outcome reuse or automatic extra episodes.

137 focused tests plus2 subtests passed before episodes. Runtime preflights,
trace invariance and independent recount passed; source hashes stayed frozen
through all stages. Final changes are confined to the Issue's STS1 paths and
whitespace review uses command-local cr-at-eol without normalizing frozen source
bytes. Model-training retention unchanged. Parent-gap/residual metrics
NOT_APPLICABLE; MCTS tree scores and provider tokens NOT_MEASURED. Real-game
acceptance NOT_RUN. No G7/PPO/STS2 changes, promotion, deployment, paid API or merge.
