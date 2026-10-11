# H21 Train10: no win gain; insufficient effective coverage

Issue #19, Draft PR #33. Frozen code
`1d55f4319583ed7f58bba27d6a4d354d620fc6fc`. Preregistered before any episode:
https://github.com/ericpeng0604-coder/sts-public-ci/issues/19#issuecomment-6106349545

Ten complete new paired Train seeds: G7 **0/10**, H21 **0/10**; candidate-only0,
parent-only0, net0, discordant0, one-sided exact sign p=1. No equivalence claim.
Exactly20 paired episodes and1 prospective first-candidate trace-off replay:
**21 episodes**, replay PASS, full action signatures/counts and outcomes equal.

There was1 legal H21 override in1 effective changed seed, with matched initial
prefix/state and changed afterstate. Every override was revalidated by the frozen
selector and native ordinal. Required coverage3 was not reached; **Probe/Dev/
confirmation NOT_RUN**. Terminal floor/HP lower/equal/higher counts both0/10/0;
all paired terminal deltas0, retained as secondary diagnostics, not selection
vetoes. All illegal/crash/timeout/communication counts0, complete terminals21,
legal paired traces20. Hard guards passed. G7 retained; improvement NOT_VERIFIED.

The hypothesis's main limitation is sparse eligibility: it tests only free
defense, with conservative card-trigger exclusions. Its shadow activation on3
previous Train traces did not generalize to the new Train10 coverage gate. Do not
relax the frozen rule or add samples. The next distinct investigation is earlier
damage prevention with an affordable plain Defend before a nonlethal attack,
rather than a zero-cost or last-turn rescue. It requires new preregistration,
fresh pools, causal validation and the same gates before any improvement claim.

Private exclusion inventory71 sources /22,896 IDs; five fresh disjoint Round015
pools70 IDs reserved before collection. Train1 consumed; every other Round015 pool
remains excluded from future rounds. All raw evidence, identities and generation
keys preserved outside Git. No held-out outcome mining, G7/PPO/STS2 changes,
promotion, deployment, paid APIs or merge.

120 focused tests plus2 subtests passed, including H19/H20 regressions, H21 guards,
trace/resource/seed validation and collection checks. Runtime preflight and
complete-episode trace invariance passed. Scope review passed; whitespace checks
retain the existing Windows CRLF representation with command-local cr-at-eol
handling, without changing frozen source bytes. Model-training retention unchanged;
MCTS tree scores/provider tokens NOT_MEASURED. Real-game acceptance NOT_RUN.
