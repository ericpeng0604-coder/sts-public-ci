# Round017: unchanged-G7 Boss preparation facts

Issue #19; prerequisite Draft PR #35 at `7e90f86fe1631395d16198881770fe846f4f890f`.
This is a new Train-only diagnostic collection, not H23, a candidate or an improvement claim.

## Fixed prospective scope

- Exactly ten newly generated Train seeds and one first-seed trace-off replay: eleven episodes maximum. Complete evidence action signatures/counts and terminal result must match for the replay. No automatic retry or extra samples after outcomes.
- Frozen corrected native binding, unchanged G7/ArmG source/vocabulary/checkpoint, pinned gameplay `7476a81954020087da31d41d16fddf475746ec2d`, Ironclad A0, MCTS2000; no policy interventions, PPO/Teacher collection or model/API calls.
- Extend the latest Round016 exclusion inventory with all five Round016 pools, including every unused reserved pool. Generate five disjoint new pools with the existing seed ledger; consume only the first Train10. All other pools remain reserved and excluded forever.
- Freeze code commit, collector/profiler/validator/ledger source, native binding, model identity, generation key, inventory and pool ledger privately before episodes. Source, pool or runtime drift blocks preflight. Public records expose counts and public code identity, never private seed IDs or private provenance hashes.
- Preserve failed starts, partial evidence, complete raw states, canonical actions and terminal/provenance records outside Git. Every game must pass existing native legality, full trace/action coverage and terminal checks; illegal/crash/timeout/communication counts must all be zero.

## Read-only diagnostic view

After each validated trace, preserve private decision-time resource profiles: deck names/upgrades, HP/maxHP, gold, run potion slots, relic labels, legal candidate semantics and selected index, and the currently visible route. Encounter entries use live player HP and live BattleContext potion inventory, never stale GameContext inventory.

Keep Boss identity at noncombat decision time, future route coverage, card capabilities and readiness explicitly `NOT_VERIFIED`; do not infer them from a name, final floor, or the later enemy. A recognized Boss name at encounter entry is an observation at that time only; minions do not erase a recognized Boss entry. The canonical no-card reward skip is valid despite its selected index -1.

Aggregate baseline outcomes, terminal floors, decision screens, observed Boss entries, final encounter HP bands and unknown-field counts. These are unpaired diagnosis, not C-only/P-only/net/p or evidence that any preparation rule would gain wins. Raw traces remain the authoritative evidence, and the profiler is not a native replay validator.
Boss name aliases have synthetic coverage only before collection; `observed_boss_entries` is a matched-name count, not a verified complete Boss-conversion denominator. Original recommended indexes and scores remain in the raw trace rather than being duplicated into the profile. Independent read-only scope/schema/exclusion review found no blocking issue.

## Stage gate and next boundary

No candidate is selected by this collection. Following evidence review, a distinct mechanism needs a separate prospective registration, fresh disjoint paired Train/Probe/Dev pools, unchanged existing guards and the Issue's confirmation and real-game gates. Do not mine Probe/Dev/Formal decisions for this research or reopen previous reserved pools. G7 remains Champion.

Focused validation before collection: 71 tests and 2 subtests passed, including preparation facts/privacy, freshness/held-out rejection, source/pool drift, backward-compatible collection, seed ledger, live resources and trace mapping. Zero-episode native smoke PASS: five live slots, copy semantics, unchanged native state, complete legality and equal MCTS2000 recommendation.
Initial test failures were a missing standalone-import path, a wrong synthetic-test purpose expectation and inherited Windows TEMP permissions; fixes used the documented script import convention, correct existing `train` schema and task-local temporary storage. No pre-existing evidence or permissions changed.

Collection and full-episode invariance are NOT_RUN at registration. New policy, win gain, confirmation and real-game acceptance remain NOT_RUN/NOT_VERIFIED. Provider token usage NOT_MEASURED.
