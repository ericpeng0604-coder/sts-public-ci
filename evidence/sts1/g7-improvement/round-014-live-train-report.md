# Round014 live-resource Train diagnosis: complete, no paired improvement claim

Issue #19; Draft PR #32. Frozen collection code:
`180ccea4461f8a787b86cb520f917844bdc641d4`, based on PR #31's corrected
live combat resource observation. Preregistration was posted before episodes:
https://github.com/ericpeng0604-coder/sts-public-ci/issues/19#issuecomment-6106087263

## Result and integrity

- Ten fresh unchanged-G7 Train episodes: **0/10 victories**, ten complete defeats.
  This is a diagnostic sample, not a paired comparison; it cannot establish a
  regression or improvement relative to the historical different seeds.
- Exactly one first-seed trace-off replay: **PASS**, matching complete selected
  action signatures/counts and terminal summary. Actual budget: **11 episodes**.
  No completed game was rerun beyond this prospective replay.
- Illegal, crash, timeout and communication counts: **0**. Complete canonical
  action/ordinal validation, provenance, terminal and live-resource guards passed.
- Pinned gameplay commit `7476a81954020087da31d41d16fddf475746ec2d`, MCTS2000.
  Corrected r6 binding identity, unchanged G7/ArmG source/vocabulary/checkpoint,
  collection dependencies and pool provenance were frozen and verified privately.
- Exclusions: **66 source manifests / 22,826 unique IDs**. Seventy new Round014
  IDs in five mutually disjoint pools were reserved before collection. Only the
  first Train10 pool was consumed; every other pool remains reserved and must be
  excluded from future rounds. None can be reopened for another hypothesis.
- Raw evidence retained outside Git: **21 ndjson files / 62,310,125 bytes**.
  Private IDs, generation keys and provenance fingerprints are storage-only.

## Bounded diagnosis

Read-only file Inventory preceded explicitly scoped **Sample(10 Train traces)**:
36,857,494 bytes, 3,836 combat decisions. No Probe/Dev/formal trajectories or
outcomes were opened for hypothesis formation. Complete raw records remain intact.

1. **Confirmed resource interpretation correction; no proven loss causality.**
   All 3,836 decisions have valid live BattleContext inventory. Legacy run slots
   differ from live combat slots in 1,037 decisions; these are expected stale
   GameContext observations retained with explicit scope, not missed potion uses.
   Twelve End Turn decisions have visible summed intent damage at least HP plus
   block. Only three have legal potion-use actions: Gambler's Brew with an empty
   hand, Liquid Memories against Time Eater, and Weak Potion against Spire Growth.
   Potion availability alone does not establish survival or a winnable continuation.
   Intent damage is a warning signal, not a complete damage/death calculation.

2. **Repeated actionable investigation surface: End Turn with playable cards.**
   Of 650 End Turn decisions, 170 retain at least one canonical legal Play Card
   action, across all ten traces. Of these, 119 have positive energy and 83 have a
   legal zero-cost card. These overlapping diagnostics are not error counts.
   Card identity was matched by exact one-based native hand index to recorded
   hand position; inaccessible cards, status cards and ordinal aliases were not
   counted as alternatives. Nine of the twelve damage-warning End Turns have zero
   energy; one nevertheless has legal Panic Button. Time Eater's warning state
   retains card actions including Bandage Up and Armaments; Spire Growth retains
   Intimidate/Finesse/Seeing Red and other actions. Native Panic Button also applies
   two turns of No Block: a proposed rescue needs subsequent-state validation.
   The leading next investigation is an earlier, concrete tactical choice with a
   clone/replay counterfactual, not indiscriminately playing every remaining card.
   MCTS value/search limitations remain unmeasured; no tree-value claim is made.

3. **Boss preparation is a second hypothesis, with observational support only.**
   Seven losses terminate at floor50, two at floor33 and one at floor47.
   Six final encounters start above 80% HP; the other four start below 60%.
   Low entry HP alone therefore does not describe the entire sample. Deck scaling,
   draw/block balance, relic choices and earlier resource expenditure are plausible
   contributors requiring decision-time and paired evidence. No deck/relic rule
   has been selected from these terminal labels.

## Validation, delivery and limits

29 focused tests plus 2 subtests passed: the collector, live resources, H7 trace
validation and seed ledger. Initial test failures came from inaccessible inherited
Windows TEMP storage; rerunning with a task-specific writable temporary directory
passed without modifying prior data or permissions. Collection then passed its
private identity/disjointness preflight and the complete-episode replay guard.

Changed scope: one STS1 collector, one focused test file, preregistration and this
report. No simulator gameplay/policy changes, G7 weights/config changes, PPO/STS2
changes, formal-data use, paid API calls, deployment, promotion or merge.

G7 retained. Candidate-only/parent-only/net/p are **NOT_APPLICABLE** to this
unpaired baseline collection. New targeted counterfactual, candidate Train/Probe/
Dev, 200-pair confirmation and real-game acceptance remain **NOT_RUN**;
win-rate improvement **NOT_VERIFIED**. Provider token usage **NOT_MEASURED**.
Next: preregister one causally testable early tactical decision on entirely new
paired Train seeds; retain null/regression outcomes and apply the existing gates.
