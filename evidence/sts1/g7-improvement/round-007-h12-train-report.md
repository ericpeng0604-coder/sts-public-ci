# Round 007 H12 Train10 — complete, no Probe

## Registration and provenance

- Issue #19 preregistration/scope amendment: comment 6079975390. H12 used only the new `train_hypothesis_1` pool; Probe10 and Dev30 remain untouched.
- Candidate commit: `9dc78b8395a4a6613b65a7258ada456ccc997259`; simulator source SHA-256 `34093825bc8c3f03d090d59a67cefea237dc4af076fb4a76eeaed3ce08d58803`; evaluator SHA-256 `fe64a9f21bd90064f3c785761b1531f7fc769ee527add8cb7b3b938f88a946e5`.
- Train pool manifest SHA-256 `5d06f80b952af7bac04fc01cd35e3711eaa334194021900305c07db7fbebeaf2`; extended exclusion inventory SHA-256 `5ab4074279680148978cac38588285598f3a0b5e3eac1c759d6b3236f91e65eb`; Round007 ledger SHA-256 `0d18d7ddb662d705280c13e9a1b479d80695158c04f940b52f7ecf61e5d2c37d`.
- Pinned parent G7 checkpoint SHA-256 `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`; Actions ZIP SHA-256 `3f4fc50b9452138e50ed5274cc7ac71a9ad011077785279de7834a478c934e66`; simulator binding SHA-256 `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`; ArmG source SHA-256 `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`; MCTS budget 2,000 per arm.
- Pools passed pairwise and exclusion-inventory disjointness checks. Seed IDs and raw traces remain private.

## Results

- Ten complete paired seeds: G7 0/10 wins; H12 0/10 wins; C-only 0; P-only 0; net 0; discordant 0; one-sided exact sign p=1.0.
- The rule was eligible and overrode G7 22 times. Candidate and parent both defeated on every pair. Terminal-floor comparison: candidate lower on 1 pair, equal on 8, higher on 1. Final HP was never lower. The preregistered strict retention guard failed, so Probe was not run.
- One candidate trace-off replay matched the trace-on summary and selected-action signature. All 21 episodes were complete; illegal actions, crashes, timeouts, and offline communication errors were 0. All 20 paired trace files passed legal-action/terminal validation.
- Trace coverage across the paired parent and candidate arms: 5,608 combat decisions, 324 encounters, 1,450 noncombat decisions, 668 route decisions, and 8,852 potion snapshots.
- Private artifact inventory: 44 files, 79,001,481 bytes; manifest SHA-256 `2a391961f5bcd0d6269742c74aad57ac5c612b305296d8fa2ca68b686a8adb28`.

## Bounded train-only diagnosis and next hypothesis

The 22 overridden recommendations comprised 17 cards not yet in the deck and 5 duplicates. G7's recorded score exceeded Skip on all 22, by 4.256495–13.548145 points (median 8.638026). In the only lower-floor pair, H12 skipped three novel G7-recommended cards at deck size 30; their G7-over-Skip margins were 5.1891–8.3402. This is train-only evidence that the broad rule can suppress a strongly preferred novel card; it does not establish that duplicate cards should be skipped.

Next, H13 is preregistered on the disjoint `train_hypothesis_2` pool (manifest SHA-256 `24279ac7bc8139efaa5298eb38291ca9ff7326c18e575d3a1a2e57305a19cbbd`): keep the same deck-size boundary, but override only when G7's recommended card is already present in the decision-time deck. This isolates duplicate-card selectivity and avoids a threshold sweep. H13 must independently pass complete/legal/safety checks, trace-on/off invariance, at least one eligible override, nonnegative train net, and no paired lower terminal floor or final HP before Probe can be considered.

No Probe, Dev, confirmation, Gate/Fresh execution, real-game run, promotion, or deployment occurred. G7 remains Champion. The local H12 source commit has not yet synchronized to the remote branch; the report is research evidence, not a completed GitHub delivery.
