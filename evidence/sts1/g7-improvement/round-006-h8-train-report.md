# H8 Round006 Train10 report — 2026-10-09

## Status

`COMPLETE_NULL_NO_PROBE`. H8 did not change any combat action, so Probe10 and Dev30 were not run. G7 remains Champion. This is a simulator train result, not a win-rate improvement claim.

## Frozen inputs and provenance

- H8 Candidate commit: `032c4312e1f81c4add7bc5f9af4195951c1c0281`.
- Pinned G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Supplied Actions ZIP SHA-256: `3f4fc50b9452138e50ed5274cc7ac71a9ad011077785279de7834a478c934e66`; its `offline-champion.pt` is the pinned G7 checkpoint, not a new Candidate.
- Simulator source SHA-256: `02a1fa471e2a3eacff41660499756046700a922b39d52c20a16e138fd5e9429a`.
- Native binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`.
- ArmG source SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`.
- MCTS budget: 2,000 for both arms; PPO/Teacher collection disabled.
- Round006 exclusion inventory: 22,896 unique IDs, SHA-256 `07ca394a9f872f06c9654cf05ca88243745fc3f93b11defbdfe3b95e69155ca8`.
- H8 train pool: 10 seeds, manifest SHA-256 `848de4877ffbed275955f8fbf0a669824d044168e6014cea5e0d512201beec50`; disjointness preflight passed for all five Round006 pools and the complete exclusion inventory.
- Private train evidence: 42 files, 104,864,772 bytes; sorted-path/file-hash manifest SHA-256 `08c61c2c26b601cd6eb14cf4d8d3d76ced789742323f80ab31dd13233215fcec`. Raw seed IDs and traces remain outside the repository.

## Paired result

| Arm | Wins |
| --- | ---: |
| G7 | 2/10 |
| H8 | 2/10 |

Candidate-only wins: 0. Parent-only wins: 0. Net: 0/10. Discordant pairs: 0. Exact one-sided sign-test p: 1.0.

The run completed 10 paired seeds plus one H8 trace-off replay (21 episodes). All paired episodes and traces passed terminal/legal/MCTS validation. Illegal actions, crashes, and timeouts were 0; communication errors are not applicable to this local simulator. Trace-on/off invariance passed.

## Trace audit and hypothesis result

The retained H8 Candidate arm traces were revalidated locally with the H7 trace validator: 10/10 complete traces, 3,922 combat decisions, 209 encounters, 903 noncombat decisions, 434 route decisions, and 5,947 potion snapshots. H8 was enabled in all 10 Candidate summaries. Candidate and G7 action signatures were identical in all 10 pairs; rescue overrides: 0.

The original H8 aggregate printed zero trace-coverage totals because its rollup used key names different from the H7 validator's returned fields. The per-trace checks passed; this report uses a bounded deterministic reaggregation of only these 10 Candidate traces. Repair the rollup before using this evaluator for another stage.

H8's single-enemy lethal-only rule did not find an eligible rescue. Across 3,922 combat decisions, reasons were: not End Turn 3,261; multiple/no single living enemy 182; unknown attack intent 202; unsupported end-turn status 78; known attack not lethal 187; non-damaging attack 12. No Probe was run because the Candidate made no action change.

Two pre-episode attempts are retained in this handoff: an initial preflight exposed a mismatch between raw-file and canonical-JSON inventory hashes; after that fix, one launch exposed an ArmG constructor wiring error. Neither attempt created an output directory, usage event, or episode. The final frozen commit corrected both issues and passed preflight before the 21-episode run.

## Next hypothesis for preregistration

H8 train traces contain five supported single-enemy End Turn decisions classified as nonlethal where a legal Block/Blood Potion was present, across four paired traces, all from train pairs that ended in defeat. This is hypothesis coverage only; the exact HP preserved by potion use has not been established. Proposed H9: use a legal Block/Blood Potion only when the pinned simulator's exact projection shows it strictly increases player HP after the known attack, with the same fail-closed state/effect checks. Allocate the still-unused, pairwise-disjoint Round006 `train_hypothesis_2` pool (10 seeds) only after Issue #19 preregistration. Do not advance to Probe10 without a safe effective action change.

Gate/Fresh outcomes were not read or used; their seed IDs were used only in the authorized exclusion inventory. No confirmation, real-game, production, promotion, or deployment evaluation was performed. GitHub source synchronization remains pending; the local commit is not represented as pushed or merged.
