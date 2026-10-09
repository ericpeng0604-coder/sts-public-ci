# Round007 H14 Train10 report

**Status: `COMPLETE_NO_PROBE`.** The registered retention gate failed, so no Probe or Dev episodes were run.

## Frozen inputs

- Candidate policy commit: `bef46c7ec7b02e0d4976eda112d3575244e4f053`.
- Candidate simulator source SHA-256: `981b33b0f39a0f139457185caa48dc84e93d179e4e0b67940cb2392890f1f72c`.
- Candidate evaluator SHA-256: `b399776ce3267b2a8e56882ac7e86cf991f83ef7bef471798123d74e0c6ee5c3`.
- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Pinned simulator binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`.
- ArmG source SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`.
- Train pool manifest SHA-256: `f566468ee12b169c0c4e3bc4f6628b0971dcdb5c9150b403baf3c597eb1098c9`; exclusion inventory SHA-256: `5ab4074279680148978cac38588285598f3a0b5e3eac1c759d6b3236f91e65eb` (22,966 IDs, private). The 10-seed pool was unique and disjoint from the exclusion inventory and all other Round007 pools. Seed IDs were not emitted.
- Preflight passed with `training_internal`, MCTS-2000, and 10 paired seeds. Torch 2.11.0+cpu, the pinned simulator binding, and ArmG imported successfully.

## Policy and paired result

H14 changed one factor: with decision-time deck size at least 30, it chooses the exact legal Skip only when G7 recommends a card already in the deck and every legal non-Skip reward card is also identified as already in the deck. If any option is novel or identity data is incomplete, it preserves G7.

The fixed workload was 10 paired G7/H14 runs plus one H14 trace-off replay (21 episodes total), with MCTS-2000 per combat. G7 and H14 each won 2/10. Candidate-only / Parent-only / net = 0 / 0 / 0; discordant pairs = 0; one-sided exact sign p = 1.0. This is a null train result, not a win-rate improvement.

H14 had 2 eligible overrides across 2 decisions. In a bounded Sample of the 10 Candidate train traces and paired terminal summaries, both overrides matched the parent's exact decision-time state; the parent selected its recommendation and H14 selected the legal Skip. One override pair was also the only lower-terminal-floor pair. This is a retention-risk association, not proof that the Skip caused the lower floor. The other 9 pairs had equal floors; no pair had lower final HP. No Probe was authorized by the failed retention guard.

## Safety, completeness, and artifacts

- Episodes complete: 21/21; paired legal traces: 20/20; trace-on/off equality: PASS.
- Illegal actions / crashes / timeouts / local simulator communication errors: 0 / 0 / 0 / 0.
- Candidate trace coverage: 7,774 combat decisions, 401 encounters, 1,758 noncombat decisions, 820 route decisions, and 11,711 potion snapshots.
- Private artifact manifest: 44 files, 103,911,649 bytes; SHA-256 `0a06dd1adc2e90ae73c41634fd6c4bcb8c4b898832e4b694886abc8845848168`. Every listed file size and SHA-256 was recomputed and matched; the manifest digest also verified.
- Focused tests: 43 passed across card-reward, stage-evaluator, simulator-trace, and potion-trace tests. Python compile and `git diff --check` passed.

Round007 episode usage is 84/140, leaving 56. H14 Probe and Dev, confirmation, formal Gate/Fresh, real-game verification, promotion, and deployment are `NOT_RUN`. Formal Gate/Fresh outcomes, traces, decisions, and labels were not read or used; the authorized seed-ID-only exclusion inventory remains storage-only. G7 remains Champion.
