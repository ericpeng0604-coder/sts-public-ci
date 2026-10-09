# Round 008 H16 Train10 report

## Result

`NOT_VERIFIED_IMPLEMENTATION_COVERAGE`. H16's paired simulator execution completed, with G7 and the recorded H16 candidate each winning 0/10; Candidate-only / Parent-only / net = 0/0/0, 0 discordant pairs, and one-sided exact sign-test p=1.0. The candidate did not advance to Probe10. A card-ID mismatch discovered after the run means these outcomes are not a valid test of the intended strategy intervention.

H16's 3,194 candidate combat decisions contained 3,117 `visible_intent_not_lethal` decisions and 77 decisions labeled `no_legal_defend`. The classifier recognized abbreviated IDs such as `DEFEND_R`, while the simulator emits the native Ironclad ID `DEFEND_RED`; those 77 labels therefore do not establish that no legal Defend existed. A retrospective train-only counterfactual found `DEFEND_RED` in hand on 20 decisions and legal on 17, but no legal red Defend closed the projected deficit on this pool. H16's reason coverage is invalid, and the consumed pool was not rerun.

## Frozen identities and data provenance

- Candidate local source commit: `0838f660eaf02754af21d29490846de76b88d521`.
- Simulator policy source SHA-256: `23f7b8891a81f0c95e0ea128530738580148a611079f5b40a734101be9bac252`.
- Candidate evaluator SHA-256: `7093887a3fccc506643f427e8fa20a7e02ca6f4957274f7cea0bfde785dec7b3`.
- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Pinned native binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`.
- ArmG source / vocabulary SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b` / `832e199c359af8408ea430ffa3f9fcdc68f32533f7292bb102848d3fb558eb6a`.
- Round008 `train_hypothesis_2` manifest SHA-256: `a71b491d582dfd358c9fc48c0a3f3e9f2a907e3623a447aff598f19417eb2190`.
- Exclusion inventory: 81 source manifests, 23,036 unique IDs; SHA-256 `0d47f8956ad9f962ede5bac7bbd99931bbcfb1f7e9892dfbfb33259f921749be`. Numeric IDs remain private.
- MCTS budget: 2,000 per arm. PPO/Teacher collection and all Gate/Fresh result use were disabled.
- A preflight-only invocation initially rejected an unverified candidate-commit argument; it created no run output or usage record and consumed no episode. Re-running with the exact `HEAD` SHA passed.

The user-supplied Actions ZIP SHA-256 was `3f4fc50b9452138e50ed5274cc7ac71a9ad011077785279de7834a478c934e66`, matching the previously registered G7 archive. The pinned local G7 checkpoint matched the recorded G7 hash. PPO critic/replay members were not read or used.

## Safety, completeness, and artifacts

- Episodes: 20 complete paired episodes across 10 unique train seeds.
- G7/H16 terminal floors and final HP: equal in all 10 pairs; the registered retention guard passed.
- Complete terminal and legal trace records: 20/20. Illegal actions / crashes / timeouts: 0/0/0. Communication errors: `N/A_LOCAL_SIMULATOR`.
- Private artifact manifest: 41 files, 88,948,934 bytes; SHA-256 `64bc960314c6ebff9afc74a50401eb82fde33b8c487413623bf7bf214c6846b4` over the canonical JSON file-entry list.
- Private summary SHA-256: `fca4e539b5507f5fa31f1c41bbdfa182c77008c0ad3bc2d04c3c6150a8c96bc6`; all 41 recorded artifact sizes and hashes were independently rechecked.
- Raw traces, per-episode evidence, numeric seed IDs, and the usage ledger remain in the private Temp evaluation directory.
- Focused tests: 22 passed (`test_sts1_h16_lethal_defend.py`, `test_sts1_h15_trace_audit.py`, `test_sts1_h7_potion_trace.py`). Torch emitted a NumPy-unavailable warning; pinned Torch, native binding, ArmG, and G7 checkpoint loaded, and preflight/evaluation completed.

## Boundaries and next step

No Probe10, Dev30, confirmation, Gate/Fresh run, or real-game run occurred. Gate/Fresh results remain `NOT_READ`; real-game remains `NOT_RUN/NOT_VERIFIED`. G7 remains Champion; no promotion or deployment occurred.

The original trace classifier did not accurately identify simulator-native Defends. H16 Train10 is consumed and will not be rerun; its allocated Probe/Dev pools remain unused and excluded. H17 used the same faulty ID mapping and is also `NOT_VERIFIED_IMPLEMENTATION_COVERAGE`. H18 tests only the native-ID recognition repair on a newly generated, disjoint Round009 train pool; it does not reuse H16/H17 seeds.
