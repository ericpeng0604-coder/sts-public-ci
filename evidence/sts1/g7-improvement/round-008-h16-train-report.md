# Round 008 H16 Train10 report

## Result

`NOT_VERIFIED_IMPLEMENTATION_COVERAGE`. H16's paired simulator execution completed, with G7 and the recorded H16 candidate each winning 0/10; Candidate-only / Parent-only / net = 0/0/0, 0 discordant pairs, and one-sided exact sign-test p=1.0. The candidate did not advance to Probe10. A card-ID mismatch discovered after the run means these outcomes are not a valid test of the intended strategy intervention.

H16's 3,194 candidate combat decisions contained 3,117 `visible_intent_not_lethal` decisions and 77 decisions labeled `no_legal_defend`. The classifier recognized abbreviated IDs such as `DEFEND_R`, while the simulator emits the native Ironclad ID `DEFEND_RED`; those 77 labels therefore do not establish that no legal Defend existed. A retrospective train-only counterfactual found `DEFEND_RED` in hand on 20 decisions and legal on 17, but no legal red Defend closed the projected deficit on this pool. H16's reason coverage is invalid, and the consumed pool was not rerun.

## Frozen identities and data provenance

- Candidate local source commit: `0838f660eaf02754af21d29490846de76b88d521`.
- Simulator policy source SHA-256: `[PRIVATE_SHA256_REDACTED]`.
- Candidate evaluator SHA-256: `[PRIVATE_SHA256_REDACTED]`.
- G7 checkpoint SHA-256: `[PRIVATE_SHA256_REDACTED]`.
- Pinned native binding SHA-256: `[PRIVATE_SHA256_REDACTED]`.
- ArmG source / vocabulary SHA-256: `[PRIVATE_SHA256_REDACTED]` / `[PRIVATE_SHA256_REDACTED]`.
- Round008 `train_hypothesis_2` manifest SHA-256: `[PRIVATE_HASH_REDACTED]`.
- Exclusion inventory: 81 source manifests, 23,036 unique IDs; SHA-256 `[PRIVATE_HASH_REDACTED]`. Numeric IDs remain private.
- MCTS budget: 2,000 per arm. PPO/Teacher collection and all Gate/Fresh result use were disabled.
- A preflight-only invocation initially rejected an unverified candidate-commit argument; it created no run output or usage record and consumed no episode. Re-running with the exact `HEAD` SHA passed.

The user-supplied Actions ZIP SHA-256 was `[PRIVATE_HASH_REDACTED]`, matching the previously registered G7 archive. The pinned local G7 checkpoint matched the recorded G7 hash. PPO critic/replay members were not read or used.

## Safety, completeness, and artifacts

- Episodes: 20 complete paired episodes across 10 unique train seeds.
- G7/H16 terminal floors and final HP: equal in all 10 pairs; the registered retention guard passed.
- Complete terminal and legal trace records: 20/20. Illegal actions / crashes / timeouts: 0/0/0. Communication errors: `N/A_LOCAL_SIMULATOR`.
- Private artifact manifest: 41 files, 88,948,934 bytes; SHA-256 `[PRIVATE_HASH_REDACTED]` over the canonical JSON file-entry list.
- Private summary SHA-256: `[PRIVATE_HASH_REDACTED]`; all 41 recorded artifact sizes and hashes were independently rechecked.
- Raw traces, per-episode evidence, numeric seed IDs, and the usage ledger remain in the private Temp evaluation directory.
- Focused tests: 22 passed (`test_sts1_h16_lethal_defend.py`, `test_sts1_h15_trace_audit.py`, `test_sts1_h7_potion_trace.py`). Torch emitted a NumPy-unavailable warning; pinned Torch, native binding, ArmG, and G7 checkpoint loaded, and preflight/evaluation completed.

## Boundaries and next step

No Probe10, Dev30, confirmation, Gate/Fresh run, or real-game run occurred. Gate/Fresh results remain `NOT_READ`; real-game remains `NOT_RUN/NOT_VERIFIED`. G7 remains Champion; no promotion or deployment occurred.

The original trace classifier did not accurately identify simulator-native Defends. H16 Train10 is consumed and will not be rerun; its allocated Probe/Dev pools remain unused and excluded. H17 used the same faulty ID mapping and is also `NOT_VERIFIED_IMPLEMENTATION_COVERAGE`. H18 tests only the native-ID recognition repair on a newly generated, disjoint Round009 train pool; it does not reuse H16/H17 seeds.
