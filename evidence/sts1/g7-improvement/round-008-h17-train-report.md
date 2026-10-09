# Round 008 H17 Train10 report

## Result

`COMPLETE_NULL_NO_PROBE`. H17 independently replicated the frozen H16 lethal-intent Defend rule on the previously unused `train_hypothesis_3` pool: G7 and H17 each won 0/10. Candidate-only / Parent-only / net were 0/0/0, there were no discordant pairs, and the one-sided exact sign-test p-value was 1.0.

H17 recorded no legal action overrides. Across candidate traces, 2,644 decisions had nonlethal visible intent and 52 had no legal Defend. The preregistered train gate requires at least one legal override, so H17 did not advance to Probe10. This is a null coverage replication, not a win-rate improvement claim.

## Frozen identities and seed provenance

- Local H17 experiment source commit: `929898121b8db8fcb3e6c4b6230d114a351199d6`.
- GitHub API code commit: `5cf5879a8be50d6ec633d3cce8211c7ff2555f43`; tree `fe33e629b8db2707ee6749715cd61acfd18efc5d` matches the local commit tree, its parent is H16 `484fd8b79d6665b2ec25a2d59b10f2c133d9ef30`, and both changed blob SHAs match local Git.
- Candidate evaluator SHA-256: `8f80decae0229886c6bf278e7d8858f272d9e6bc75f367ef7225c245d249ca57`.
- Simulator policy source SHA-256: `23f7b8891a81f0c95e0ea128530738580148a611079f5b40a734101be9bac252`.
- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Native simulator binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`.
- ArmG source / vocabulary SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b` / `832e199c359af8408ea430ffa3f9fcdc68f32533f7292bb102848d3fb558eb6a`.
- H17 train pool: 10 seeds, manifest SHA-256 `fe8f0bc33ba50bac941ddb40278775cb0f36934df027b9e2c1df595b7b3d4b7d`; MCTS budget 2,000 per arm.
- The read-only exclusion preflight checked 81 source manifests and 23,036 unique IDs; inventory SHA-256 `0d47f8956ad9f962ede5bac7bbd99931bbcfb1f7e9892dfbfb33259f921749be`. Numeric IDs remain private. Gate/Fresh data was read only for the authorized seed-ID overlap exclusion; no outcomes, traces, decisions, or labels were used.
- The private H17 allocation record was written before preflight, as the first usage-ledger entry. It records H16's completed train allocation, the unrun H16 Probe/Dev association, and the H17 pool hashes; it contains no numeric seed IDs.

## Safety and artifact audit

- Episodes: 20 complete paired episodes across 10 unique train seeds.
- Complete terminal records / legal traces: 20/20. Illegal actions / crashes / timeouts: 0/0/0. Local simulator communication errors: `N/A_LOCAL_SIMULATOR`.
- Floors and final HP were equal in all 10 pairs; the registered retention guard passed.
- Private artifact manifest: 41 entries, 73,214,714 bytes; canonical entry-list SHA-256 `06e9bfd64821e2db61302e907f226e61f2557bb3c0421e13de5b41192a70f7e8`.
- Private summary SHA-256: `9fc6c8c588411d980b5fab18110c28e8d73a0b646b46920e240c89b71170d7ff`. All 41 artifact sizes and SHA-256 values were independently rechecked. The private usage-ledger SHA-256 is `97840e953e151c27c08f2283800888a352c78f4cda387c0771fbb35861ba1a5d`.
- Torch emitted a nonfatal warning that NumPy was unavailable; pinned simulator preflight and all 20 episodes completed successfully.
- Focused H16/H17 runner, H15 trace, and H7 trace tests: 27 passed.

Raw traces, per-episode evidence, and seed IDs remain in private Temp storage and are not included in this report or the repository.

## Next step and boundaries

H17 Train10 is consumed. H17 Probe10 and Dev30 were not run, and their pools remain unexecuted. H16 remains a completed null/coverage result; neither H16 nor H17 provides a positive strategy signal. Choose a different evidence-backed factor and a new, disjoint train pool for the next round. No confirmation, Gate/Fresh execution, or real-game run occurred. G7 remains Champion; there was no promotion or deployment.
