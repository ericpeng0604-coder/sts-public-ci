# Round 008 H17 Train10 report

## Result

`NOT_VERIFIED_IMPLEMENTATION_COVERAGE`. H17's paired simulator execution completed on the registered `train_hypothesis_3` pool: G7 and the recorded H17 candidate each won 0/10; Candidate-only / Parent-only / net were 0/0/0, there were no discordant pairs, and the one-sided exact sign-test p-value was 1.0. A card-ID mismatch discovered after the run means these are not a valid estimate of the intended strategy intervention.

Across candidate traces, 2,644 decisions had nonlethal visible intent and 52 were labeled `no_legal_defend`. The helper recognized abbreviated IDs such as `DEFEND_R`, while the simulator emits native Ironclad ID `DEFEND_RED`, so the 52 labels do not establish that no legal Defend existed. A retrospective train-only counterfactual found `DEFEND_RED` in hand on 21 decisions and legal on 18; six legal red Defends closed the projected deficit, and five nonselected choices across three pairs were potential overrides under the registered rule. This is a post-run hypothesis audit only, not a win-rate effect or a replacement evaluation. H17 did not advance to Probe10, and its consumed train pool will not be rerun.

H16 used the same faulty ID map. Its trace classification is also invalid; retrospective analysis found no legal red Defend that closed the projected deficit in that consumed train pool. H16 and H17 remain preserved as `NOT_VERIFIED_IMPLEMENTATION_COVERAGE`, not null strategy tests.

## Frozen identities and seed provenance

- Local H17 experiment source commit: `929898121b8db8fcb3e6c4b6230d114a351199d6`.
- GitHub API code commit: `5cf5879a8be50d6ec633d3cce8211c7ff2555f43`; tree `fe33e629b8db2707ee6749715cd61acfd18efc5d` matches the local commit tree, its parent is H16 `484fd8b79d6665b2ec25a2d59b10f2c133d9ef30`, and both changed blob SHAs match local Git.
- Candidate evaluator SHA-256: `[PRIVATE_SHA256_REDACTED]`.
- Simulator policy source SHA-256: `[PRIVATE_SHA256_REDACTED]`.
- G7 checkpoint SHA-256: `[PRIVATE_SHA256_REDACTED]`.
- Native simulator binding SHA-256: `[PRIVATE_SHA256_REDACTED]`.
- ArmG source / vocabulary SHA-256: `[PRIVATE_SHA256_REDACTED]` / `[PRIVATE_SHA256_REDACTED]`.
- H17 train pool: 10 seeds, manifest SHA-256 `[PRIVATE_HASH_REDACTED]`; MCTS budget 2,000 per arm.
- The read-only exclusion preflight checked 81 source manifests and 23,036 unique IDs; inventory SHA-256 `[PRIVATE_HASH_REDACTED]`. Numeric IDs remain private. Gate/Fresh data was read only for the authorized seed-ID overlap exclusion; no outcomes, traces, decisions, or labels were used.
- The private H17 allocation record was written before preflight, as the first usage-ledger entry. It records H16's completed train allocation, the unrun H16 Probe/Dev association, and the H17 pool hashes; it contains no numeric seed IDs.

## Safety and artifact audit

- Episodes: 20 complete paired episodes across 10 unique train seeds.
- Complete terminal records / legal traces: 20/20. Illegal actions / crashes / timeouts: 0/0/0. Local simulator communication errors: `N/A_LOCAL_SIMULATOR`.
- Floors and final HP were equal in all 10 pairs; the registered retention guard passed.
- Private artifact manifest: 41 entries, 73,214,714 bytes; canonical entry-list SHA-256 `[PRIVATE_HASH_REDACTED]`.
- Private summary SHA-256: `[PRIVATE_HASH_REDACTED]`. All 41 artifact sizes and SHA-256 values were independently rechecked. The private usage-ledger SHA-256 is `[PRIVATE_HASH_REDACTED]`.
- Torch emitted a nonfatal warning that NumPy was unavailable; pinned simulator preflight and all 20 episodes completed successfully.
- Focused H16/H17 runner, H15 trace, and H7 trace tests: 27 passed.

Raw traces, per-episode evidence, and seed IDs remain in private Temp storage and are not included in this report or the repository.

## Next step and boundaries

H17 Train10 is consumed. H17 Probe10 and Dev30 were not run; they remain excluded and will not return to training. H18 applies the same registered lethal-intent rule with only the simulator-native `DEFEND_RED` identifier recognized, on a fresh Round009 pool. H16/H17 do not establish a positive or null strategy effect. No confirmation, Gate/Fresh execution, or real-game run occurred. G7 remains Champion; there was no promotion or deployment.
