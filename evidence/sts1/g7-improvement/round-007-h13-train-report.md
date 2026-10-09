# Round 007 H13 Train10

## Result

H13 ran the preregistered 10 paired train seeds with MCTS-2000 per arm, plus one candidate trace-off replay (21 episodes total). G7 and H13 each won 0/10. Candidate-only / parent-only / net = 0 / 0 / 0; discordant pairs = 0; one-sided exact sign p = 1.0. This is a null train result, not a win-rate improvement.

The rule made 21 eligible duplicate-only Skip overrides across the train sample. The strict retention gate passed: candidate terminal floor was lower on 0 pairs, equal on 9, higher on 1; candidate final HP was lower on 0 pairs. The registered progression gate permits Probe because the action was exercised, net was nonnegative, retention passed, and trace-on/off matched. Probe authorization is a stage transition only, not a positive win-rate claim.

All 21 episodes were complete; 20 paired traces had complete legal-action records; trace-on/off invariance passed. Illegal actions, crashes, timeouts, and local-simulator communication errors were all 0. Trace coverage was 8,017 combat decisions, 450 encounters, 1,918 noncombat decisions, 878 route decisions, and 12,323 potion snapshots.

## Frozen identities

- Candidate policy commit: `22153109e0732586703cba741275b89df3b47091`.
- Candidate simulator source SHA-256: `f0ff9d751b8c100f3666433527ca2c30de09be585c0859aa6f3ac48d94057281`.
- Frozen candidate evaluator SHA-256: `1ff7d778f7486c906200ca850d40966e62e2ba00488b60166bdc17987bfbcc82`.
- Stage-aware evaluator SHA-256: `371abd2731adff8bf9b251fb3ac1c0feeee04ca036e7ae884b220e12dc333b77`.
- Pinned G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Native simulator source label: `7476a81954020087da31d41d16fddf475746ec2d`; pinned binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`.
- ArmG source SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`.
- H13 train pool manifest SHA-256: `24279ac7bc8139efaa5298eb38291ca9ff7326c18e575d3a1a2e57305a19cbbd`.
- Round 007 exclusion inventory SHA-256: `5ab4074279680148978cac38588285598f3a0b5e3eac1c759d6b3236f91e65eb`; private ledger SHA-256: `0d18d7ddb662d705280c13e9a1b479d80695158c04f940b52f7ecf61e5d2c37d`. The 22,966-ID exclusion inventory and all seed IDs remain private.
- Private train payload: 44 files / 111,762,697 bytes; manifest SHA-256 `922d3c95a665eae20570c79078afec73bc430606bfd8745d6b7c3582a30d469b` (recomputed successfully).

## Probe preregistration

The H13 Probe preregistration used the untouched Round 007 Probe pool (10 seeds; manifest SHA-256 `97540602fbdbf5a69dc30ee6845fd054c98b3292f18e0b35c612f845c3bd8de3`), 20 paired episodes plus one candidate trace-off replay, and the same frozen candidate, pinned G7, simulator, and MCTS-2000 budget. The stage-aware evaluator passed Probe/Dev seeds through the simulator's `heldout_seeds` contract; it did not mark them as training seeds. The preflight validated all pool manifests, disjointness, prior use, Candidate source at the frozen commit, complete legal traces, and pinned simulator inputs. Any integrity or safety failure is `NOT_VERIFIED` and stops the stage. The subsequent Probe result and gate decision are recorded in [the H13 Probe report](round-007-h13-probe-report.md).

Preflight history is retained: attempt 1 failed closed because the registered stage evaluator SHA had a two-character suffix typo; it started no episodes and did not consume the pool. Attempt 2 passed after the correction, then was superseded before any episodes when the runner gained an explicit preregistered `heldout_internal` contract check. Attempt 3 passed with the final evaluator SHA above; all attempts emitted no seed IDs.

## Subsequent H13 Probe10 result

After preregistration, H13 completed 10 paired heldout seeds plus one trace-off replay. G7 and Candidate each won 1/10; Candidate-only / Parent-only / net = 1 / 1 / 0; exact sign p = 0.75. Safety and completeness passed, but retention failed on one lower terminal-floor pair and one lower-final-HP pair. The registered gate therefore stopped before Dev30. See the separate Probe report for full evidence and hashes.

## Boundaries

H12 and H13 train used only their registered train pools. No confirmation batch, formal Gate/Fresh evaluation, or real-game run was performed. Gate/Fresh values, traces, decisions, and labels were not read or used; the protected seed-ID inventory was used only for overlap exclusion. G7 remains Champion. Promotion and deployment remain `NOT_RUN`.
