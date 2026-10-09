# Round 007 H13 Probe10

## Result

H13 ran the preregistered 10 paired heldout seeds using MCTS-2000 per arm, plus one candidate trace-off replay (21 episodes total). The simulator recorded the pool under the `heldout_internal` seed contract. G7 won 1/10 and H13 won 1/10. Candidate-only / parent-only / net = 1 / 1 / 0; discordant pairs = 2; one-sided exact sign p = 0.75. This is not a positive win-rate signal.

The candidate made 20 eligible duplicate-only Skip overrides. Trace-on/off invariance passed. All 21 episodes were complete; all 20 paired traces had complete legal-action records; illegal actions, crashes, timeouts, and local-simulator communication errors were 0. The strict retention guard failed: candidate terminal floor was lower on 1 pair (equal 8, higher 1), and candidate final HP was lower on 1 pair.

The pre-registered progression gate stopped the branch before Dev30 because retention failed. Seed selection was not modified after outcomes, no samples were added, and H13 Probe evidence was not returned to training. Its result is used only for the preregistered stage decision.

## Frozen identities and pool

- Candidate policy commit: `22153109e0732586703cba741275b89df3b47091`.
- Candidate simulator source SHA-256: `f0ff9d751b8c100f3666433527ca2c30de09be585c0859aa6f3ac48d94057281`.
- Frozen candidate evaluator SHA-256: `1ff7d778f7486c906200ca850d40966e62e2ba00488b60166bdc17987bfbcc82`.
- Stage-aware evaluator commit: `a4bb3b37116edf4bbbd92c5e1936b36a773d5a9d`; executed-file SHA-256: `371abd2731adff8bf9b251fb3ac1c0feeee04ca036e7ae884b220e12dc333b77`.
- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Pinned native simulator source label: `7476a81954020087da31d41d16fddf475746ec2d`; binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`.
- ArmG source SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`.
- Probe pool manifest SHA-256: `97540602fbdbf5a69dc30ee6845fd054c98b3292f18e0b35c612f845c3bd8de3`.
- Exclusion inventory: 22,966 IDs, SHA-256 `5ab4074279680148978cac38588285598f3a0b5e3eac1c759d6b3236f91e65eb`; all five Round 007 pools passed pairwise disjointness and exclusion checks. Numeric seed IDs and raw traces remain private.
- Private Probe payload: 44 files / 101,867,172 bytes; artifact manifest SHA-256 `4655768d5b1c48dd1ff0e2baf58c22887ea3f404762b89ee34eca5fb61cef52d` (recomputed successfully).
- Trace coverage: 7,192 combat decisions, 428 encounters, 1,688 noncombat decisions, 798 route decisions, and 11,016 potion snapshots.

## Boundaries

No Dev30 or confirmation batch was run. Formal Gate/Fresh results, trajectories, decisions, and labels were not read or used; the private seed-ID inventory was used only for overlap exclusion. Real-game verification, promotion, deployment, and Champion replacement remain `NOT_RUN`. G7 remains Champion.
