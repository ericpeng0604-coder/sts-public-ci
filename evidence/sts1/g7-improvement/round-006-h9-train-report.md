# Round 006 H9 train result - null, safe, complete

- Date: 2026-10-09
- Frozen candidate commit: `2bd7487b99523dc2e5f00a3baf1d3062b3c5156c`
- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`
- Simulator commit: `7476a81954020087da31d41d16fddf475746ec2d`
- Candidate source SHA-256: `a9ac8c65ac665c2e75f7e1e807dd140d67f3532acd092b60b5d5a55092cadf72`
- Evaluator SHA-256: `ef7ec2aec13f73cef7656b96b3f0ca7cbcbf72b3e2baf6435e3cc0c250239ca9`
- Native binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`
- ArmG source SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`
- Search budget: MCTS-2000 for both arms.

## Registered data and result

- Train pool: 10 paired seeds; manifest SHA-256 `fe704802b804a2ffe41fd7690f43bf1a810d9f8edd6721202245b8e194559335`.
- Episodes: 20 paired-arm episodes plus one trace-off replay, 21 total.
- G7 wins: 2/10; H9 wins: 2/10.
- Candidate-only / Parent-only / net: `0 / 0 / 0`; discordant pairs: 0; exact one-sided sign `p=1.0`.
- H9 made one legal effective potion override. This did not change any paired win result.
- Trace-on/off invariance: passed.

Trace coverage across the H9 arm: 3,196 combat decisions, 180 encounters, 811 noncombat decisions, 377 route decisions, and 5,008 potion snapshots.

## Safety and retention

All 21 episodes were complete and legal. Illegal / crash / timeout / local simulator communication errors: `0 / 0 / 0 / 0`. Every paired terminal floor, final HP, and remaining potion-slot count matched. Both arms used 65 potion actions. Mean terminal floor was 39.5, mean final HP 5.3, and mean remaining potion slots 0.4 in both arms.

Private artifact inventory: 42 files, 89,784,476 bytes; manifest SHA-256 `bf3450af4c004acaa561fbca518e93038785cf36fea1540a2f8b0fe7bf67e101`. Raw traces and seed IDs remain private.

## Decision

This is a safe null train result, not a win-rate improvement. The predeclared nonnegative-train gate allowed Probe10; the candidate stayed frozen. No confirmation was run. Gate/Fresh data were not read or used; real-game verification, promotion, and deployment were not run. G7 remains Champion.
