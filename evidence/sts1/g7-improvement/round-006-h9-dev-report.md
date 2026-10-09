# Round 006 H9 Dev30 result - null, safe, complete

- Date: 2026-10-09
- Frozen candidate commit: `2bd7487b99523dc2e5f00a3baf1d3062b3c5156c`
- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`
- Simulator commit: `7476a81954020087da31d41d16fddf475746ec2d`
- Candidate source SHA-256: `a9ac8c65ac665c2e75f7e1e807dd140d67f3532acd092b60b5d5a55092cadf72`
- Native binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`
- ArmG source SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`
- Search budget: MCTS-2000 for both arms.

## Registered data and result

- Dev pool: 30 paired seeds; manifest SHA-256 `a9f195b41525aab30e8c58ec78db6564685d086bc2f7e54a83c4cf31b10572c5`.
- Episodes: 60.
- G7 wins: 3/30; H9 wins: 3/30.
- Candidate-only / Parent-only / net: `0 / 0 / 0`; discordant pairs: 0; exact one-sided sign `p=1.0`.
- H9 made two effective potion overrides. This did not change paired wins.

Trace coverage across H9: 9,642 combat decisions, 576 encounters, 2,414 noncombat decisions, 1,138 route decisions, and 15,076 potion snapshots.

## Safety and retention

All 60 episodes were complete and legal. Illegal / crash / timeout / local simulator communication errors: `0 / 0 / 0 / 0`. Trace-on/off invariance was not run for Dev.

All 30 pairs had the same terminal floor and final HP. Mean terminal floor was 39.5333 and mean final HP was 3.4 in both arms. Parent/Candidate potion-use actions were 249/251. Mean remaining potion slots were 0.2/0.1333; the paired count matched in 29/30 pairs and Candidate retained one fewer slot in one pair.

Private artifact inventory: 121 files, 269,373,150 bytes; manifest SHA-256 `485dc3cf862db05f421a8c4b43a6c86280f05fcb1bf72622a2a8ebac5f8dc00`. Raw traces and seed IDs remain private.

## Decision and boundaries

This is a null Dev30 result with no positive win-rate signal. No confirmation was run and H9 is not accepted as an improved Candidate. Gate/Fresh outcomes and traces were not read or used; real-game verification, promotion, and deployment are `NOT_RUN`/`NOT_VERIFIED`. G7 remains Champion.
