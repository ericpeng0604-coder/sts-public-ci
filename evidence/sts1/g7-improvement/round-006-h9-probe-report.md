# Round 006 H9 Probe10 result - null, safe, complete

- Date: 2026-10-09
- Frozen candidate commit: `2bd7487b99523dc2e5f00a3baf1d3062b3c5156c`
- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`
- Simulator commit: `7476a81954020087da31d41d16fddf475746ec2d`
- Candidate source SHA-256: `a9ac8c65ac665c2e75f7e1e807dd140d67f3532acd092b60b5d5a55092cadf72`
- Native binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`
- ArmG source SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`
- Search budget: MCTS-2000 for both arms.

## Registered data and result

- Probe pool: 10 paired seeds; manifest SHA-256 `5b9d7e4f2aa22cb6e8db339a8b288f615876b7b0a96f5d8ff2513d6f4b43a95c`.
- Episodes: 20.
- G7 wins: 0/10; H9 wins: 0/10.
- Candidate-only / Parent-only / net: `0 / 0 / 0`; discordant pairs: 0; exact one-sided sign `p=1.0`.
- H9 made one legal effective potion override; it did not change any paired win result.

Trace coverage across H9: 3,341 combat decisions, 188 encounters, 754 noncombat decisions, 358 route decisions, and 5,047 potion snapshots. Trace-on/off invariance was not run for Probe.

## Safety and retention

All 20 episodes were complete and legal. Illegal / crash / timeout / local simulator communication errors: `0 / 0 / 0 / 0`. Each pair had identical terminal floor, final HP, and remaining potion slots; both arms used 68 potion actions. Mean floor was 37.2, mean final HP was 0, and mean remaining potion slots was 0.4 in both arms.

Private artifact inventory: 41 files, 88,089,166 bytes; manifest SHA-256 `6f550e23e156f6979d3204de0c4e097a64f97ee6658dea68e13f75e8346f7ec4`. Raw traces and seed IDs remain private.

## Decision

The preregistered safe, complete, nonnegative Probe gate allowed the one Dev30 evaluation. This Probe result is null and supplies no positive win-rate signal. The candidate remained frozen through Dev. No confirmation was run. Gate/Fresh data were not read or used; real-game verification, promotion, and deployment were not run. G7 remains Champion.
