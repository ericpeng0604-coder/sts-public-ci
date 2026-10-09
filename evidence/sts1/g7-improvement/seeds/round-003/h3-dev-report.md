# H3 round-003 Dev30 result

**Status: COMPLETE_REAGGREGATED.** The simulator completed the preregistered 30 paired seeds plus one trace-invariance episode. Its first runner returned `NOT_VERIFIED` at final aggregation because it called a shared helper limited to 10 pairs. The original status and all raw evidence were preserved; the 30 pairs were not rerun. A corrected, read-only reaggregator independently checked the saved evidence and produced this result.

## Frozen setup

- Candidate policy commit: `009e233bb38f71a2df3aea611db0b8ab9b664aa1`.
- Run checkout commit: `4d171f5828ba56d2ec43434a2da1ccd269158943`.
- Held-out stage evaluator commit/source SHA-256: `48f60b04868714cbd026b90b12e7d68ef0a35c7e` / `46b6272d76f9c4a7cbd88a5608347993a7498ebd60068724f5c78040eabdda02`.
- Reaggregator source SHA-256: `a3c519a445b2e9b2235ced9c5353432ef05a8fb7cee657e07e2a751a7a7b13b6`.
- Dev30 manifest SHA-256: `1c5902006c427aee215e7b79eca65e58cf489ef125b18113a4b99c72e007d856`; seeds were resolved with the `heldout_internal` contract.
- G7 checkpoint, H3 policy source, pinned simulator/native binding, ArmG source, and MCTS-2000 matched preregistered identities.

## Result and checks

- G7: 3/30 wins; Candidate: 3/30 wins.
- Candidate-only / G7-only / net: 0 / 0 / 0; discordant pairs: 0; exact one-sided sign p = 1.0.
- 23 low-HP legal-potion opportunities produced 23 effective overrides; paired wins were unchanged.
- 61 episodes completed. All 60 trace-enabled terminal records were complete and legal; trace-on/off invariance passed. All 21,106 recorded combat decisions used MCTS-2000.
- Illegal actions = 0; crashes = 0; timeouts = 0. Communication errors are not applicable to this local simulator.
- The original `NOT_VERIFIED` usage record remains. A later `COMPLETE_REAGGREGATED` record references the immutable evidence inventory SHA-256 `b5408ac53f15e5e9cccb264af6516dd6a47a9f1d434f42f30726ae8615c767d6` and private aggregate file SHA-256 `054b846fd14efe1584c0738a7d1f447364fb0333b7f83dede073a38973acc905`.

## Interpretation

H3 is a null Dev30 result with 23 actual legal action changes. It does not support a win-rate gain, so H3 is retired and must not go to confirmation. The post-run aggregation bug is fixed and covered by focused tests; the Dev30 pool remains consumed and was not rerun.

G7 remains Champion. No formal Gate/Fresh outcome use, confirmation, promotion, deployment, or real-game evaluation occurred.
