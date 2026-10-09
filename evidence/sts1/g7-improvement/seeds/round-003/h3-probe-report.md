# H3 round-003 Probe10 result

**Status: COMPLETE.** The frozen H3 policy produced no paired win difference on this small probe sample.

## Frozen setup

- Candidate policy commit: `009e233bb38f71a2df3aea611db0b8ab9b664aa1`.
- Held-out evaluator commit: `48f60b04868714cbd026b90b12e7d68ef0a35c7e`; runner SHA-256: `46b6272d76f9c4a7cbd88a5608347993a7498ebd60068724f5c78040eabdda02`.
- Probe10 manifest SHA-256: `d6ed4a75d4fca4b0df414572e1effe8585778b2be6cb7f88995b002e3b7ba477`.
- Seeds were resolved under the `heldout_internal` contract, not the training contract. G7 checkpoint, pinned simulator/native binding, ArmG source, and MCTS-2000 matched the preflight identities.

## Result and checks

- G7: 0/10 wins; Candidate: 0/10 wins.
- Candidate-only / G7-only / net: 0 / 0 / 0; discordant pairs: 0; exact one-sided sign p = 1.0.
- Six low-HP legal-potion opportunities produced six effective overrides. No paired win changed.
- 21 episodes completed, including the trace-disabled invariance episode. All 20 trace-enabled terminal records were complete and legal; trace-on/off invariance passed. All 5,803 recorded combat MCTS budgets were 2,000.
- Illegal actions = 0; crashes = 0; timeouts = 0. Communication errors are not applicable to this local simulator.
- Private aggregate artifact SHA-256: `a2c9d4786b3bc43725ac00d17ebb6b52626c87c484186e6e15468540cac9bc72`; raw traces remain in the private local evidence directory.

## Interpretation and next stage

This is a null Probe10 result, not a win-rate improvement claim. It is safe and nonnegative with six effective policy changes, so the preregistered stage rule permits a fresh Dev30 comparison. The Dev30 pool is still unused and has manifest SHA-256 `1c5902006c427aee215e7b79eca65e58cf489ef125b18113a4b99c72e007d856`; it will be run only after its preflight and preregistration are recorded, with the same frozen H3 policy and held-out seed contract.

G7 remains Champion. No confirmation, formal Gate/Fresh result use, promotion, deployment, or real-game evaluation occurred.
