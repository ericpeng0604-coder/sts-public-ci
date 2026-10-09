# H3 round-003 train result

**Status: COMPLETE.** This is a 10-seed paired simulator train result, not a win-rate improvement claim.

## Frozen setup

- Candidate policy commit: `009e233bb38f71a2df3aea611db0b8ab9b664aa1`.
- Intervention: at HP/maxHP <= 0.5, choose the lowest-slot exact legal potion action; otherwise preserve G7's MCTS recommendation.
- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Pinned simulator commit: `7476a81954020087da31d41d16fddf475746ec2d`; native binding SHA-256: `dea5e3b88097c7e7b7ffb74f03c227b7244a548b07fc5344c6b195d737c65512`.
- MCTS budget: 2,000 for both paired arms. H3 train manifest SHA-256: `3f3d5f8f0e9fd762529bddc7577bd02ea29331e66d72cbdf9b36001d48669f52`.
- The five round-003 pools (70 seeds) were pairwise disjoint and excluded the protected inventory (22,756 IDs); seed IDs and raw traces remain private.

## Result and checks

- G7: 1/10 wins; Candidate: 1/10 wins.
- Candidate-only / G7-only / net: 0 / 0 / 0; discordant pairs: 0; exact one-sided sign p = 1.0.
- Candidate had four low-HP legal-potion opportunities and made four effective overrides across 3,300 combat decisions. The policy changed actions, but paired wins did not change.
- 21 episodes completed, including the trace-disabled invariance episode. All 20 trace-enabled terminal records were complete and legal; trace-on/off invariance passed. Across trace-enabled combat decisions, all 6,576 recorded MCTS budgets were 2,000.
- Illegal actions = 0; crashes = 0; timeouts = 0. Communication errors are not applicable to this local simulator.
- The private aggregate artifact SHA-256 is `df827e229c2ea148323da034373208dc6ca459cd006ca87f954943ff71476e13`; raw evidence stays in the private local evidence directory.

## Interpretation and next stage

H3 is a null train result with four actual policy changes. It does not establish a win-rate gain. Under the continuous Goal's stage rule, proceed to the untouched Probe10 only if this run's safety/completeness checks pass; they did. The Probe10 manifest was generated before seeing these results, has SHA-256 `d6ed4a75d4fca4b0df414572e1effe8585778b2be6cb7f88995b002e3b7ba477`, and remains unused at report time. For Probe/Dev, the evaluator must pass the selected pool via the simulator's held-out seed contract, not the training seed contract.

G7 remains Champion. No promotion, deployment, confirmation, formal Gate/Fresh outcome use, or real-game evaluation occurred.
