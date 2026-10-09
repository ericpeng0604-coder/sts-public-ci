# STS1 G7 potion-reserve iteration — 2026-10-09

## Decision

The frozen potion-reserve candidate did not improve the independent Dev30 result. Keep G7 as Champion; do not send this candidate to confirmation, promote it, or change the Champion artifact. The continuous Goal remains unfinished.

## Fixed identities and pools

- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Pinned simulator commit: `7476a81954020087da31d41d16fddf475746ec2d`; native binding SHA-256: `dea5e3b88097c7e7b7ffb74f03c227b7244a548b07fc5344c6b195d737c65512`.
- Candidate policy source SHA-256: `e11ffb0e679b5e06e15409e805abd810bcee5e82cc038cf4828cf9b8afa5fce7`.
- Combat used a fixed MCTS-2000 budget. ArmG noncombat decisions used the verified G7 artifact.
- H1 train pool: 10 seeds; canonical manifest SHA-256 `eff9960687e4757eb95900227a631a8e57980a3cf29dc809c06cccc6f59dfec3`.
- Probe pool: 10 seeds; canonical manifest SHA-256 `f97ce28a2063c37ea93c24397703133598344971249b35d8fa05f6e6dfc2ceb1`.
- Dev pool: 30 seeds; canonical manifest SHA-256 `b8eb7b9c625a5d9cc0c3ab8f6a53479546b171a5a7ff00a511489b5771ffaa78`.
- The three pools are pairwise disjoint and disjoint from the authorized exclusion inventory. Raw manifests, seed IDs, and episode traces remain in private local storage; this report exposes no IDs.

## Paired results

All rows compare the same seed per parent/candidate pair. One-sided exact sign tests use Candidate-only wins as the positive direction.

| Stage / policy | Parent wins | Candidate wins | C-only | P-only | Net | Discordant | Exact one-sided p | Effective potion overrides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 train — initial reserve rule | 1/10 | 0/10 | 0 | 1 | -1/10 | 1 | 1.0 | 375 |
| H1 train — no-empty-turn refinement | 1/10 | 1/10 | 0 | 0 | 0/10 | 0 | 1.0 | 112 |
| Probe10 — frozen refinement | 1/10 | 2/10 | 2 | 1 | +1/10 | 3 | 0.5 | 99 |
| Dev30 — same frozen refinement | 4/30 | 2/30 | 0 | 2 | -2/30 | 2 | 1.0 | 257 |

The Probe10 result was a small positive signal, not evidence of a reliable gain. Dev30 had two G7-only wins and no Candidate-only wins; this candidate fails the version-selection gate.

The first rule could replace a recommended potion with `END_TURN` when no non-potion action was available. In the only H1 parent-win pair, its first intervention at floor 4 did exactly that; G7 used the MCTS-recommended potion at the same state. The refinement permits a reserve override only when a legal card or selection action exists. It removed that specific regression in train, but did not improve Probe10+Dev30.

## Integrity and validation

- H1 parent, both train variants, Probe10, and Dev30 completed with complete terminal records and complete legal combat/noncombat choices. Selected native action indices resolved into the canonical legal-action lists.
- MCTS was 2000 on every traced combat decision. Candidate potion-slot annotations matched distinct `use_potion` slots in canonical native legal actions; no empty-turn reserve override occurred in the refined policy.
- Illegal actions, crashes, and timeouts were all zero for parent and candidate. Communication errors are not applicable to these local simulator runs.
- Focused tests: `tests/test_sts1_potion_reserve.py` and `tests/test_sts1_simulator_trace.py` — 10 passed.
- The pinned `GameContext` does not expose a potion inventory snapshot. That snapshot remains incomplete; action-level usable potion slots are complete and trace-verified.
- Raw train/Probe/Dev evidence was retained privately. No Actions artifacts were produced. Gate/Fresh outcome or trace data, real-game evaluation, confirmation, promotion, and deployment were not run.

## Next research direction

Using only the H1 train traces, low-health map choices (`HP < 50%` of max HP) selected an Elite route in four decisions across two defeat runs and none in the one victory run. Three of those four decisions had a legal non-Elite alternative; ArmG scored the selected Elite above its best non-Elite alternative in all three. This is weak, train-only evidence. The next bounded counterfactual will test a fixed low-health Elite-avoidance rule on the separate H2 train pool; Probe/Dev/confirmation remain untouched until its train and safety gates pass.
