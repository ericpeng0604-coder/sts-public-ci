# Round008 H15 G7-only train trace direction review

## Status

`PASS_DIAGNOSTIC_ONLY`. This run used unchanged pinned G7 and did not train, tune, compare, or evaluate a Candidate. It is not a win-rate improvement result and did not advance to Probe10.

## Pinned inputs and pool provenance

- Round: `round-008-20261009`; diagnostic: `h15_g7_train_trace_audit`.
- Pool: `round-008-20261009-train_hypothesis_1` (10 train seeds); manifest SHA-256 `b24f900ee92864b32dfc090997c20e252f9fda4a952ea1f64ce714da313831ae`.
- Exclusion inventory: `g7-exclusions-plus-round-007-20261009`; SHA-256 `0d47f8956ad9f962ede5bac7bbd99931bbcfb1f7e9892dfbfb33259f921749be`; 81 source manifests and 23,036 unique excluded IDs. Seed IDs remain private.
- Other unused Round008 pools were checked before execution: Train10 ×2, Probe10, and Dev30. All five pools were legal-format, hash-valid, excluded-inventory-disjoint, and pairwise disjoint.
- G7 checkpoint SHA-256 `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Simulator Python source SHA-256 `981b33b0f39a0f139457185caa48dc84e93d179e4e0b67940cb2392890f1f72c`; native binding SHA-256 `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`.
- ArmG source SHA-256 `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`; vocabulary SHA-256 `832e199c359af8408ea430ffa3f9fcdc68f32533f7292bb102848d3fb558eb6a`.
- H15 runner source SHA-256 at invocation: `5d22bbb1a33c229fa9495ab393cf439b97025c880cdb12a14ec1343b7775323e`.
- Post-run safety follow-up pins one canonical private output and usage-ledger path; the checked-in harness rejects both alternate paths and reuse after those artifacts exist. This follow-up does not change simulation or policy behavior.
- Fixed combat budget: MCTS-2000. Tracing was enabled for 10 runs and disabled for one replay of the first seed.

The handoff summary's exclusion-inventory hash was stale. The first preflight stopped before any episode; the inventory file's actual SHA matched both the seed ledger and all pool manifests. The runner's pinned hash was corrected to that value, after which the full preflight passed before the 11 episodes began.

## Results and integrity

- G7: 2 victories / 8 defeats on 10 train seeds. The extra trace-off replay is an invariance check and is not an additional seed.
- Trace-on/off result, summary, and selected-action signatures matched.
- All 10 traces had complete terminal, encounter, combat, noncombat, route, and legal-action records: 3,191 combat decisions, 184 encounters, 771 noncombat decisions, and 366 route decisions.
- Illegal actions / crashes / timeouts: 0 / 0 / 0. Terminals were complete. Communication errors are `N/A_LOCAL_SIMULATOR`.
- Defeat terminal floors: 6 in floors 17–33 and 2 in floors 34–49; both victories reached floor 50+.
- All 771 noncombat selections matched G7's own recommendation. This verifies wiring/coverage for this sample; it does not prove the recommendations are optimal.

## Bounded train-only findings

Three hypotheses were checked against only this preregistered train trace scope:

1. **Noncombat recommendation wiring/coverage:** no mismatch was found in 771 complete decisions.
2. **Low-HP Elite routing:** among 13 map decisions below 50% HP, no Elite route was selected. This does not reproduce the earlier low-HP Elite concern.
3. **Combat lethal-intent defense:** four decision states across two defeat runs had visible incoming intent damage at least equal to current HP plus block and had a legal `Defend`. G7 selected a non-Defend action in three of those states; in two, the estimated deficit was within one legal Defend's base/upgraded block. This is a small-sample diagnostic signal, not a counterfactual win claim.

The third finding motivates a single-factor H16 train counterfactual on the separate Round008 `train_hypothesis_2` pool: when visible enemy intents predict lethal damage, G7 selects a non-Defend action, and a complete legal Defend action's known block closes the projected deficit, select the legal Defend with greatest block (ties: lowest hand index). The rule uses only decision-time state, current enemy intents, the complete legal-action list, and Defend upgrade status. No change will be made between paired runs. H16 must be preregistered before its code change or run.

## Artifacts and boundaries

- Public pool manifest summary: [`seeds/round-008-20261009/summary.json`](seeds/round-008-20261009/summary.json).
- Private traces, per-episode evidence, seed IDs, and the private usage ledger remain outside the repository. Private audit-summary SHA-256: `42a602ff47e136c69598b44215685f2088c4de5364a7e9e6915f0da6bc731ed6`.
- Gate/Fresh results and trajectories: `NOT_READ`; Gate/Fresh execution and use: `NOT_RUN`.
- Candidate, paired policy evaluation, Probe, Dev, confirmation, retention comparison, real-game, deployment, and promotion: `NOT_RUN`.
- G7 remains Champion.
