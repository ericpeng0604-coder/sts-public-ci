# Round005 H6 Train10 Report

Date: 2026-10-09
Status: `COMPLETE_NULL_SIGNAL`; no Probe10 or Dev30 advance.

## Registered hypothesis and inputs

H6 tested the existing fail-closed low-HP Elite route rule. When G7 selected a legal Elite route below 50% HP and complete legal route choices, ArmG scores, and known room labels were available with a legal non-Elite alternative, the Candidate selected the highest-scored legal non-Elite route using the existing stable index tie-break. Other choices remained G7's.

Hypothesis generation used only the bounded, already-run Round005 H5 parent train traces. Of 330 map decisions, 330 had complete route/score records; 330 had complete legal-choice coverage and consistent Elite/non-Elite classification. Two decisions were below 50% HP with G7 selecting Elite despite a legal non-Elite alternative. In both, the Elite route was still ArmG score top-1. This supports a bounded test only; it is not causal proof.

The H6 run used only the preregistered Round005 `train_hypothesis_2` pool: 10 paired seeds, manifest SHA-256 `79ef213f04549431937649f30c0a2087184c3fc14040e8adf48cced9b4099200`. The raw seed list remains private. Preflight verified 22,826 excluded seed IDs, disjointness from the exclusion inventory and all five Round005 pools, and a `GENERATED_NOT_RUN` manifest before allocation.

| Input | Pinned identity |
|---|---|
| G7 checkpoint SHA-256 | `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0` |
| Simulator source SHA-256 | `fe735348978f2886fe7b5bc3a840c743c0de6fc3e37f124ceb6644367a1e0a49` |
| Native simulator binding SHA-256 | `dea5e3b88097c7e7b7ffb74f03c227b7244a548b07fc5344c6b195d737c65512` |
| ArmG source SHA-256 | `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b` |
| MCTS budget | 2,000 per combat action |
| Runtime | Python 3.12.14; Torch 2.11.0+cpu; native `slaythespire` import passed |

## Paired result

10 parent/Candidate pairs plus one trace-off invariance episode: 21 episodes total.

| G7 wins | H6 wins | Candidate-only | Parent-only | Net | Discordant | One-sided exact p |
|---:|---:|---:|---:|---:|---:|---:|
| 1/10 | 1/10 | 0 | 0 | 0 | 0 | 1.0 |

H6 had four low-HP Elite recommendations, three legal overrides, and three fail-closed decisions (two unknown route/room cases and one case without a known non-Elite route). The fail-closed decisions retained G7's action. The result is null and does not support a win-rate claim.

## Integrity and safety

- Runner status: `COMPLETE`; train usage ledger status: `COMPLETE`.
- Trace-on/off invariance: `PASS`.
- All 20 paired terminal traces were complete and legal; illegal actions, crashes, and timeouts were 0 for both arms.
- The runner validated each combat action against MCTS-2000 and checked complete noncombat choices, map route/index consistency, intervention accounting, and terminal guards.
- Local-simulator communication errors: `N/A_LOCAL_SIMULATOR`.
- Focused tests: 17 passed; `git diff --check`: passed.

Private evidence remains at `C:\Users\ericp\AppData\Local\Temp\sts1-g7-private-evaluations-20261009\round-005-h6-train`. Its 43 original run files total 88,799,596 bytes. Private artifact inventory SHA-256: `aff834b1a85a823fa458ff8f0a947f4855b8be96d7d1cf2ff4f7349c01097d1e`; the local `artifact-manifest.json` SHA-256 is `e7b7632b195a42e053befa5ae158cd5c333ac405481cfbd29c476ddf52237f5b`. Seed IDs and raw traces are not included in this report.

## Disposition and boundaries

H6 is retired after this null train result. Probe10 and Dev30 were not run, and this pool will not be reused. G7 remains Champion. No formal Gate/Fresh data was read or used; confirmation, real-game verification, promotion, and deployment are `NOT_RUN`/`NOT_VERIFIED`.

H6 evaluator and report changes are local on `codex/sts1-g7-h5-effective-rest-heal-20261009`. Remote code synchronization remains blocked by the earlier automatic-review refusal to upload the full evaluator source; no alternate source transport was attempted. The sanitized preregistration and result are reported on Issue #19. PR #24 remains Draft and unmerged.
