# H18 Round009 Train10 — NOT_VERIFIED

Date: 2026-10-10 (Asia/Taipei)

## Frozen evaluation setup

- Repository: `ericpeng0604-coder/sts-public-ci`.
- Registered remote code: `c385de7e6236f1ae2b582f5815e15db1f9430c05`.
- Local run-checkout commit: `4d0284daa9284b186c81f80f3f4daac8f647fba5`, with the same tree as the registered remote commit.
- Pinned gameplay commit: `7476a81954020087da31d41d16fddf475746ec2d`; MCTS budget: 2,000 per arm.
- Zero-episode preflight passed for the registered train pool, disjointness, and runtime identity pins.

## Run disposition

The preregistered Train10 batch stopped fail-closed during pair 2. Three episode executions completed before the fourth attempt ended incomplete at the 600-step bound. The incomplete attempt recorded `timeout=1` and `crash=1` for that same episode; `illegal=0` and `communication_error=0`. Canonical safety-counter completeness was false.

The batch is `NOT_VERIFIED`, not a valid defeat or completed paired evaluation. No aggregate wins, C-only/P-only, net, or p-value is reported. No Probe10 or Dev30 stage started. The partial evidence and allocation history remain preserved privately; the consumed pool will not be rerun or reused.

## Bounded mechanism observation

A bounded suffix of the failed episode's private trace showed 50 consecutive Act 3 Shop decisions while potion capacity was full (5/5). Each selected action was classified as a potion purchase; the potion inventory did not change, gold decreased by one per decision, and the screen remained the Shop. The trace contained 15 legal choices at each decision. Seed IDs and raw trace contents remain storage-only and are intentionally absent here.

This pattern is consistent with a repeated non-advancing Shop action, but the pinned native action implementation is not present in the checkout and the observation comes from one incomplete episode. The causal root and cross-seed recurrence are `NOT_VERIFIED`; this is not evidence of a policy win-rate effect. The current H18 native-Defend hypothesis did not establish effective override coverage before the integrity failure.

## Safety and next-step boundary

- Keep the existing step bound and all safety/completeness gates unchanged.
- Preserve this failed attempt permanently; any follow-up requires a diagnosed cause, a prospective protocol/scope update, and a fresh disjoint train pool.
- G7 remains Champion. No promotion or deployment occurred.
- Formal Gate/Fresh outcomes, traces, decisions, and labels were not read or used. Real-game evaluation is `NOT_RUN`.
- The earlier focused code verification remains 50 passing tests across four STS1 suites; this report-only update does not claim new code verification or any win-rate result.
