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

The pinned native source confirms the mechanism: `GameAction::getAllShopActions` and `isValidShopAction` expose an in-stock potion when it is affordable, without checking potion capacity; `Shop::buyPotion` calls `obtainPotion` and then charges gold. See the [pinned action validation and enumeration](https://github.com/gamerpuppy/sts_lightspeed/blob/7476a81954020087da31d41d16fddf475746ec2d/src/sim/search/GameAction.cpp#L217-L239) and [pinned purchase implementation](https://github.com/gamerpuppy/sts_lightspeed/blob/7476a81954020087da31d41d16fddf475746ec2d/src/game/Shop.cpp#L142-L154). Together with the trace, this verifies that full-capacity potion purchases can consume gold while leaving the potion inventory unchanged.

This verifies a wasted-purchase mechanism, not the full cause of the 600-step bound: only one incomplete episode was observed, and the remaining run path was not tested. Cross-seed recurrence and whether this mechanism alone caused the timeout remain `NOT_VERIFIED`. The current H18 native-Defend hypothesis did not establish effective override coverage before the integrity failure.

## Safety and next-step boundary

- Keep the existing step bound and all safety/completeness gates unchanged.
- Preserve this failed attempt permanently; any follow-up requires a diagnosed cause, a prospective protocol/scope update, and a fresh disjoint train pool.
- G7 remains Champion. No promotion or deployment occurred.
- Formal Gate/Fresh outcomes, traces, decisions, and labels were not read or used. Real-game evaluation is `NOT_RUN`.
- The earlier pre-run focused verification had 50 passing tests. A later shared-guard code repair and its 58-test verification are recorded below; neither is a win-rate result.

## Follow-up: shared full-potion Shop guard repair

This is an integrity repair, not a new strategy candidate or episode batch. The shared evaluator now skips a recommended potion purchase on `SHOP_ROOM` only when all five native potion slots are occupied, using the unique legal leave/SKIP action. It preserves the complete legal-choice list and logs the recommended and executed actions and reason in both evidence and decision traces. Incomplete inventory semantics or a missing/ambiguous legal skip fails closed. Parent and candidate use the same guard; it is excluded from candidate-specific override coverage.

- Local code commit: `588013b25acdaf393cc9fe5dc173ca9b624e31d6`.
- Synchronized GitHub code commit on PR #28: `6c0f37bbbc6569e0e115cfe7f814a6dab25b9a17`; local and remote tree are identical (`8f76549662c0f5c0039c8b353a62c29c2582bb31`), and both changed blobs were verified identical.
- Tests: 58 passed across the H16 evaluator, H7 potion trace, simulator trace, and H15 trace-audit suites using Python 3.11.9 with the existing Torch installation. Pytest reported a cache-write warning under the sandbox; it did not affect test execution.
- No new episodes or seed pools were used. The partial H18 pool remains consumed and excluded. G7 remains Champion. Formal Gate/Fresh outcomes were not read or used; real-game remains `NOT_RUN`.
- Before another episode, re-verify the pinned CPython 3.12 simulator module/binding identities and preregister a fresh disjoint train pool with its runtime provenance. Those follow-up checks remain `NOT_VERIFIED`.
