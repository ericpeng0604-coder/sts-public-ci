# H19 Round010 Train10 — NOT_VERIFIED

## Outcome

The preregistered Round010 Train10 attempt stopped before the candidate arm. The parent-arm setup failed when the pinned Python binding did not expose `GameContext.potions`, which the required complete potion-inventory integrity guard reads. The candidate arm did not start. No complete matched pair or valid win-rate aggregate exists; this is an integrity failure, not a game loss or strategy result.

Round010 is withdrawn and must not be rerun. Its seed IDs remain reserved in the private exclusion inventory and are excluded from every later pool. No Gate/Fresh outcomes, traces, decisions, or labels were accessed.

## Cause and repair

The wrapper failed closed on the missing native property (`native_potion_property_missing`). The scoped repair adds a read-only binding for the exact five native potion-slot values, validates enum bounds, and rejects assignment. It does not alter gameplay rules, action selection, state transitions, RNG, or search. The original source, build, and installed module were preserved; the pinned gameplay source and existing hook patch remain unchanged.

The separate pinned-source build completed. A zero-episode smoke verified all five empty slots and that the property is read-only; it started no episode. The built module and private runtime fingerprints remain in local private storage and are not included here.

## Verification and next stage

The native-binding repair and Round011 loader passed focused tests together with the H7 potion trace, H16 evaluator, seed-ledger privacy, and simulator trace suites (71 tests and 2 subtests) under the pinned CPython 3.12 goal environment. This includes Dev positive-net selection without a p-value cutoff, hard-guard blocking, floor/HP diagnostics, and confirmation safeguards.

H19 stage-gate v2 remains in force: Train/Probe require net at least zero and effective overrides on at least 3 of 10 distinct seeds; Dev requires positive net and complete hard-guard-passing evidence, with exact p reported as a diagnostic. Floor/HP changes remain secondary diagnostics. Round011 pools are generated and disjoint from Round010 and the complete exclusion inventory; no game episode has started. Round011 still requires synchronized code, a private runtime identity lock, and a passing zero-episode preflight.

G7 remains Champion. No promotion, deployment, formal Gate/Fresh evaluation, or real-game evaluation occurred. Real-game status remains NOT_RUN.
