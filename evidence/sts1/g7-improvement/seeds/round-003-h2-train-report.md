# STS1 G7 H2 train-only result — 2026-10-09

## Decision

H2 did not change any action on its target cases: the two low-HP Elite recommendations had no known legal non-Elite route, so the fail-closed policy correctly kept the ArmG action. The paired train result was null (net 0); retire this H2 rule and do not run Probe10 or Dev30 from it. No candidate is being sent to confirmation. Keep G7 as Champion and continue with a distinct, trace-supported train hypothesis.

## Frozen identities and pool provenance

- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`.
- Pinned simulator commit: `7476a81954020087da31d41d16fddf475746ec2d`; native binding SHA-256: `dea5e3b88097c7e7b7ffb74f03c227b7244a548b07fc5344c6b195d737c65512`.
- H2 simulator/policy source file SHA-256: `dfb04a0c993c653c1ad23c7a1cb46f99dab53e826971e7a228631f10328fff91`; the local file was byte-identical to the PR branch file at pre-run head `b285afa7d7924a43f4fe04487637fb47ee2caf62`.
- Combat budget: MCTS-2000 for both arms; paired seeds and identical conditions.
- H2 train pool: 10 seeds; manifest SHA-256 `101477197215c2b53773848f1f11182e3d9e9aa5b1539bcbc8b836b7ec9ba8a0`; pool ID and raw manifest remain private.
- The round-003 ledger used `sha256-counter-rejection-v1`, inventory SHA-256 `8fa7cf83b46f57d37da6b9a6c60870d29423f13f66d99a095c74472ccf0a8f1b`, and five disjoint pools totaling 70 seeds. Preflight verified pairwise disjointness and exclusion from the 22,756-ID protected inventory. Only manifest hashes and counts are published.

## Paired train result

| G7 wins | Candidate wins | Candidate-only | G7-only | Net | Discordant | Exact one-sided p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/10 | 1/10 | 0 | 0 | 0/10 | 0 | 1.0 |

The runner completed 21 episodes: 10 paired parent/candidate comparisons (20 episodes) plus one trace-off invariance episode. Trace-on/off invariance passed. Independent validation covered 21 evidence files and 20 paired traces with zero validation failures; terminal and legal-action records were complete.

Across 407 candidate map decisions, 401 were unchanged and 6 safely failed closed. There were two low-HP Elite recommendations, zero overrides, and zero effective action changes. Fail-closed reasons were `no_known_non_elite_route` (2) and `map_room_or_target_unknown` (4). The latter cases also retained the parent action.

Illegal actions, crashes, and timeouts: 0. Communication errors: not applicable to this local simulator run.

## Pool history and boundaries

- Round-001 H2 pool: not run because its inventory fingerprint was stale; preserved and retired.
- Round-002 H2 pool: interrupted and **NOT_VERIFIED** after a validator event-type mismatch and a safe fail-closed no-op was misclassified. No paired outcome was computed or used; the pool was retired and not rerun.
- Round-003 H2 train pool: complete as reported above. The other round-003 train/probe/dev pools remain unused.
- Focused H2/trace tests on the exact runner revision: 33 passed, 2 subtests passed; AST parse passed.
- Gate/Fresh outcomes, traces, decisions, and labels were not read or used. The authorized read-only seed-ID exclusion inventory was used only for overlap prevention. Confirmation, real-game checks, promotion, and deployment were NOT_RUN.
- Raw manifests, seed IDs, and run traces remain in private local storage. No H2 Actions evaluation artifact was produced.

G7 remains Champion. This is a null train result, not evidence of a win-rate improvement or completion of the continuous Goal.
