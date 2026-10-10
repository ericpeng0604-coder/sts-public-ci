# H20 frozen selector and Train shadow result

Prerequisite base: PR #29, `2e9adfd13cdc327e676e31bf58dca2e7ea820158`.
H20 was registered in Issue #19 before its implementation:
https://github.com/ericpeng0604-coder/sts-public-ci/issues/19#issuecomment-6094949466

Read-only scope: ten existing Round011 parent Train traces, 29,650,339 bytes,
2,936 combat decisions. All decisions retain complete canonical legal actions
and the exact MCTS-2000 recommendation equals the parent selected action. All
ten source terminals passed completeness and zero error-counter checks.
Header stage/arm/round/runtime identities and distinct source identities are
validated inside the audit; private hashes and IDs are not emitted.

Frozen final selector source SHA-256:
`2a25bf289329f0da67dceb182d212e912bef2477e45578e6019176ebee193b19`.
Final shadow result: 6 exact legal action changes across 4/10 distinct traces.
Reason counts: End-Turn not recommended 2,408; no eligible Inflame 485;
unknown player power 26; excluded enemy 8; excluded hand curse 3; selected 6.

The first implementation treated identical non-targeted action projections as
ambiguous (3 selected states / 2 traces). A bounded schema comparison found the
extra entries were byte-equivalent public action objects, not distinct projected
decisions. Native INFLAME queues only +2/+3 Strength and does not use its target.
The final selector preserves the first exact native-list ordinal alongside the
unchanged canonical action. Differing projections still fail closed. No new
episodes or fresh outcomes were inspected before this representation correction.

These are shadow proposals on fixed recorded states, not an executed trajectory,
matched counterfactual, replayed simulator, coverage gate for a fresh stage,
or win-rate improvement. The next unresolved prerequisite is an explicitly
instrumented runtime/evaluator preserving native-list ordinal, recommendation,
executed action, reason, and raw evidence with matched-prefix and identity guards.
Then allocate fresh, excluded-inventory-disjoint Round013 Train10 and run one
MCTS-2000 paired counterfactual plus trace invariance. No consumed pool may be
reused. Do not treat the shadow 4/10 as the future Train coverage denominator.

Runtime Candidate activation, new pools/episodes, model training, Probe/Dev,
confirmation and real-game: NOT_RUN. Win-rate gain: NOT_VERIFIED. G7 remains
Champion; no model, gameplay, guard or existing experiment outcome changed.
No policy provider call was added; provider token usage NOT_MEASURED.

Changed paths are the H20 selector, bounded shadow CLI, focused tests and these
two reports, all inside Issue #19's allowed paths. The existing shared evaluator
and simulator are unchanged; H20 is not enabled in them.

Validation: 69 focused tests passed (H20 selector/audit plus existing H16/H19
guard suite), no skips. Audit tests reject Probe metadata and repeated source
identities, prove aggregate-only output and immutable input evidence, and repeat
deterministically. Final Train shadow counters matched the frozen discovery
mechanism; selector source hash above includes the exact native-ordinal fix.
