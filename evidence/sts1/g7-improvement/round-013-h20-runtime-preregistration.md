# Round013 H20 runtime evaluation

Issue #19; extends the preregistered frozen selector in PR #30. G7 remains parent.

H20 has independent activation, reason, detail and override fields. It executes
the selector's exact ordinal from the complete native action list. The evaluator
revalidates each override against the frozen selector and checks matched parent
prefixes and changed afterstates before counting effective seed coverage.
H19 retains its own fields and Round011 identity.

Use a new Round013 exclusion inventory extending Round011 with all five reserved
Round011 pools; do not inspect held-out outcomes. Generate new disjoint pools and
keep their IDs, hashes, identity lock, traces and usage ledger outside Git.
Freeze runtime, evaluator and selector identities before the first episode.

Train: 10 pairs plus one candidate trace-off invariance replay. Probe: 10 new
pairs only after Train passes. Dev: 30 new pairs only after Probe passes. Existing
coverage, retention and safety gates remain unchanged. The user's request for
more games is handled by the existing two fresh 100-pair confirmation batches
after selection; a small exploratory result alone is not a win-rate gain.
Confirmation execution remains NOT_VERIFIED until its runtime is implemented
and checked. No formal seed outcomes enter this work.

Pre-run validation: 72 selector/evaluator tests and 25 simulator/trace/seed tests
passed, including 2 subtests. Paired outcome: NOT_RUN at this preregistration.
Provider usage: NOT_MEASURED. Real-game acceptance: NOT_RUN.
