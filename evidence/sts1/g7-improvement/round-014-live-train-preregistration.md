# Round014: fresh G7 live-resource Train diagnosis

Issue #19; prerequisite Draft PR #31 at
`5b656eca65c53031a062688278954a9f5b3740f2`. This round collects unchanged G7
diagnostics using the corrected native BattleContext potion getter. It selects
no candidate and makes no win-rate claim.

Before any episode, freeze collector/source hashes, G7/ArmG identities, corrected
native binding, MCTS2000, generated manifests and exclusion inventory privately.
Extend the Round013 exclusions with all five Round013 pools, including unused
reserved seeds. Allocate new disjoint Round014 Train10/10/10, Probe10 and Dev30
pools. Only the first Train10 pool is consumed for baseline diagnosis. All other
pools stay reserved and are excluded from later rounds; none can be reopened.

Run ten baseline episodes and exactly one trace-off replay of the first seed:
11 episodes, inside the ordinary 140-episode round cap. The replay must match
terminal summary and the complete selected-action signature before subsequent
seeds run. Preserve all raw evidence, selected actions, canonical legal actions,
terminal records and provenance outside Git. Every completed game must have
zero illegal/crash/timeout/communication counters and valid live-resource traces.
Missing provenance, incomplete actions, malformed resources or replay mismatch
stops the round as NOT_VERIFIED. No automatic rerun or extra sampling.

Read metadata/Inventory first, then a bounded Sample of these ten Train traces
only. Diagnose available legal resources, premature end-turn opportunities and
encounter/deck/resource context. Current intent damage is a diagnostic signal,
not proof of avoidable defeat. Establish cross-seed and causal support before
freezing any candidate in a subsequent fresh paired round. No Probe/Dev/formal
outcomes or trajectories may form hypotheses. Existing H20 outcomes are retained.

No G7/PPO/STS2 changes, paid APIs, promotion, deploy or merges. New candidate,
paired improvement, confirmation and real-game acceptance remain NOT_RUN.
Provider usage: NOT_MEASURED. The collection result will be published separately.
