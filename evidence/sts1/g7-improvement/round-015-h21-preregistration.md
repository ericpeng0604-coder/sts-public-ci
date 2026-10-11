# Round015 H21: zero-cost defense before End Turn

Issue #19; prerequisite PR #32 at
`d19ffa164a5b6076c9879ad1359bc3a49f052adb`. One fixed hypothesis class:
continue a recommended End Turn with a canonical legal zero-cost DEFEND_RED,
GOOD_INSTINCTS or INTIMIDATE when visible incoming attack damage exceeds block.
Choose the lowest hand index, retaining its exact native ordinal. Intimidate
requires an attacking enemy without existing positive Weak. No paid card,
draw/card-generation rule, Panic Button, potion rule or model training is added.

Excluded: Gremlin Nob, Time Eater, Heart; unknown player/enemy powers; Pain or
Normality in hand; Velvet Choker, Dead Branch, Ink Bottle, Unceasing Top and
Charon's Ashes. Missing or ambiguous resources, hand positions, intent or legal
actions abstain. Awakened One is allowed because these are skills, not powers.
Primary intervention exclusivity and MCTS2000/no training collection remain
enforced. G7/ArmG weights and all existing safety/retention gates remain unchanged.

Inventory then Sample of completed Round014 Train10 only: 3,836 combat decisions,
5 eligible changes across 3 traces (Good Instincts1, Intimidate4). This is a frozen
selector shadow result, not causal/effective coverage or win-rate evidence. The
three leading causes and unknowns remain in the Round014 report; this candidate
tests one concrete tactical choice without using held-out outcomes.

Before episodes, freeze code, selector/config and private runtime/pool hashes.
Extend all Round014 exclusions with all five reserved Round014 pools. Generate
new disjoint Round015 Train10/10/10, Probe10, Dev30 pools. Only Train1 is used for
H21. Every other pool is reserved; no old pool is reopened. G7 and H21 receive
identical fresh seeds, pinned simulator, MCTS2000 and starting conditions.

Train10:20 paired episodes plus the prospectively registered first candidate
trace-off replay. Replay must preserve terminal summary, full action signature,
action count and H21 override count. Effective changed-seed coverage requires
matched prefix/state and changed afterstate, at least3, with nonnegative net and
all hard guards. Eligible Probe10:20 episodes, same coverage/net/guards. Eligible
Dev30:60 episodes, exactly30 complete pairs, positive net; sign p is reported as
diagnostic. At most101 episodes for this single hypothesis; ordinary round cap140
unchanged. No extra sampling, reruns of completed successes or significance chasing.

Report wins, candidate-only, parent-only, net, exact one-sided sign p, effective
coverage, floor/HP diagnostics, action reasons and zero illegal/crash/timeout/
communication counts. Preserve all null/regression samples and full raw evidence
privately. Revalidate every override using the frozen selector; retain native
aliases and legal ordinal execution. Intervention trace/result fields are separate
from H19/H20 even though the common End Turn execution path is reused.

Validation before episodes:120 focused tests and2 subtests passed. Model-training
retention unchanged; parent_gap/residual deltas NOT_APPLICABLE to this deterministic
combat rule (no residual training or MCTS tree scores recorded). New paired result,
confirmation and real-game acceptance NOT_RUN; improvement NOT_VERIFIED. No G7/
PPO/STS2/formal-data changes, paid APIs, promotion, deployment or merge.
