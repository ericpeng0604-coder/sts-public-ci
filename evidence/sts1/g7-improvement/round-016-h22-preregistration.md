# Round016 H22: early HP preservation with unused energy

Issue #19; prerequisite PR #33, which permanently retains H21's null Train10
result and failed coverage gate. H21 is unchanged and not extended or rerun.

Distinct hypothesis: prevent earlier nonlethal visible attack damage rather than
attempt a last-turn rescue or limit the intervention to zero-cost cards. When
MCTS recommends End Turn, require positive known remaining energy and
`block < incoming visible attack damage < hp + block`. Select one affordable,
canonical legal, positive-cost **DEFEND_RED** (upgrade0/1), minimizing actual
cost then hand index then native ordinal. Retain actual costs and state; no
artificial cost normalization. Multiple later decisions use the same fixed rule.
No other card, draw, potion, residual or model-training rule is introduced.

Known player/enemy power guards are retained. Unknown/malformed resources,
intent, hand positions or legal actions abstain. Reject Pain/Normality, Gremlin
Nob, Time Eater, Heart, Ink Bottle, Unceasing Top and Letter Opener. Plain Defend
does not exhaust: Dead Branch and Charon's Ashes do not trigger. Velvet Choker's
cap is enforced by the complete native legal list; no action beyond the cap is
invented. No safety/retention/legal threshold is changed. Known No Block is
outside the allowed player powers. Visible intent remains a diagnostic cue,
not a complete combat survival proof.

Inventory then Sample(10 completed Round014 Train traces),36,857,494 bytes,
3,836 combat decisions: final frozen selector has9 eligible changes across7
traces. During implementation, reusing H21's exhaust/card-cap relic exclusions
yielded6 changes across4; those exclusions were corrected using plain Defend's
native effect before freezing and before any H22 episode. This was effect-based
implementation validation, not outcome-based parameter search. Shadow coverage
is not causal/effective coverage or win-rate evidence. No held-out outcomes used.

Generate entirely new Round016 pools after excluding all five Round015 pools,
including every unused reserved pool. Freeze code/selector/config and private
runtime/pool identities before outcomes. Use the unchanged corrected native
binding, frozen G7/ArmG and MCTS2000 in both arms. Preserve all raw evidence,
complete canonical legal actions, native ordinals, terminal and provenance data.

Fixed stages: Train10=20 paired episodes plus one first-candidate trace-off replay;
eligible Probe10=20 episodes; eligible Dev30=60 episodes. Maximum101 for this
single hypothesis, inside the140 ordinary round cap. Train/Probe each require
at least3 effective changed seeds with matched prefix/state and changed afterstate,
nonnegative net and all hard guards. Dev requires exactly30 complete pairs and
positive net; exact sign p is diagnostic. Existing retention and floor/HP rules
unchanged. No additional sampling or successful reruns. Every override is
independently revalidated using the frozen H22 selector and its own trace fields.

Before episodes:137 focused tests and2 subtests passed, including H19/H20/H21
regressions, H22 resource/effect/ordinal guards and trace/seed checks. C-only/
P-only/net/p and safety will be reported even for null/regression. Parent-gap/
residual metrics NOT_APPLICABLE to this deterministic rule; no MCTS tree scores
or provider token telemetry recorded. New paired gain, confirmation and real-game
acceptance NOT_RUN; improvement NOT_VERIFIED. G7 retained. No G7/PPO/STS2/formal-
data changes, paid APIs, promotion, deployment or merge.
