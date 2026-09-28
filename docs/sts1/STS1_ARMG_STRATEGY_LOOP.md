# STS1 ArmG Strategy-Only Self-Improvement Loop

## Goal

Improve long-horizon STS1 A0 win rate by training **only non-combat
Strategy decisions**. Combat is frozen to one exact pure-MCTS budget.

This is deliberately not a combat-model training loop.

## Frozen combat contract

For a Strategy lineage:

- `combat_mcts_sims` is frozen in `strategy-state.json`;
- Champion and Candidate use the exact same pure-MCTS budget;
- `student_action_count == 0`;
- `hybrid_student_vote_count == 0`;
- `hybrid_student_tiebreak_count == 0`;
- changing the MCTS budget requires a new state directory.

The gate fails closed if any of these conditions drift.

## What Strategy learns

The loop trains the ArmG scorer on every multi-choice non-combat decision
exposed by `ArmGNoncombatPolicy.choices()`.

This includes the upstream ArmG kinds available in the pinned runtime, such as:

- map path;
- card reward / skip;
- shop decisions;
- rest-site decisions;
- event decisions;
- relic / boss-relic decisions when exposed through ArmG;
- any later non-combat ArmG decision kind that uses the same descriptor/scorer
  contract.

The loop records `examples_by_kind` every round. A missing kind is therefore
visible evidence, not silently claimed coverage.

## Teacher

The Teacher is branch rollout, not a combat Student.

At one non-combat state:

1. clone the exact `GameContext`;
2. try every legal ArmG choice when the choice count is within the configured
   exact-branch limit;
3. after that one choice, finish the run with current Strategy + frozen pure
   MCTS combat;
4. rank branches by victory, then floor, then remaining HP;
5. convert branch values into a soft target distribution;
6. execute the Teacher-best choice on the untouched original state.

Clone parity is checked before branching, and the original state is checked
again after all branches. If parity fails, data collection stops.

This is one-step policy improvement with a long-horizon terminal evaluation.

## Replay and training

Every example stores:

- decision kind;
- public ArmG observation vector;
- exact candidate descriptors;
- current Strategy choice;
- Teacher-best choice;
- every branch terminal result;
- soft target probabilities;
- Teacher margin;
- priority.

Replay is deduplicated by `kind + observation + descriptors`.

Priority is higher for:

- Strategy/Teacher disagreement;
- clear branch-quality margins.

Replay retention is round-robin across decision kinds so common map/card
examples cannot silently remove rare shop/rest/event examples.

Training starts from the exact current Strategy checkpoint and uses:

- soft-target cross entropy;
- per-kind balancing;
- example priority;
- L2 anchor to the current Strategy parameters;
- gradient clipping.

No combat parameters exist in this training path.

## Seed isolation and promotion

All 50 upstream held-out seeds supplied by `--dev-seed-file` are permanently
excluded from training. The first 30 are the reusable fast development gate.

A Candidate must pass:

1. **Dev-30** — paired current vs Candidate on the same 30 fixed development
   seeds.
2. **Hidden-50** — paired current vs Candidate on 50 newly generated seeds that
   were never used for training or earlier evaluation.
3. **Fresh-100** — another 100 never-before-used paired seeds. This gate also
   requires paired superiority under a one-sided sign test.

Hidden-50 and Fresh-100 seed sets are consumed after use and stored in loop
state, so a later Candidate cannot reuse those exam papers.

Only when all three gates pass is `current-strategy.pt` replaced.

The production / real-game Champion is never changed by this loop.

## Fail-closed properties

The loop stops instead of guessing when:

- the simulator cannot clone `GameContext`;
- clone choice identity differs;
- a branch mutates the original state;
- a branch cannot finish within step bounds;
- Strategy checkpoint identity drifts;
- the combat MCTS budget changes inside a lineage;
- evaluation uses hybrid/Student combat;
- evaluation has a safety failure;
- paired seed sets differ.

## State and evidence

Persistent state:

- `strategy-state.json`
- `current-strategy.pt`
- `strategy-replay.jsonl`

Per-round evidence:

- `strategy-branch-dataset.jsonl`
- Candidate checkpoint;
- Dev-30 current/Candidate aggregates;
- Hidden-50 current/Candidate aggregates when reached;
- Fresh-100 current/Candidate aggregates when reached;
- `round-report.json`

Loop-level evidence:

- `loop-summary.json`

Every checkpoint is identified by SHA-256.

## GitHub workflow

`.github/workflows/sts1-armg-strategy-loop.yml` has two layers:

- automatic contract tests: cheap and safe;
- manually dispatched heavy Strategy round: hydrates the pinned simulator and
  ArmG upstream, then runs the real Strategy loop.

The heavy job uploads state and evidence as artifacts. It does not promote the
production Champion and does not merge anything.

## Later combat distillation

Combat traces should be kept as evidence, but combat-model training stays off
until Strategy + pure MCTS reaches a strong enough win rate.

At that later stage, the strong MCTS policy can become the Teacher for a fast
combat model without changing this Strategy training contract.
