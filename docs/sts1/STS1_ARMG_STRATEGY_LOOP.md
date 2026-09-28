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

## Data-Efficiency v2

Before spending terminal branch-rollout compute, the loop first scouts the
current Champion trajectory and builds a candidate pool of non-combat states.

Candidate states are ranked using:

- Strategy uncertainty;
- later-floor importance;
- low-HP danger;
- choice complexity;
- state-bucket diversity;
- decision-kind coverage.

Only the configured top states receive expensive branch Teacher labels. This
raises useful labels per round without simply multiplying MCTS work.

## Elite Candidate mining v3

A Candidate that fails the Champion gate can still contain useful Strategy
knowledge. v3 therefore separates **promotion** from **teaching value**.

An Elite Candidate:

- never bypasses Dev-30 / Hidden-50 / Fresh-100 promotion rules;
- must have reached Fresh-100;
- must be complete and safety-clean;
- must use the frozen pure-MCTS combat policy;
- must show a positive Fresh win delta;
- must have more paired-better than paired-worse seeds;
- must satisfy the looser Elite confidence threshold;
- is stored only as a proposal model, not as Champion.

The historical Round-7 near-miss is the initial bootstrap Elite only when its
checkpoint SHA, parent Champion SHA, and recorded evaluation evidence all match
the frozen expected values.

### How Elite knowledge becomes training data

Evaluation seeds are never recycled into training.

Instead, Elite mining creates new training-only fresh seeds and follows the
Champion trajectory. At each identical non-combat state:

1. Champion and Elite score the exact same legal choices;
2. if they agree, no extra work is done;
3. if they disagree, the state can be sent to the ordinary branch Teacher;
4. the Elite choice is accepted only when the Teacher independently selects
   that same choice and its branch quality beats the Champion choice by the
   configured minimum margin;
5. only then is the row added to Replay with elevated priority.

This means an Elite Candidate can suggest a lesson, but cannot label its own
lesson.

Expensive Elite Teacher checks are bounded per mining seed. If the Elite
disagrees frequently but is usually wrong, CPU usage therefore remains bounded.

### Elite lifecycle

Durable state adds:

- `elite-pool.json`;
- `elite-candidates/*.pt`.

The pool is SHA-validated on restore. It retains only the configured top
near-miss Candidates.

When a new Champion is promoted, the old Elite pool is cleared because those
Candidates were measured against the previous Champion. Future Candidates must
earn Elite status against the new Champion.

During the v2→v3 handoff, the latest v2 Replay and seed histories are merged
once, without overwriting v3 Champion, Elite, or round counters. This prevents
the last v2 data from being lost while preserving train/evaluation seed
isolation.

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
- `elite-pool.json`
- `elite-candidates/*.pt`

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
- an automatically dispatched heavy Strategy round: hydrates the pinned
  simulator and ArmG upstream, then runs the real Strategy loop;
- a supervisor that continues, retries with load shedding, or pauses according
  to the durable control/state contract.

The heavy job uploads state and evidence as artifacts. It does not promote the
production Champion and does not merge anything.

## Later combat distillation

Combat traces should be kept as evidence, but combat-model training stays off
until Strategy + pure MCTS reaches a strong enough win rate.

At that later stage, the strong MCTS policy can become the Teacher for a fast
combat model without changing this Strategy training contract.
