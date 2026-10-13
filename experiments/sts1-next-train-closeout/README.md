# Fresh-cohort noncombat experiment closeout

Recovery run: https://github.com/ericpeng0604-coder/sts-public-ci/actions/runs/38353967978

**Completed successfully; HOLD G7. No calibrated advantage and no production replacement.**

- Eight label shards complete, all preregistered case/world budgets satisfied.
  Completed labels were reused and three interrupted shards resumed without
  dropping samples or resampling failed worlds.
- Two frozen nonlinear architectures trained for 40 epochs on 640 cases:
  240 consumed pilot cases and 400 new own-state cases.
- Independent calibration used 200 cases from 32 new source runs, 16 paired
  future worlds per option. This calibration cohort is now consumed.
- ArmG v4 control at its 0.02 predicted-gain gate intervened on 76 decisions
  across 29 runs. Cluster-averaged paired rollout win advantage was -0.0066810
  (-0.668 percentage points) and the 12-comparison-corrected bootstrap lower
  estimate was -0.0316451. These are conditional future-world estimates on
  intervened card states, NOT a measured full-game win-rate change.
- The v4 0.05 gate had only 14 interventions across 14 runs, below the
  preregistered minimum of 50 interventions and 15 runs. Higher v4 gates and
  all six v5 card-identity gates had zero qualifying interventions.
- No model/gate passed. Per the preregistered protocol, the 1200 reserved
  full-game evaluation pairs were not executed. No new full-game win rate
  is available. No more tuning may treat these 200 cases as fresh validation.

Models artifact: 11792949028; final report artifact: 11793058775.
Exact source/freeze hashes, calibration rows and weight hashes are in results.json.
Recovery took approximately 32 minutes, including label recovery and training.

## Simulator defect and next data version

Diagnosis exposed Courier shop restock pricing defects. That engine was retained
through the frozen cohort so previous state references and completed labels stayed
consistent. The game-step cap was mechanically increased from 600 to 10000 for
both labels and evaluation before calibration, after tracing a shop sequence with
strictly decreasing gold. Combat remains MCTS 2000. Unknown or failed continuations
were never coerced into defeats.

The separate `experiments/sts1-pricing-fix/courier-pricing.patch` corrects potion/
relic price calculation, the Membership Card condition for card restocking and
assignment of restocked relic prices. Native regression checks reproduced 576
failures out of 1024 checks in the original implementation and zero in the corrected
implementation. This patch is prepared and verified, but NOT integrated into the
active baseline or used to collect this cohort.

The next experiment should integrate the corrected engine under a new provenance
version, collect new pre-decision states and labels, and use a new calibration
cohort. Do not combine incompatible engine trajectories. Existing reserved,
unexecuted evaluation seeds remain unconsumed. All results here are simulator
research evidence and cannot support real-client performance claims.
