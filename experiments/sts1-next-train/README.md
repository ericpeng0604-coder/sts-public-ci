# Fresh-cohort noncombat model comparison

This is the next stage after successful diagnostic run 38342627848. It preserves
G7 and pure MCTS 2000 for source collection and rollout combat. Production remains
G7 regardless of research results.

Training uses 400 new states plus the consumed 240 pilot states. The old 64-case
refinement shortlist is extended to 32 shared worlds; already-refined cases are
reused. New training states receive 8 worlds per option. The 200 independently
captured calibration states receive 16 worlds per option. Source-run cohorts,
state identities, original observations, action sets and teacher choices are
checked; consumed pilot validation is never used as fresh calibration.

Two fixed 128x128 nonlinear models are trained for 40 epochs: the ArmG v4 control
and v5 with appended native card identity, selected-card upgrade information,
deck copies and smooth G7 score/gap features. This fixes the observed Violence /
Jack Of All Trades alias without changing the simulator or battle policy. Human
histories remain the same already-public source-balanced starting preference.

The v5 auxiliary head predicts variance of paired per-world win differences,
with a finite-sample variance floor; it does not produce epistemic confidence or
a statistical lower bound. Mean expected improvement drives action selection;
the intervention gate must pass independent calibration. No seeds, outcomes,
world results or unaudited relic counters enter model features.

The two models and six fixed thresholds form 12 preregistered comparisons. The
gate requires 50 interventions, 15 independent source runs and positive
source-cluster bootstrap lower bound with Bonferroni correction for all 12 tests.
The candidate and gate are frozen after calibration. No additional tuning or
sample expansion based on these calibration results is permitted.

Only a calibrated candidate triggers a fixed 1,200-pair fresh simulator test,
using the first 1,200 of the 2,000 seeds reserved before new collection. This
sample plan targets approximately 80% power for a 5 percentage-point gain when
paired discordance is at most 0.4 (one-sided alpha .05); it is not a power
guarantee. All pairs must complete without safety errors, candidate net wins
must be positive with exact one-sided sign p <= .05, and mean floor must not
decrease. There is no sequential retesting or automatic production replacement.

The public synthetic captures and source-run lists remain artifacts. Weights,
noise diagnostics, calibrated thresholds, freeze hashes and final reports are
saved as artifacts. Original G7 is consumed from the existing public artifact;
no private/raw human histories or opaque checkpoints are added to Git.

The stage1 eight-case timing gives a rough labeling cost range, not a guaranteed
runtime. Eight shards each use two labeling workers. Job time limits cap labeling
at 180 minutes and eligible evaluation at 90 minutes; scheduler delays remain
outside these execution limits. Low-token background monitoring polls every
five minutes and stops when the workflow completes.
