# Mechanical recovery of the frozen cohort

Original run 38344535817 completed five label shards and failed three. Two
historical shop decisions chose different argmax indices. Replay may restore the
recorded legal index only for identical ArmG descriptors and finite scores within
1e-5, as demonstrated in diagnosis run 38350942615. Selected card states, actions,
observations, descriptors and G7 teacher indices remain exact-match checks.

The third error was a genuinely progressing shop sequence, not a stuck screen:
the final diagnostic steps spent one gold each (83,82,81,80,79). The pinned native
simulator incorrectly prices Courier replacement potions at one gold. Its
`Shop::getNewPrice` overwrites the input base price with the rounded random factor
and discards discount calculations. This is an existing simulator fidelity defect.
It is deliberately documented rather than changing the engine mid-cohort, which
would invalidate captured states and completed labels. Fixing that engine defect
and collecting a separate corrected cohort is required before real-game claims.

The mechanical game-step bound is increased from 600 to 10000 for labels and both
evaluation arms before training/calibration is executed. Policy, MCTS 2000, native
engine, RNG, world seed order, cohort membership, architectures, optimization,
calibration gates and the reserved 1200 paired evaluation seeds are unchanged.
Previously terminal rollouts have exactly the same transitions and outcomes.
Failed worlds are neither resampled nor encoded as defeats. A remaining bounded
or invalid continuation fails the run. Conditional evaluation remains a simulator
research comparison; production adoption and real-game performance claims are
explicitly disabled.

Recovery validates and reuses completed case/world prefixes, resumes missing
cases only, checkpoints after every complete paired world, filters each artifact
to its assigned case IDs even on failure, and regenerates all eight shard summaries.
Training still requires the complete preregistered dataset and all eight summaries.
