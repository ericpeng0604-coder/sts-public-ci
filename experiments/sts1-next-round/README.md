# STS1 next-round diagnostic and data-quality pilot

Combat remains pure MCTS 2000, noncombat remains G7 for source collection and
future continuations. No production model is changed by this workflow.

The existing 2,068 states all come from the same 112 runs already represented
in the consumed 240-state pilot. The remaining states are training/diagnostic
material, not independent calibration. This workflow captures 48 fresh training
runs and 32 fresh calibration runs, reserving 2,000 other seeds without executing
them. This is reservation capacity, not the preregistered evaluation sample size.
Only accessible public/local seed manifests are excluded; global private history
coverage is not claimed.

The actual public model and eight complete original/recovery label shards are
used for diagnosis. Outcomes, seeds and replay logs never enter model features.
The reports examine 8-world sparsity, stratified disagreements, clipped score
features and indistinguishable action inputs. Old validation is now consumed
diagnostic material and cannot validate a modified model or gate.

An eight-case pilot adds paired worlds to 16, then 32 if simultaneous corrected
intervals remain unresolved. Every option uses the same future roots; original
eight worlds are preserved. Wide bounds and zero observed variance are not proof
of certainty. The purpose is cost/noise measurement, not fresh win-rate proof.
The 64-case shortlist is saved for subsequent planned refinement.

New state selection uses only before-choice state and deterministic hashes,
never end-of-run victory labels. Training selection balances act/HP coverage;
calibration uses hash ordering, one state per eligible source first, and a cap of
12 states per source. All sources are independent across cohorts. Identical
consumed or cross-cohort states are excluded from calibration. Unreliable fields
remain unknown; unaudited relic counters are not promoted into features.

All eight shard summaries, all 80 source seeds, eight refinements and 400/200
unique selected states must pass checks. Coverage shortfalls stop the workflow;
it never substitutes consumed states for independent calibration. Training and
calibration labels are still pending after this stage, and no candidate is
adopted or evaluated. The final report provides measured cost for the next stage.

Outputs are public GitHub artifacts: synthetic state captures, lists, diagnostic
reports and paired simulations. No new human raw histories or opaque checkpoint
are uploaded to Git. G7 is read from the already-public original artifact.
