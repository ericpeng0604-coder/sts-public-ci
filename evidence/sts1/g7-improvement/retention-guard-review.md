# Retention and Guard Protocol Review

Date: 2026-10-10  
Status: protocol v2 approved by the user for H18 Train10, before any H18 episode.

## Floor and HP are secondary diagnostics

The previous exploration gate vetoed a Candidate if even one paired terminal floor or final-HP result was lower than G7. That zero-regression rule applied to H12-H17 and remains part of those historical decisions; this protocol does not reclassify or reopen those runs.

Starting with the still-unseen H18 Train10 stage:

- Train→Probe requires complete/legal paired evidence, zero hard-safety counters, verified effective overrides on at least 3/10 distinct seeds, and paired net ≥ 0.
- Probe→Dev uses the same hard-safety and 3/10 coverage requirements with paired net ≥ 0.
- Dev→confirmation requires all 30 complete pairs and hard guards, and net > 0 (C-only > P-only). The exact p-value is reported but is not a selection threshold.
- Both confirmation batches retain the original two fresh, disjoint 100-seed rule, positive net in each batch, combined net ≥ 10/200, and the preregistered one-sided exact sign-test alpha for historical k.
- Floor and final HP never independently block these transitions. They remain required complete fields and are reported as lower/equal/higher counts, paired-delta summaries, and separate strata for both defeats, both victories, C-only wins, and P-only wins. These diagnostics do not replace the win-rate KPI.

No per-pair tolerance was added. Regressions remain visible in every report.

## Hard safety and integrity guards retained

These guards remain blocking and fail closed:

- Every paired arm must have a known complete victory/defeat terminal record and valid paired provenance.
- Illegal action, crash, timeout, and applicable communication-error counters must all be present and exactly zero.
- Missing arms, duplicate or overlapping seeds, unknown outcomes, candidate/simulator/checkpoint identity mismatch, missing canonical legal-action evidence, incomplete traces, or failed trace-on/off invariance stop the stage as NOT_VERIFIED.
- G7 remains the Champion; candidate changes require a new frozen identity and new confirmation batches.

## Model-training retention guards are unchanged

The residual-adapter trainer's _retention_ok in scripts/sts1/sts1_build_rescue_adapter_v32.py remains unchanged: at least 256 parent high-confidence decisions, parent top-1 agreement ≥ 0.995, parent KL ≤ 0.002, and winner top-1 drop ≤ 0.01. Its preservation check remains top-1 agreement of 1.0 with minimum preservation margin ≥ 0.0; training also retains the minimum fully learned-seed requirement of 1.

Those guards protect a learned residual adapter and Teacher/preservation behavior. H18 is a deterministic rule intervention with frozen G7/Teacher weights and no model training, so those training-only guards do not apply to H18. They are not relaxed or removed from any training workflow.

## Applicability and history

Protocol v2 is to be registered on Issue #19 with its code hashes before H18 Train10 begins. The previous strict floor/HP and Dev p<0.05 rules remain recorded as historical protocol for already-seen stages. This change is prospective only and cannot revive H13/H14/H16/H17 or reuse their consumed pools.
