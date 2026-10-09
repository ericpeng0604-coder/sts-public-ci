# Retention and Guard Protocol Review

Date: 2026-10-10  
Status: protocol v2 was approved after the partial H18 attempt and applies prospectively from H19 Train10 only.

## Floor and HP are secondary diagnostics

The previous exploration gate vetoed a Candidate if even one paired terminal floor or final-HP result was lower than G7. That zero-regression rule applied to H12-H17 and remains part of those historical decisions; this protocol does not reclassify or reopen those runs.

Protocol v2 was registered before the first H19 episode. Round010 was its first eligible Train10 attempt, but failed closed before the candidate arm; its seed pool is withdrawn. Any continuation uses a fresh disjoint Round011 pool and the same pre-registered gate rules:

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

Protocol v1 remains the historical classification for H12-H18, including the incomplete H18 attempt. Protocol v2 is registered on Issue #19 before any H19 episode and cannot revive a prior candidate or reuse consumed pools.

## Prospective protocol registration

- Protocol ID: `sts1-g7-stage-gate-v2-2026-10-10`.
- Effective time: the Issue #19 body amendment timestamp; first eligible stage: the registered H19 Round010 Train10 after runtime and provenance preflight.
- H18 remains `NOT_VERIFIED`; its partial pool is consumed and withdrawn. H19 Round010 also remains `NOT_VERIFIED` because the native potion property was missing before the candidate arm. Neither result is reclassified as a strategy result under v2.
- H19 candidate implementation remains frozen to the committed simulator policy source. Candidate policy source Git blob: `f449f5dabf8429ac51554972bc0a162fca7564ab`; Round011 stage evaluator Git blob: `ee26fff922280f00a84765f8f339bbaa1326ec6b`. The evaluator writes this protocol ID into preflight and stage summaries. Confirmation execution remains NOT_VERIFIED in this evaluator and must be wired or independently verified before any confirmation run.
- Dev30 must have 30 complete paired outcomes and pass all hard guards. Candidate wins must exceed G7 wins. The exact one-sided p-value is retained as a diagnostic and does not gate Dev-to-confirmation selection.
- Train10 and Probe10 require net >= 0, at least 3 distinct seeds with a verified effective candidate override, and all hard guards. A change in total override count without distinct-seed coverage does not pass.
- Confirmation keeps two fresh, mutually disjoint 100-seed batches; each batch must have positive net, the combined net must be at least 10/200, and the combined exact one-sided sign test must pass the permanent trial-specific alpha. No floor/HP regression can waive these win-rate conditions.
- Floor and final HP are complete secondary diagnostics: lower/equal/higher counts, paired-delta summaries, and separate strata for both defeats, both victories, candidate-only wins, and parent-only wins. They do not independently block exploration or confirmation.

### Public-source privacy and H19 runner boundary

- The public evaluator no longer embeds checkpoint, native-binding, ArmG, exclusion-inventory, or H17/H18 allocation fingerprints. It reads the expected runtime identities from a required private identity-lock file outside the repository and records only the validated identities in private stage evidence.
- The runner accepts only H19 exploration stages and validates the Round011 inventory, ledger, manifests, pairwise disjointness, and private allocation record. H16-H18 and consumed Round010 pools cannot be selected through its CLI.
- This evaluator refactor does not change the candidate policy. Round011 remains NOT_STARTED until the synchronized remote code, private identity lock, new pool lineage, and runtime preflight all pass.

### Guards retained

- Simulator integrity guards: complete matched arms, known terminal outcomes, valid seed provenance/disjointness, complete legal-action and terminal evidence, stable policy/evaluator/simulator/checkpoint identities, and trace-on/off invariance where preregistered. These prevent invalid comparisons.
- Runtime safety guards: illegal-action, crash, timeout, and applicable communication-error counters must be present and zero. They prevent treating execution failures as game losses.
- Residual-adapter training retention: `_retention_ok` in `scripts/sts1/sts1_build_rescue_adapter_v32.py` still requires at least 256 parent high-confidence decisions, parent top-1 agreement >= 0.995, parent KL <= 0.002, and winner top-1 drop <= 0.01. The preservation check remains top-1 agreement 1.0 with preservation margin >= 0.0; minimum fully learned seeds remains 1. These protect trained residual behavior and are unchanged; they do not apply to the deterministic H19 rule candidate.

### Scope

Allowed paths for this protocol implementation:

```text
scripts/sts1/sts1_g7_h16_lethal_defend_eval.py
tests/test_sts1_h16_lethal_defend.py
evidence/sts1/g7-improvement/retention-guard-review.md
evidence/sts1/g7-improvement/experiment-ledger.md
```

Forbidden: STS2, PPO workflows, G7/Champion weights or identity, unrelated code/workflows, formal Gate/Fresh outcomes/traces/decisions/labels, pool reuse, safety-guard relaxation, promotion, deployment, merge, force-push, and paid services/APIs.
