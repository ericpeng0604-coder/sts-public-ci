# H19 Round011 bounded loss-trace direction review

Date: 2026-10-10
Status: train-only diagnostic; not a new Candidate and not a win-rate result.

## Scope and pinned inputs

- Reused only the already-consumed H19 Round011 Train evidence. No new seeds or episodes were used; Probe, Dev, Gate, Fresh, and confirmation evidence were not read for this review.
- Experiment runner/candidate commit: `f3286335b862bf380574188b6ead790bb8e38251`; policy-source Git blob: `f449f5dabf8429ac51554972bc0a162fca7564ab`; evaluator Git blob: `ee26fff922280f00a84765f8f339bbaa1326ec6b`.
- Gameplay simulator remained pinned to `7476a81954020087da31d41d16fddf475746ec2d`, MCTS-2000, and the existing G7 checkpoint. G7 remains Champion.
- The review sampled the last at most three combat decisions in each of 10 parent-arm Train terminal-loss traces. It emitted aggregates only; raw rows, seed IDs, and manifests remain private.

## Bounded Train evidence

- The sample contains 28 decision rows because two terminal encounters had only two rows in the selected window. All 28 had complete legal-action evidence.
- On a decision-time cue only, visible enemy intent damage summed across living enemies exceeded the traced player HP plus block in 21/28 rows, across 9/10 traces. This omits simulator modifiers and is not an exact counterfactual or a claim that a different action would win.
- The parent-selected action matched the recorded MCTS recommendation in all 28 rows. A legal Defend was present in 5 rows across 5 traces; G7 selected Defend in 1 row.
- Corrected potion inventory: the 28 rows expose 140 slots, of which 124 are `EMPTY_POTION_SLOT` and 16 are occupied. No Block or Blood Potion appeared in this sample, so the previously tested H9 Block/Blood rescue rule has no matching opportunity here. A populated potion had at least one legal use action in 10 rows across 4 traces; six of those rows also had an End Turn recommendation, across 4 traces. The occupied potion labels were sparse and none recurred in three distinct traces.
- Diagnostic correction: an initial local aggregation treated `EMPTY_POTION_SLOT` as occupied. That result was invalidated before this report; all inventory figures above use the simulator's explicit empty-slot sentinel. No episode aggregate or historical result was changed.

## Direction review

- **Hypothesis recurrence:** the lethal-intent Defend mechanism is visible across Train traces, but H19 Round011 changed actions on 3/10 Train seeds and only 2/10 Probe seeds. Both stages had 0/10 wins per arm and net 0; Probe failed the preregistered 3/10 coverage gate, so Dev did not run. Do not widen the same rule from the terminal-loss sample alone.
- **H16/H17 coverage:** their `DEFEND_R` versus native `DEFEND_RED` classifier defect remains an implementation-coverage failure. Their recorded outcomes and consumed pools remain unchanged and are not evidence against a correctly executed intervention.
- **Earlier potion evidence:** H9 Dev30 had two effective potion overrides but C-only/P-only/net were 0/0/0 and exact one-sided p was 1.0. The current sample has none of H9's supported Block/Blood types. Other potion effects lack a verified decision-time projection in this sample; do not infer value from a full-looking inventory or from a legal action alone.
- **Encounter recurrence:** the existing Train loss audit found 10 terminal losses spread across enemy groups, with no group occurring more than twice. This does not support a single-enemy-specific rule.
- **Combat versus noncombat:** H19's change was combat-only; prior paired traces showed no changed noncombat decisions. The current bounded sample supplies no repeated noncombat failure mechanism.
- **Teacher and MCTS:** the available evidence does not isolate Teacher quality or an MCTS-2000 limitation. Keep both fixed; do not change model training or search budget based on these traces.
- **Data and guards:** H18's binding/timeout records and H16/H17's mapping defects remain `NOT_VERIFIED` for their intended coverage. H19 Round011 is complete and legal but produced no positive paired signal. Protocol v2 already removes the Dev p<0.05 cutoff and floor/HP per-pair veto prospectively; hard safety, identity, provenance, completeness, and formal confirmation requirements remain unchanged.

## Protocol verification and next bounded step

The prospective gate implementation and synthetic regressions are already present in PR #28 and Issue #19. Focused verification in the pinned local test environment: 57 passed, 1 skipped, and 2 subtests passed. The skipped test requires `STS1_POTION_BINDING_MODULE`; this run did not verify that native-extension-specific test. The H19 CLI still does not execute confirmation stages; confirmation execution remains `NOT_VERIFIED`, and no confirmation run is authorized by this report.

No evidence-backed next policy factor is established yet. Continue with a zero-episode, Train-only audit of native potion ID-to-effect and legal-action mapping, bounded to these already-consumed Train traces and trusted simulator/binding sources. Require an effect mechanism recurring on at least three distinct Train seeds and a verifiable counterfactual before registering a new Candidate. Any new Candidate needs fresh disjoint stage pools and a pre-run protocol record. Do not reuse H19 Probe/Dev as Train or relabel either pool.

No new pool was allocated, no episodes were added, and no change was made to G7, the Candidate, simulator, training guards, or confirmation thresholds. Runtime minutes are `NOT_MEASURED`.
