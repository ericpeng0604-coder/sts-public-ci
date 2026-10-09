# Round009 Direction Review

Date: 2026-10-10  
Status: completed before H18 episodes; no new strategy result or Candidate-selection claim.

## Recent evidence

The consecutive strategy candidates H12, H13, and H14 had no positive Dev signal:

- H12 Train10: G7 and Candidate each won 0/10; net 0; exact one-sided p=1.0; 22 overrides; the current strict floor guard stopped the run before Probe.
- H13 Train10: each arm won 0/10; net 0; p=1.0. Probe10: each arm won 1/10; C-only/P-only/net 1/1/0; p=0.75. Probe also had one lower-floor pair and one lower-HP pair, so the current guard stopped it before Dev.
- H14 Train10: each arm won 2/10; net 0; p=1.0; two overrides; one lower-floor pair stopped progression.

H9 Dev30 had G7 and Candidate at 3/30 each, net 0, with no discordant pairs and exact p=1.0. H15 was a diagnostic round, not a strategy test: its bounded sample found four lethal-intent decision states across two defeats. In three states G7 chose a non-Defend action; in two, a legal Defend could close the visible projected deficit. That small sample motivated the single-factor lethal-intent Defend hypothesis.

H16 and H17 each completed paired Train10 episodes, but both used a helper that recognized DEFEND_R while the pinned simulator emits DEFEND_RED. Their apparent no_legal_defend coverage is invalid. Keep their 0/10 recorded outcomes and consumed pools intact, but classify both as NOT_VERIFIED_IMPLEMENTATION_COVERAGE. H17's later offline inspection of five possible overrides across three pairs is a hypothesis audit, not an executed counterfactual or win-rate result.

## Direction and falsifiable next step

The existing lethal-intent Defend rule has a decision-time mechanism grounded in H15's diagnostic sample and H17's offline audit. H18 therefore remains a corrective implementation replication of that same hypothesis, with no strategy change. The first evidence question is whether the repaired classifier produces a legal override that changes both the selected action and post-action state at the same paired decision prefix.

Run only the already allocated Train10 plus one trace-off replay after the evaluator, source, binding, checkpoint, seed ledger, and candidate identities pass preflight. Require at least 3 of 10 distinct seeds to show verified effective overrides, nonnegative paired net, complete legal/terminal/provenance evidence, zero safety counters, and trace-on/off action-signature equality. Floor and HP are recorded as secondary diagnostics, not a veto. Do not start Probe10 unless all registered Train→Probe gates pass. Do not start Dev30 unless Probe independently passes the same 3/10 coverage and hard gates.

If Train10 has fewer than three covered seeds, stop at ten pairs and diagnose the exact visibility, ID, legal-action, selected-action, and afterstate coverage counts. Do not add seeds. If coverage is present but the paired net is negative or retention fails, preserve the failure and choose a new hypothesis from bounded train-only evidence. Retrospective traces are not used to relabel H16/H17 or to claim H18 success.

## Direction checks

- Coverage and wiring: H16/H17 demonstrate why reason labels alone are insufficient. H18 now checks the native ID, legal action, matching pre-action signature, selected action, and post-action signature.
- Teacher and MCTS: Current reports do not isolate a Teacher-quality or MCTS-budget mechanism. Keep the checkpoint and MCTS-2000 fixed; do not change them in this replication.
- Combat versus noncombat: H15's only mechanism evidence points to a combat choice. Do not make unrelated noncombat edits unless a new bounded train audit demonstrates a repeated decision-time failure.
- Data and reproducibility: Current H18 pools have private ID-only exclusion provenance. Public reports contain aggregate counts only; raw IDs, traces, and private pool/inventory hashes remain private. No Gate/Fresh outcomes, traces, decisions, or labels are used. Unknown-purpose seeds remain excluded.
- Research interpretation: H18's 3/10 gate is an engineering coverage threshold, not a significance claim. Train/Probe net 0 permits exploration only when the gate is satisfied; it is not a win-rate improvement. A positive Dev net is a candidate-selection signal; Dev p is reported but is not a hard threshold under prospective protocol v2. Neither result is a formal improvement claim.

## Boundaries

This is simulator research only. Gate/Fresh are NOT_RUN; real-game validation is NOT_RUN/NOT_VERIFIED; G7 remains Champion; no promotion or deployment occurred. No new episode had started when this review was written.
