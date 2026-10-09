# H19 Round011 Train10 / Probe10 report

## Registration and scope

- Trial: H19 Round011, prospective stage-gate protocol v2.
- Experiment code commit used for both stages: `f3286335b862bf380574188b6ead790bb8e38251`; tree `fdbe916200d41dad204b8efe490a69db12f603e5`.
- Pinned gameplay commit: `7476a81954020087da31d41d16fddf475746ec2d`; MCTS budget: 2000 for both arms.
- Public pool summary SHA-256: `a4b2609e35368bf9dd1bbaf3e763e962514faa60493ce2f32abe4efeced6d808`. It describes 70 fresh IDs in mutually disjoint train, probe, and dev pools. IDs, generation key, exclusion fingerprints, runtime identities, and raw traces remain private.
- Round010 is withdrawn after its native-potion-binding integrity failure. No Round010 pool was rerun.

## Train10

- 10 matched seeds; 20 paired episodes plus one trace-off invariance replay (21 executions total).
- Parent: 0 wins / 10 defeats. Candidate: 0 wins / 10 defeats. C-only 0, P-only 0, net 0, exact one-sided p=1.0 because there were no discordant pairs.
- Six effective legal candidate changes covered 3/10 distinct seeds. Train's nonnegative-net and 3/10 coverage exploration gates passed.
- All 21 terminal records were complete. Illegal, crash, timeout, and communication-error counters were all zero.
- Trace-off replay matched trace-on outcome, action signature, and override count. Terminal floor and final HP were equal across all 10 paired both-defeat outcomes.

## Probe10

- 10 fresh matched seeds; 20 paired episodes.
- Parent: 0 wins / 10 defeats. Candidate: 0 wins / 10 defeats. C-only 0, P-only 0, net 0, exact one-sided p=1.0 because there were no discordant pairs.
- Two effective legal candidate changes covered 2/10 distinct seeds, below the preregistered 3/10 coverage gate. The stage did not advance to Dev30.
- All 20 terminal records were complete. Illegal, crash, timeout, and communication-error counters were all zero. Terminal floor and final HP were equal in all 10 paired both-defeat outcomes.

## Bounded coverage diagnosis

Inventory was limited to the 10 Candidate traces from Train and the 10 Candidate traces from Probe. A deterministic local sample emitted only per-stage and per-file intervention counters; it did not emit seed IDs, raw states, legal-action payloads, outcomes, or trace rows.

- Train logged six `legal_defend_closes_visible_lethal_deficit` events. Each was backed by complete legal-action evidence and changed the action relative to the MCTS recommendation; these events covered three seeds.
- Probe logged two such events, also with complete legal-action evidence and action changes; they covered two seeds.
- The same traces logged 2,554 Train and 2,789 Probe `visible_intent_not_lethal` decisions, 317 / 345 decisions where G7 already chose Defend, and 16 / 22 cases where block did not close the visible deficit. These are event counts, not independent seed counts.
- Together with the native `DEFEND_RED` regression tests, this shows the corrected action path is live. The current exact-lethal trigger is sparse and its seed coverage did not reproduce at 3/10 on Probe. This is a coverage limitation, not a positive win-rate result.

## Decision and boundaries

- Do not run Dev30 from this Candidate: the pre-registered Probe coverage gate failed. Do not append games to this pool or weaken the gate.
- Preserve Train and Probe pools as consumed/seen. Do not call them untouched or reuse them as confirmation.
- No Candidate change was made between Train and Probe. No confirmation, Gate/Fresh outcomes, traces, decisions, or labels were used. G7 remains Champion; no promotion, deployment, or real-game run occurred. Real-game remains NOT_RUN.
- Next research step: inspect training-only evidence for a different, decision-time-visible hypothesis and validate any proposed factor with fresh disjoint pools. The bounded coverage sample does not establish that a broader Defend rule would improve full-game wins.
- Provider token totals and per-episode elapsed time are NOT_MEASURED.
