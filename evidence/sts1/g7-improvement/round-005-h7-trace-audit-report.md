# H7 potion-inventory trace audit — NOT_VERIFIED

- Stage: bounded diagnostic only; episodes completed: `1`.
- Failure class: `EvaluationIntegrityError`. Individual seed IDs and raw evidence remain private.
- No win-rate conclusion, candidate, Probe10, Dev30, or confirmation result is claimed.
- G7 remains Champion.

## H7 recovery attempt NOT_VERIFIED

- Episodes completed across attempts: `1`. Failure class: `EvaluationIntegrityError`.
- All prior files were retained; no completed seed was rerun.

## Recovery result after validator repair

- Status: `COMPLETE_DIAGNOSTIC_ONLY_AFTER_VALIDATOR_REPAIR`.
- Date: `2026-10-09T08:41:53.316608+00:00`.
- Scope: unchanged G7 on 10 registered Round005 train-only seeds, plus one trace-off replay (11 episodes).
- Recovery: the first attempt stopped after one complete episode because the validator incorrectly required deck/relics on a combat-only snapshot. The validator was corrected and that retained episode was revalidated; its seed was not rerun.
- Pool manifest SHA-256: `9e088016ba055d58a45df132ed20423fcea8becc3677aa04613794dae64e64cc`; seed IDs and raw traces remain private.
- G7 checkpoint SHA-256: `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`; simulator source SHA-256: `92b641f3d3568d5fec0cc09744976b151d96bc62a368a0a942ee1f7ef49a288e`; native binding SHA-256: `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`; MCTS budget: `2000`.
- Native simulator commit: `7476a81954020087da31d41d16fddf475746ec2d`; ArmG source SHA-256: `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`; ArmG vocabulary SHA-256: `832e199c359af8408ea430ffa3f9fcdc68f32533f7292bb102848d3fb558eb6a`.
- Trace-on outcomes (diagnostic baseline only): victories `0`, defeats `10`; no candidate comparison or win-rate claim.
- Trace coverage: combat decisions `2340`, encounters `142`, noncombat decisions `560`, map routes `270`, complete potion snapshots `3612`.
- Trace-on/off invariance: `PASS`; legal terminal traces `10`; illegal actions/crashes/timeouts `0/0/0`; communication errors `N/A_LOCAL_SIMULATOR`.
- Private artifact inventory: `25` files, `31023129` bytes; manifest SHA-256 `d706291754851fb930d534e4462df3fed92cc4383094ed9925b12595607ab586`.
- Tests: focused H7 trace tests and existing simulator trace tests passed before the diagnostic run.
- Probe10, Dev30, confirmation, Gate/Fresh, real-game verification, promotion, and deployment: `NOT_RUN`.
- G7 remains Champion; this audit made no policy or checkpoint changes.

## Sanitized trace analysis — correlation only

- Across the 10 trace-on train runs, all `462/462` End Turn selections matched the MCTS recommendation. `68/462` decisions had at least one legal, usable potion action.
- Of those 68 opportunity states, 12 were at or below 50% HP and 6 at or below 25% HP. In 31, recorded HP was lower by the next combat decision, with 211 HP of aggregate decline across those transitions. This is temporal association only: the trace does not establish that a potion was beneficial or would have prevented the later loss.
- In the final encounters, 16 End Turn states with a usable potion appeared across 3 runs; 6 healing/regeneration potion slots were present in one run. The last selected action before defeat was a played card in all 10 runs; none ended on an End Turn with a usable potion.
- The H3 broad low-HP emergency-potion override had null paired results across its train/probe/dev checks. H7 does not justify restoring that broad override. A narrower, item-effect-aware End Turn valuation is only a low-confidence hypothesis and requires a fresh train-pool counterfactual before any policy change.
- The user-provided Actions ZIP was matched to run `36674639174` artifact `11079234863`: archive SHA-256 `3f4fc50b9452138e50ed5274cc7ac71a9ad011077785279de7834a478c934e66`. Its `offline-champion.pt` member SHA-256 equals the pinned G7 checkpoint above. Replay payloads were not decoded or used.
- These observations select no Candidate and establish no win-rate gain. A new hypothesis requires a separately preregistered scope and fresh, disjoint seeds; Probe10 and Dev30 remain `NOT_RUN`.
