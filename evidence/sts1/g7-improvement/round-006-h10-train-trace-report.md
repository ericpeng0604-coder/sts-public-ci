# Round 006 H10 G7-only train trace audit — complete, no policy change

## Registered scope and provenance

- Issue #19 preregistration: comment 6079340202. This is a diagnostic trace audit, not a Candidate comparison.
- Pool: `round-006-20261009-train_hypothesis_3`, 10 seeds; manifest SHA-256 `61e017611edb50bf0b68ad633ee49034e1985c8176a3cb8fcd043aaea80fb9aa`. The pool passed pairwise disjointness and exclusion-inventory preflight before execution. Seed IDs remain private.
- Parent: unchanged G7 checkpoint SHA-256 `8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`. Candidate policy: G7 unchanged.
- Budget: MCTS-2000; 10 complete G7-only episodes; pinned native simulator source label `7476a81954020087da31d41d16fddf475746ec2d`; wrapper source SHA-256 `a9ac8c65ac665c2e75f7e1e807dd140d67f3532acd092b60b5d5a55092cadf72`; binding SHA-256 `bc2a3d272c5dc1f51f66619604fb1b202e0915f29dddd719837e5ca0b8cfc89e`; ArmG source SHA-256 `7b4417484ade4320996f4ce0f2154944e6bd75e90500ab8f209bc84ab67d7f3b`; evaluator SHA-256 `ef7ec2aec13f73cef7656b96b3f0ca7cbcbf72b3e2baf6435e3cc0c250239ca9`.
- Trace-on/off invariance was reused from the H9 check because the simulator source SHA is unchanged; it was not rerun on these H10 seeds.

## Results

- Outcomes: 2 victories, 8 defeats. Terminal floor bands: 2 at 50+, 8 at floors 17–33.
- Safety and completeness: illegal actions 0, crashes 0, timeouts 0, communication errors 0; all 10 terminal records complete.
- Trace coverage: 2,832 combat decisions, 164 encounters, 733 noncombat decisions, 350 route decisions, and 4,472 potion snapshots.
- Across 8 defeats, the last recommendation was End Turn in 8; none of those defeats ended at floor 45 or later.
- At floor 45+, there were 14 End Turn decisions. All 14 had no legal Skill and had complete potion inventory; 5 had a legal Use Potion action and usable potion. None of the 14 states had a fully supported attack-intent/damage/hit projection for every living enemy.
- Private evidence inventory: 21 files, 37,895,984 bytes; artifact manifest SHA-256 `f117ccf04c7dfcd688910107d544004a8909c8f1f1a12f4962db0b2f7fa29d6e`. Raw traces and seed-level records remain private.

## Interpretation and next direction

This train-only sample does not reproduce the earlier H6 pattern of floor-50 defeats ending with an End Turn recommendation. Although late-floor End Turn states occurred, the available legal options and incomplete enemy-intent projection do not support a safe, deterministic intervention. No policy or model was changed, and no Probe, Dev, or confirmation run was started from this diagnostic.

After the repeated null potion-focused results and this non-replication, the next direction is to inspect already-seen Dev noncombat/reward traces for a bounded hypothesis-generation review. That pool must be relabeled with its history before reuse; any strategy counterfactual must use a new, disjoint train pool. H10's pool is consumed and will not be reused.

## Boundaries

- This is a simulator diagnostic only. Gate/Fresh results, trajectories, decisions, and labels were not accessed or used; real-game and production acceptance are NOT_RUN/NOT_VERIFIED.
- G7 remains Champion. No Candidate promotion, deployment, or confirmation claim is made.
- The report and summary are local until their commit is synchronized to GitHub; remote publication is NOT_VERIFIED.
