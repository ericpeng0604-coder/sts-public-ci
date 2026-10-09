# Round 006 H11 direction review — complete, no episodes

## Registered scope and data handling

- Issue #19 comment 6079488103 registered this local deterministic review before reading decisions.
- The already-seen H9 Dev30 pool was reclassified before trace access as `hypothesis_generation_only`; its earlier `H9_dev30` use and null result remain in the ledger history.
- Pool manifest SHA-256: `a9f195b41525aab30e8c58ec78db6564685d086bc2f7e54a83c4cf31b10572c5`. The existing private artifact manifest SHA-256 is `485dc3cf862db05f421a8c4b43a6c86280f05fcb1bf72622a2a8ebac5f8dc00`.
- Read only 30 G7 parent trace files. Candidate traces, per-seed outcomes, terminal labels, and seed IDs were not read or published. Raw traces remain private. No episodes ran.

## Aggregate decision coverage

All 2,414 parent noncombat decisions had complete legal choices and a valid selected legal-action index; malformed trace lines: 0.

| Screen | Decisions | G7 selected category | Available legal-choice categories |
| --- | ---: | --- | --- |
| Event | 203 | option 0: 18; 1: 72; 2: 72; 3: 39; 4: 2 | option 0: 178; 1: 195; 2: 109; 3: 41; 4: 3; 5: 2 |
| Map | 1,138 | combat: 379; elite: 77; event: 253; rest: 210; shop: 83; treasure: 70; boss: 66 | combat: 760; elite: 111; event: 309; rest: 241; shop: 94; treasure: 101; boss: 66 |
| Rest | 210 | option 0: 134; option 1: 76 | option 0: 180; 1: 191; 2: 210; 3: 6; 4: 6 |
| Card reward | 554 | take card: 554; skip: 0 | card choices: 1,666; skip choices: 554 |
| Shop | 309 | buy card: 191; buy potion: 13; buy relic: 5; leave: 100 | card: 1,041; potion: 500; relic: 194; leave: 309 |

Categories are coarse and aggregated across the pool. The report excludes card names, event identifiers, item names, exact states, route paths, seed IDs, and per-seed results.

## Direction and next experiment

The most direct new hypothesis is card-reward selectivity: G7 selected a card on all 554 observed reward decisions even though a skip action was legal once per decision. This establishes repeated exposure and a testable action contrast, not that skipping would improve wins. Map, event, rest, and shop choices were also fully traceable, but these aggregates alone do not establish which alternatives were better.

Next, preregister a single skip-selectivity counterfactual on a new disjoint train pool. Define the eligibility rule using only decision-time state and parent scores before running; retain every eligible and ineligible sample. Do not fit or tune the rule on this already-seen Dev pool. A null or negative train result will be recorded without Probe or Dev follow-on.

## Boundaries

- This is hypothesis generation from previously seen Dev traces, not a new evaluation or training run.
- Gate/Fresh data were not accessed. Confirmation, real-game verification, promotion, and deployment remain NOT_RUN/NOT_VERIFIED.
- G7 remains Champion. The report and ledger are local until GitHub synchronization succeeds.
