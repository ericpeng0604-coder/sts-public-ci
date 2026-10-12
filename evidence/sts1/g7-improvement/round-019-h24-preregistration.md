# Round019 H24 preregistration — two target draw cards

Registered before any Round019 episode. Issue #19; prerequisite unmerged PR #38. H23 remains stopped for Probe coverage below three; no old pool is reopened. G7 remains Champion.

## Train-only basis and one fixed intervention

Round017's ten validated Train preparation profiles contain seven selector-eligible rewards across four runs. Inventory: ten profiles, 8,698,575 bytes. Scoped Deep10/10 analysis used reward/deck fields only, checked original status/ordered usage/provenance/bounded headers and derived aggregates with `load_round17`, then invoked the exact H24 selector using hash-verified native types. Raw traces were not independently revalidated and shadow opportunities are not counterfactual wins. Probe/Dev/Formal outcomes were not used to select or tune H24.

On a card reward, preserve the G7 recommendation unless it accepts a card outside the fixed target set {Pommel Strike, Shrug It Off, Battle Trance}, the fully identified deck contains fewer than two cards in that set, and exactly one offered choice is an exact base-name target card. Select that existing legal ordinal. Existing target upgrades must have consistent native names/counters and reviewed effects. Preserve G7 on unknown cards/types/deck/choices, incomplete choices, multiple targets, missing target, target already recommended, or G7 Skip. The threshold is fixed at two before fresh outcomes; no deck-size threshold, reward-score sweep, combat/route/shop/campfire change or H23 combination.

The target set contains reviewed draw effects, not a complete draw/readiness catalog. Other cards and relics can draw. Battle Trance prevents further draw that turn; all three consume hand space and have different costs/effects. This tests a narrow preparation hypothesis, not proof that all such rewards are beneficial or that the deck lacks draw. No model/API or training is introduced.

## Prospective allocation, budget and evidence

Use new mutually disjoint Round019 pools excluding the full prior inventory plus all five Round018 USED/RESERVED pools, including untouched Dev and unused train pools. Freeze code, selector, reviewed parser/native sources, prefix helper, corrected binding, G7/ArmG/checkpoint, evaluator/helpers and complete inventory/manifests before outcomes. Both arms: corrected gameplay `7476a81954020087da31d41d16fddf475746ec2d`, Ironclad A0, G7, MCTS2000.

Train10 = 20 paired episodes plus one first-candidate trace-off replay. Conditional Probe10 = 20; conditional Dev30 = 60. Maximum 50 pairs and 101 episodes. This is a fixed prospective stage structure, not an instruction to bypass a failed gate or add samples for significance. No automatic retry. Confirmation is separate and NOT_RUN.

Train/Probe require at least three effective override seeds, nonnegative net and all existing hard guards. Dev requires complete pairs and positive net with guards; exact sign p is diagnostic. Floor/HP deltas are secondary. An integrity/safety failure stops the round with partial evidence retained.

Every override is recomputed against the frozen selector and exact canonical choices. Effective coverage requires matched recorded state/action prefix up to the first changed reward, equal pre-action deck/choices/G7 recommendation, then exact card multiset verification: parent adds the recommended card and candidate adds the selected target, preserving all previous cards. Changing HP/gold alone cannot count. This does not serialize hidden RNG state. Preserve full raw states/actions/terminal/provenance privately; output public counts only.

Pre-episode verification: 128 tests and 2 subtests passed; corrected native resource smoke PASS with zero episodes. Independent review identified and prompted repair of a coverage-proof gap before freeze; the fixture now rejects identical added cards with a different HP, wrong added cards and alteration of existing cards. Runtime outcomes, trace-off replay, formal confirmation and actual-game acceptance remain NOT_RUN/NOT_VERIFIED; provider usage NOT_MEASURED. Report positives, negatives and nulls; no promotion or Champion replacement.
