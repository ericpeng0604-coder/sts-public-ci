# Round017 Train-only card preparation analysis

Issue #19; prerequisite Draft PR #36 at `523590561e808e99d82a9fc20a5f928b723c140e`.
This stage adds deterministic card facts, a bounded offline audit, and an unexecuted single-mechanism shadow rule. It starts zero episodes and leaves G7, the simulator entrypoint and all existing guards unchanged.

## Evidence and scope

Inventory before analysis: ten preparation profiles, 8,698,575 bytes. Scoped Sample(10/10): all existing Round017 Train profiles and each trace's first header only. Validate the frozen collection commit, Train purpose, MCTS2000, pair index, exact eleven-start/completion order, complete safe results, invariance PASS and unchanged public preparation aggregates. Freeze analysis input hashes privately; these hashes authenticate this analysis snapshot, not the original raw trace retrospectively. Raw trace legality/replay verification is reused from the completed collector, not independently repeated.

The card type parser verifies five exact source-file hashes against the corrected frozen native source and aligned native name/type/color arrays. It resolves 364 unique native names across all card colors; this also supports foreign/generated cards in Ironclad decks. Types come from `cardTypes`, not card color or name heuristics. Effects are reviewed only for 23 named cards; unreviewed effects remain unknown.

## Findings

| Observation | Count |
| --- | ---: |
| Reward decisions / accepted cards | 214 / 214 |
| Exact unique legal Skip available / selected Skip | 214 / 0 |
| Accepted known card already present in decision-time deck | 55 |
| Accepted unresolved card identity | 1 |
| Reward decks with unknown native card type | 0 |
| Clash shadow skip opportunities / distinct baseline runs | 10 / 7 |

Reward counts by decision-time act: Act1 93, Act2 76, Act3 45. Upgrade suffixes are normalized for duplicate identity, so base/upgraded copies share a name. The one unresolved selected identity stays unknown; it is not assigned a native card by guessing a vocabulary index.

The four matched Act3 Boss-entry decks total 175 cards: 99 attacks, 52 skills, 17 powers and 7 curses. Reviewed positive-effect counts include 19 cards with direct draw, 5 with persistent Strength and 4 Demon Form cards with Strength per turn. These counts are overlapping card occurrences, not effect magnitudes or proof of an executable engine. One hundred of these 175 cards have unreviewed effects. Therefore neither missing draw nor missing scaling is established by this catalog, and Boss readiness remains NOT_VERIFIED.

## Reviewed effect contract

Source basis: frozen gameplay `7476a81954020087da31d41d16fddf475746ec2d`, with the already corrected native binding. Positive nominal effects are read from [BattleContext.cpp](https://github.com/gamerpuppy/sts_lightspeed/blob/7476a81954020087da31d41d16fddf475746ec2d/src/combat/BattleContext.cpp); conditional actions from [Actions.cpp](https://github.com/gamerpuppy/sts_lightspeed/blob/7476a81954020087da31d41d16fddf475746ec2d/src/combat/Actions.cpp); playability from [CardInstance.cpp](https://github.com/gamerpuppy/sts_lightspeed/blob/7476a81954020087da31d41d16fddf475746ec2d/src/combat/CardInstance.cpp); turn/exhaust timing is checked in the native implementation. The public module records exact reviewed-source hashes and base/upgraded values.

- Damage/conditions: Clash, Anger, Wild Strike, Bash; damage plus draw: Pommel Strike and conditional Dropkick.
- Draw/energy: Shrug It Off, Battle Trance, Offering, Seeing Red; Dark Embrace requires exhaust events.
- Strength: Inflame, Demon Form, conditional Spot Weakness; Flex is temporary Strength with a LoseStrength debuff that can be blocked.
- Block/upgrade/exhaust: Feel No Pain, Metallicize, Armaments, Flame Barrier, Impervious, Power Through, True Grit, Second Wind.

The catalog is partial positive nominal effects, not full card rules or combat predictions. It does not incorporate energy-cost changes, exhaust/ethereal metadata, relic/status/Artifact modifiers, hand caps or all interactions. For example, Offering's nominal HP cost is not guaranteed actual damage, and Flex's debuff is not guaranteed to remove Strength. Inconsistent upgrade suffix/counters, unknown cards and unsupported multi-upgrades have unknown effects. Conditional draw/block/energy/Strength is kept separate from direct effects.

## One next hypothesis

Proposed H23 mechanism: only on a card reward, when G7 recommends Clash (base or upgraded), the entire current deck has known native types and includes at least one nonattack, choose the exact unique legal card Skip. Unknown deck/type/identity or missing/ambiguous Skip preserves G7. This rule uses decision-time state only, has no deck-size threshold and does not alter shop, campfire, map or combat decisions.

The native Clash constraint is on the current *hand*, not on the entire deck. A mixed deck does not prove Clash will be unplayable. This shadow rule hypothesizes that declining another conditional attack may improve consistency; only paired rollouts can assess its cost, benefit, regressions or nulls. The ten observed opportunities do not represent a simulated candidate trajectory: after one real changed reward, later states may diverge.

This differs from previously tested H12 broad deck-size skipping, H13 recommended-duplicate skipping and H14 all-options-duplicate skipping. Those prior rules have no established positive win result and are not reopened here. H23 is a proposed mechanism, not a registered or evaluated candidate. Before rollout, integrate one legal-action-preserving entrypoint, freeze the final code/config, prospectively register a distinct fresh round and exclude every USED/RESERVED pool including all Round017 pools. Use Issue #19's paired Train10, conditional Probe10/Dev30, confirmation and real-game gates without changing thresholds or consuming old seeds.

## Validation and boundaries

57 focused tests passed: nominal upgrade/conditional semantics, unknowns, exact legal shadow Skip, immutable inputs, aligned native arrays/source drift, held-out header rejection, fixed ordered collection, unsafe/bool counters, extra profiles and aggregate/header drift; existing preparation/collector regressions included. Independent source and bounded aggregate review PASS. Final aggregates: 214 accepts, zero skips, 55 known duplicates, ten shadow opportunities across seven runs.

The initial color-filtered analysis incorrectly excluded Combust; it was corrected to parse type independently of color. Both superseded derived summaries and their input snapshots are retained privately. The earlier counts 54 duplicates and eight shadow opportunities across six runs are superseded, not additional samples. No raw evidence was rewritten.

G7 remains retained. Policy execution, fresh paired candidate runs, net win gain, complete card effects, decision-time Boss visibility, formal confirmation and actual-game acceptance are NOT_RUN/NOT_VERIFIED. No model/API call was introduced; provider usage NOT_MEASURED. Original parent-workspace dirty changes remain untouched.
