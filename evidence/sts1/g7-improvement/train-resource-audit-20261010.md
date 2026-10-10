# G7 Train loss diagnosis: stale combat resource observations

During combat, legacy diagnostic potion inventories read GameContext, while
actions consume BattleContext's separate inventory. Native exitBattle synchronizes
the latter back to GameContext only after combat. A previously consumed potion can
therefore appear in the legacy run inventory beside no legal use action. Such a
record cannot support the claim that G7 overlooked an available potion.

Issue #19; parent prerequisite PR #30 at
`6ec9c90577b16a934b8b6053542070a4ffe5a65b`. Branch:
`codex/sts1-g7-train-loss-audit-20261010`.

## Bounded evidence

Read-only Inventory followed by explicitly scoped Sample of the 10 Round013
Train parent traces: 36,084,074 bytes, 4,092 combat decisions. No Probe/Dev/formal
traces or outcomes were used to form a new strategy. Private identifiers and
provenance fingerprints remain outside Git.

There were 95 canonical potion actions with another decision in the same
encounter. In 92 cases, the next decision retained the old run potion name while
the corresponding legal-use slot disappeared; 8 of 10 traces were affected.
These are suspicious inventory observations, not 92 missed uses or losses.
Native source confirms separate ownership: `BattleContext::init` copies the run
inventory; potion actions use `bc.potions`; `BattleContext::exitBattle` copies it
back. The registered legacy Python binding exposed only GameContext slots.

Leading diagnostic findings:

1. **Confirmed observation defect:** stale run resources can mislabel consumed
   potions as available. This undermines resource-based failure attribution,
   without proving it caused G7's losses.
2. **Limited late-turn correction opportunity:** 11 End Turn decisions across
   6 traces had intent damage at least HP plus block; 9 had zero energy. Earlier
   decisions or automatic revival may matter. Intent damage alone does not prove
   avoidable death, and stale inventory cannot establish a rescue option.
3. **Weak intervention evidence:** the prior completed H20 pairs had no discordant
   outcomes. Adding a rarely selected card did not establish a correctable loss
   cause. This is historical result reporting, not held-out-data strategy mining.

## Correction

Add a bounds-checked read-only `BattleContext.combat_potions` getter to the pinned
binding. Preserve all five slots and return a copy; modify no gameplay source.
Keep legacy run inventory intact with explicit `run_context` scope, and record
live `combat_potion_inventory` separately in diagnostic states and signatures.
Missing or malformed live inventory fails closed; there is no GameContext or
history-inference fallback. New trace headers require validated live resources;
historical traces remain retained as historical run-context evidence.

The versioned binding patch is verified against its unchanged registered hash.
Its bytes are restored to LF and a scoped .gitattributes rule preserves that
exact representation on Windows checkouts.
Its contexts are matched exactly and uniquely after the registered hook rearranges
unused wrappers; ambiguous or absent context is rejected. Original pinned source,
hook, patch, safety and legal-action checks remain in force.

## Validation and limits

118 focused tests passed, with 2 additional subtests. Local pinned native build passed with the new getter.
Zero-episode native smoke confirmed five live slots, a copy rather than writable
native state, unchanged native state and complete legal-action bits, and an equal
MCTS2000 recommendation on untouched versus observed clones. This is a synthetic
single-decision check, not full-episode trace invariance or win-rate acceptance.

GitHub Windows native CI also passed for code commit
`fc2a60580d84b01903d259d578ae9cdbc2e0247e`:
https://github.com/ericpeng0604-coder/sts-public-ci/actions/runs/38052879894
The build, native smoke, focused binding checks and artifact upload all completed
successfully. This subsequent report update changes documentation only.

All failed build/preflight attempts and logs were retained privately: CRLF patch
identity, reference-source mismatch, missing dependency trees, legacy CMake policy
compatibility, and shifted patch line numbers. The successful r6 build used a
new pristine source and genuinely configured reference; no old artifacts changed.

No new evaluation episodes were run; no candidate action rule or G7 weight changed.
The prior 20 paired seeds still have G7/H20 wins 3/20 each, C-only/P-only 0/0,
net 0, p=1. Win-rate improvement remains NOT_VERIFIED. New candidate selection,
full-episode invariance, fresh paired evaluation, confirmation and real-game
acceptance are NOT_RUN. Provider usage is NOT_MEASURED.
Existing potion interventions must not be reused without independently checking
that their effect/identity inputs come from live BattleContext resources.

Changed scope: builder, simulator diagnostics, H7 trace validator, Train-only
resource audit and zero-episode smoke under allowed STS1 script/source paths;
three focused test files and this report. No STS2/PPO, Champion, formal data,
deployment, promotion or merge changes.

Changed files:

- `scripts/sts1/build_sts1_pinned_binding.py`
- `control/sts1-g7-improvement/native-patches/.gitattributes`
- `scripts/sts1/sts1_g7_h7_potion_trace_audit.py`
- `scripts/sts1/sts1_g7_train_resource_audit.py`
- `scripts/sts1/sts1_combat_resource_smoke.py`
- `src/roguelike_ai/sts1_phase3/simulator.py`
- `tests/test_sts1_pinned_binding_builder.py`
- `tests/test_sts1_combat_resources.py`
- `tests/test_sts1_train_resource_audit.py`
- `evidence/sts1/g7-improvement/train-resource-audit-20261010.md`
