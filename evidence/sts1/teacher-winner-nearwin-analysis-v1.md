# STS1 Teacher — Winner vs Near-win analysis v1

## Purpose
Record early hypotheses from rich-telemetry winner mining before changing Teacher behavior. These are hypotheses to validate on more fresh seeds, not hard-coded rules.

## Current paired observations

### Pair A
- Winner seed: `594207388`
- Near-win seed: `426128805`
- Both reached the late game / Act 3 boss region and showed overlapping capabilities/cards including Perfected Strike, Power Through, Shockwave, Impervious, Reaper, and Inflame.
- Winner later added `Feel No Pain -> Second Wind+ -> Battle Trance+`, creating a more coherent exhaust/defense/draw package.
- Near-win continued adding multiple Perfected Strikes and Power Throughs plus Immolate+, increasing raw output but also deck thickness; it died at floor 50.

### Pair B
- Winner seed: `672547106`
- A comparable near-win had overlapping Perfected Strike / Offering / Power Through capabilities.
- Winner's late additions included `Flame Barrier+`, `Pommel Strike+`, and `Shockwave`.
- The comparable near-win also reached floor 50 but did not convert to a victory.

## Working hypothesis
ArmG/card-choice valuation may overweight individually strong cards and underweight:
1. current deck capability gaps,
2. deck thickness / cycle quality,
3. late-game defense and draw consistency,
4. synergy packages such as exhaust + defense + draw,
5. requirements of the upcoming Act 3 boss / threat.

Do **not** encode this as a fixed archetype preference. The intended Teacher improvement is contextual: evaluate what the current deck can already do, what it lacks, and what the next threat requires.

## Validation gate before changing Teacher
Continue fresh-seed winner mining. Re-run winner↔near-win matching as the winner sample grows. Promote this hypothesis into a Teacher Candidate only if the same pattern repeats across additional independent winners / near-winners.

## Collection status
Winner Mining remains enabled with MCTS-2000, rich combat play traces, semantic ArmG non-combat telemetry, 8-way parallel sharding, merge, and automatic next-round dispatch.
