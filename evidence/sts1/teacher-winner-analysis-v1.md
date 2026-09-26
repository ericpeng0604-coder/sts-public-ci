# Teacher winner-mining analysis v1

Source run: 36210475180
Search budget: fixed MCTS-2000
Fresh seeds: 100
Wins: 10
Near-wins (floor >=45): 28

## First observed signal
The winning trajectories preserve more HP than near-wins while using the same MCTS budget.

- Mean combat-state HP, winners: 68.31
- Mean combat-state HP, near-wins: 65.56
- Mean minimum combat-state HP, winners: 15.70
- Mean minimum combat-state HP, near-wins: 9.68

By act (mean combat-state HP):
- Act 1: winners 72.9 vs near-wins 69.1
- Act 2: winners 67.4 vs near-wins 66.0
- Act 3: winners 64.4 vs near-wins 62.2

This is correlation, not yet a causal rule. Do not hard-code an HP heuristic from these numbers alone.

## Next experiment
Keep MCTS at 2000. Mine another independent fresh-seed batch and test whether the HP-preservation signal repeats. Also retain full winner and near-win trajectories for state/action matching so later Teacher changes can target decision quality rather than search volume.

Formal fixed-50 seeds remain an exam only and are excluded from mining/training batches.
