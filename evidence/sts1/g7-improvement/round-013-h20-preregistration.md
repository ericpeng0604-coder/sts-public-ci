# H20: Inflame before MCTS End Turn

Issue #19. Discovery: read-only Round011 parent Train-only Sample(10), 2,936
combat decisions with complete canonical legal actions. Seven decisions across
four eligible traces offered legal Inflame when MCTS ended its turn; one decision
against Awakened One is excluded, leaving six decisions across four traces.
This is recurrence, not causal or win-rate evidence. No Probe/Dev/formal trace
or outcome was used to choose H20. Raw rows and seed IDs remain private.

The Fire Potion alternative is rejected for this round: three end-turn states
across three Train traces offered legal use, but only one had an enemy within
20 damage and that enemy had no current attack intent. Do not broaden this
alternative or count it as cross-seed support for a lethal-attack intervention.

## Frozen decision rule

Override only a canonical legal MCTS End Turn, with complete canonical legal
actions and well-formed public decision state. Choose a legal, playable,
non-targeted INFLAME with nonnegative integer cost covered by current energy;
break ties by lowest hand position. Repeated identical non-targeted action
projections retain the first exact native-list ordinal; differing projections
at the same hand position remain ambiguous and fail closed. Pinned BattleContext::playCard queues
Strength +2 (+3 upgraded) for INFLAME. It costs a card/energy and can interact
with relic/card mechanics; recurrence alone does not prove benefit.

Fail closed for Awakened One, Time Eater or Corrupt Heart; Pain or Normality in
hand; malformed card/enemy/power state; empty living-enemy list; unknown power
names; ambiguous legal Inflame at a hand position. Allow enemy power names only
Strength, Weak and Vulnerable; allow player powers only Strength. These narrow
public-state exclusions are frozen before fresh episodes and may not be widened
after their results. Preserve every recommendation, exact selected action,
reason, full raw evidence and legal actions in any runtime implementation.

## Prospective gates and boundaries

H20 is a hypothesis implementation, not an enabled runtime Candidate. No pool
or episode is allocated here. Before a fresh Round013 Train10 counterfactual:
freeze simulator/policy/runner identities, private runtime identity lock,
exclusion inventory and mutually disjoint Train/Probe/Dev pools. Use G7,
pinned gameplay 7476a81954020087da31d41d16fddf475746ec2d and MCTS-2000 for both
arms. Require complete matched initial state/prefix and exact legal selections.
Train/Probe require effective changes on >=3 distinct seeds, nonnegative net,
and every hard guard. Dev requires 30 complete pairs, positive net and hard
guards; exact p and floor/HP are reported under the prospective H19+ contract.
Confirmation gates and permanent trial counting remain unchanged.

Do not reuse any consumed pool, mutate G7/Champion, promote, merge or launch
the real game. Runtime integration, fresh paired outcomes, confirmation and
win-rate gain remain NOT_RUN/NOT_VERIFIED until separately proven. Do not
reinterpret an offline action change as a win. Provider usage NOT_MEASURED.
