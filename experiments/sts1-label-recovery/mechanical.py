"""Mechanical replay and termination fixes; unchanged simulator/policies/worlds."""
import math

GAME_STEPS = 10000

def recorded_index(source, ordinal, kind, index, floor, descs, scores):
    from replay_v4 import recorded_index as original
    try:
        return original(source, ordinal, kind, index, floor, descs, scores)
    except RuntimeError:
        expected = source['journal'][ordinal] if ordinal < len(source['journal']) else None
        if expected and kind == 'shop' and expected['kind'] == kind and expected['floor'] == floor:
            old = expected['index']
            if (0 <= old < len(descs) and 0 <= index < len(descs)
                    and len(scores) == len(descs) and descs[old] == descs[index]
                    and all(math.isfinite(scores[i]) for i in (old, index))
                    and abs(scores[old] - scores[index]) <= 1e-5):
                return old, {'seed': str(source['seed']), 'ordinal': ordinal,
                             'kind': kind, 'actual': index, 'recorded': old,
                             'reason': 'equivalent_shop_inputs_float32_tie',
                             'score_gap': abs(scores[old] - scores[index])}
        raise

def finish(gc, sts, policy):
    from roguelike_ai.sts1_phase3.simulator import _set_pauses
    from collect_v3 import battle
    agent = sts.Agent()
    _set_pauses(agent)
    for step in range(GAME_STEPS):
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        agent.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        if gc.screen_state == sts.ScreenState.BATTLE:
            battle(gc, sts)
        else:
            policy.step(gc, sts)
    if gc.outcome == sts.GameOutcome.UNDECIDED:
        raise RuntimeError('game_bound_10000:floor='+str(gc.floor_num)+':screen='+str(gc.screen_state))
    return {'outcome': 'victory' if gc.outcome == sts.GameOutcome.PLAYER_VICTORY else 'defeat',
            'floor': int(gc.floor_num), 'hp': int(gc.cur_hp), 'max_hp': int(gc.max_hp)}
