import pathlib
def battle(gc,sts):
    bc=sts.BattleContext();bc.init(gc)
    for _ in range(1200):
        if bc.outcome!=sts.Outcome.UNDECIDED:break
        legal=list(sts.get_legal_actions(bc))
        if not legal:raise RuntimeError('no_legal_combat_action')
        chosen=legal[0] if len(legal)==1 else sts.mcts_recommend(bc,2000)
        if chosen is None:raise RuntimeError('mcts_no_action')
        chosen.execute(bc)
    else:raise RuntimeError('battle_bound')
    bc.exit_battle(gc)

def finish(gc,sts,policy):
    from roguelike_ai.sts1_phase3.simulator import _set_pauses
    agent=sts.Agent();_set_pauses(agent)
    for _ in range(600):
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        agent.playout(gc)
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        if gc.screen_state==sts.ScreenState.BATTLE:battle(gc,sts)
        else:policy.step(gc,sts)
    else:raise RuntimeError('game_bound')
    return {'outcome':'victory' if gc.outcome==sts.GameOutcome.PLAYER_VICTORY else 'defeat',
            'floor':int(gc.floor_num),'hp':int(gc.cur_hp),'max_hp':int(gc.max_hp)}
