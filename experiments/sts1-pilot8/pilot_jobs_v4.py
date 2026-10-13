import json,pathlib,time
from collect_v4 import full_state,G7
from collect_v3 import battle
from worlds_v4 import label
from replay_v4 import recorded_index
PILOT=pathlib.Path(__file__).parent/"data"

def label_run(path):
    import torch;torch.set_num_threads(1)
    import slaythespire as sts
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_set_pauses
    source=json.loads(pathlib.Path(path).read_text());policy=ArmGNoncombatPolicy(root=pathlib.Path(__file__).parent/'armg',weight_path=G7)
    gc=sts.GameContext(sts.CharacterClass.IRONCLAD,source['seed'],0);agent=sts.Agent();_set_pauses(agent)
    byordinal={r['provenance']['noncombat_ordinal']:r for r in source['rows']};ordinal=0;count=0
    out=PILOT/'labels';out.mkdir(exist_ok=True)
    for step in range(600):
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        agent.playout(gc)
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        if gc.screen_state==sts.ScreenState.BATTLE:battle(gc,sts);continue
        kind,index,descs,execs,scores=policy.decide(gc,sts)
        index,tie=recorded_index(source,ordinal,kind,index,int(gc.floor_num),descs,scores)
        if tie:
            with (PILOT/'replay-ties.jsonl').open('a') as handle:handle.write(json.dumps(tie)+'\n')
        if ordinal in byordinal:
            row=byordinal[ordinal];dest=out/(row['id']+'.json')
            if full_state(gc,policy,descs)!=row['state']:raise RuntimeError('replay_public_state_mismatch')
            from state_v3 import action_names
            if action_names(gc,policy,kind,descs)!=row['actions']:raise RuntimeError('replay_action_space_mismatch')
            if index!=row['g7_index']:raise RuntimeError('replay_card_teacher_mismatch')
            if policy.module.obs_vec(gc)!=row['armg_observation'] or descs!=row['armg_descriptors']:raise RuntimeError('replay_model_inputs_mismatch')
            if not dest.exists():
                result=label(gc.clone(),policy,sts,row,max_worlds=8);temporary=dest.with_suffix('.tmp');temporary.write_text(json.dumps(result));temporary.replace(dest)
                (PILOT/'label-progress.jsonl').open('a').write(json.dumps({'id':row['id'],'seed':source['seed'],'worlds':len(result['world_results']),'unix':time.time()})+'\n')
            count+=1
        ordinal+=1
        if index<0:gc.skip_reward_cards()
        else:execs[index](gc)
    else:raise RuntimeError('teacher_replay_bound')
    if count!=len(source['rows']) or ordinal!=len(source['journal']):raise RuntimeError('teacher_missing_replay_cases')
    return count
