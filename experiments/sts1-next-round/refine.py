"""Add shared future worlds to selected consumed-pilot states; no fresh evaluation."""
import argparse,json,pathlib,sys,time
from common import ROOT,PILOT,OUT,save
sys.path.insert(0,str(PILOT))

def extend(gc,policy,sts,old):
    from collect_v4 import full_state
    from collect_v3 import finish
    from state_v3 import action_names
    from worlds_v4 import world_seed,aggregate
    row=dict(old);worlds=list(old['world_results']);seeds=list(old['world_seeds'])
    assert len(worlds)==8 and seeds==[world_seed(row['id'],i) for i in range(8)]
    actions=row['actions'];parent=row['g7_index']
    for target in (16,32):
        for index in range(len(worlds),target):
            seed=world_seed(row['id'],index);base=gc.sample_public_future(seed)
            kind,descs,_=policy.choices(base)
            assert full_state(base,policy,descs)==row['state'] and action_names(base,policy,kind,descs)==actions
            outcomes=[]
            for option in range(len(actions)):
                branch=base.clone();kind,descs,execs=policy.choices(branch)
                assert action_names(branch,policy,kind,descs)==actions
                execs[option](branch);outcomes.append(finish(branch,sts,policy))
            kind,descs,_=policy.choices(base);assert full_state(base,policy,descs)==row['state']
            worlds.append(outcomes);seeds.append(seed)
        estimates=aggregate(worlds,parent)
        # Precision diagnostic only. Use 3-look-corrected distribution-free radius;
        # stop early only if one option has separated simultaneous intervals.
        import math
        radius=math.sqrt(2*math.log(2*len(actions)*3/.05)/len(worlds))
        best=max(range(len(actions)),key=lambda a:estimates[a]['win_advantage'])
        lower=0. if best==parent else estimates[best]['win_advantage']-radius
        separated=lower>max(0. if a==parent else estimates[a]['win_advantage']+radius for a in range(len(actions)) if a!=best)
        if separated:break
    radius=math.sqrt(2*math.log(2*len(actions)*3/.05)/len(worlds))
    for e in estimates:e['three_look_lower']=0. if e['index']==parent else max(-1,e['win_advantage']-radius);e['three_look_upper']=0. if e['index']==parent else min(1,e['win_advantage']+radius)
    return {**row,'world_seeds':seeds,'world_results':worlds,'estimates':estimates,'worlds_added':len(worlds)-8,'uncertain':not separated,'refinement_use':'consumed old-pilot diagnostic/training only','all_options_share_worlds':True,'original_eight_worlds_preserved':True}

def main(index,downloads):
    import torch;torch.set_num_threads(1)
    import slaythespire as sts
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_set_pauses
    from collect_v4 import full_state,G7
    from collect_v3 import battle
    from replay_v4 import recorded_index
    from state_v3 import action_names
    path=next(downloads.rglob('refinement-shards/'+str(index)+'.json'));targets=set(json.loads(path.read_text()));assert len(targets)==1
    source=next(json.loads(p.read_text()) for p in (PILOT/'input-scouts').glob('*.json') if targets&{r['id'] for r in json.loads(p.read_text())['rows']})
    original=json.loads(next(downloads.rglob('original-labels/'+next(iter(targets))+'.json')).read_text())
    target_rows={r['provenance']['noncombat_ordinal']:r for r in source['rows'] if r['id'] in targets}
    policy=ArmGNoncombatPolicy(root=PILOT/'armg',weight_path=G7);gc=sts.GameContext(sts.CharacterClass.IRONCLAD,source['seed'],0);agent=sts.Agent();_set_pauses(agent)
    ordinal=0;done=[];ties=[];start=time.time()
    for step in range(600):
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        agent.playout(gc)
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        if gc.screen_state==sts.ScreenState.BATTLE:battle(gc,sts);continue
        kind,index_now,descs,execs,scores=policy.decide(gc,sts)
        index_now,tie=recorded_index(source,ordinal,kind,index_now,int(gc.floor_num),descs,scores)
        if tie:ties.append(tie)
        if ordinal in target_rows:
            row=target_rows[ordinal]
            assert full_state(gc,policy,descs)==row['state'] and action_names(gc,policy,kind,descs)==row['actions']
            assert index_now==row['g7_index'] and policy.module.obs_vec(gc)==row['armg_observation'] and descs==row['armg_descriptors']
            result=extend(gc.clone(),policy,sts,original);save('refined-labels/'+row['id']+'.json',result);done.append(row['id'])
            if set(done)==targets:break
        ordinal+=1
        if index_now<0:gc.skip_reward_cards()
        else:execs[index_now](gc)
    assert set(done)==targets
    summary={'status':'PASS','shard':index,'cases':done,'worlds':len(result['world_results']),'new_rollouts':result['worlds_added']*len(result['actions']),'seconds':time.time()-start,'replay_ties':ties,'combat_mcts':2000}
    save('refinement-summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True);parser.add_argument('--downloads',type=pathlib.Path,required=True);args=parser.parse_args();main(args.shard,args.downloads)
