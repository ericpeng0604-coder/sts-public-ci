import argparse,concurrent.futures,json,multiprocessing,pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).parent.parent/'sts1-next-train'))
from paths import ROOT,PILOT,NEXT,OUT,save
sys.path.insert(0,str(PILOT))

def job(task):
    import torch;torch.set_num_threads(1)
    import slaythespire as sts
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_set_pauses
    from collect_v3 import battle
    from mechanical import finish
    from collect_v4 import full_state,G7
    from mechanical import recorded_index
    from state_v3 import action_names
    from worlds_v4 import world_seed,aggregate
    source=json.loads(((PILOT/'input-scouts') if task['source']=='pilot' else (OUT/'scouts')).joinpath(task['file']).read_text())
    targets={r['provenance']['noncombat_ordinal']:r for r in source['rows'] if r['id'] in task['ids']};assert len(targets)==len(task['ids'])
    policy=ArmGNoncombatPolicy(root=PILOT/'armg',weight_path=G7);gc=sts.GameContext(sts.CharacterClass.IRONCLAD,source['seed'],0);agent=sts.Agent();_set_pauses(agent)
    ordinal=0;done=[];ties=[]
    reserved=set(json.loads((NEXT/'cohort-manifest.json').read_text())['cohorts']['evaluation_reserved'])
    for step in range(600):
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        agent.playout(gc)
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        if gc.screen_state==sts.ScreenState.BATTLE:battle(gc,sts);continue
        kind,index,descs,execs,scores=policy.decide(gc,sts);index,tie=recorded_index(source,ordinal,kind,index,int(gc.floor_num),descs,scores)
        if tie:ties.append(tie)
        if ordinal in targets:
            row=targets[ordinal];assert full_state(gc,policy,descs)==row['state'] and action_names(gc,policy,kind,descs)==row['actions']
            assert index==row['g7_index'] and policy.module.obs_vec(gc)==row['armg_observation'] and descs==row['armg_descriptors']
            old_path=OUT/'labels'/(row['id']+'.json')
            old=json.loads(old_path.read_text()) if old_path.exists() else row
            for key in ('state','actions','g7_index','armg_observation','armg_descriptors','provenance'):assert old[key]==row[key], 'resume_reference:'+key
            worlds=list(old.get('world_results',[]));seeds=list(old.get('world_seeds',[]))
            assert len(worlds)==len(seeds) and len(worlds)<=task['worlds']
            assert seeds==[world_seed(row['id'],i) for i in range(len(seeds))]
            for windex in range(len(worlds),task['worlds']):
                seed=world_seed(row['id'],windex);assert str(seed) not in reserved;root=gc.sample_public_future(seed)
                k,ds,_=policy.choices(root);assert full_state(root,policy,ds)==row['state'] and action_names(root,policy,k,ds)==row['actions']
                outcomes=[]
                for a in range(len(row['actions'])):
                    branch=root.clone();k,ds,executors=policy.choices(branch);assert action_names(branch,policy,k,ds)==row['actions']
                    executors[a](branch)
                    try:outcomes.append(finish(branch,sts,policy))
                    except Exception as exc:raise RuntimeError('continuation_context:'+repr({'source_seed':str(source['seed']),'id':row['id'],'world_index':windex,'option':a})+':'+str(exc)) from exc
                k,ds,_=policy.choices(root);assert full_state(root,policy,ds)==row['state']
                worlds.append(outcomes);seeds.append(seed)
                checkpoint={**old,'g7_scores':scores,'training_cohort':task['cohort'],'world_results':worlds,'world_seeds':seeds,'estimates':aggregate(worlds,index),'teacher_combat_mcts':2000}
                temporary=old_path.with_suffix('.tmp');temporary.write_text(json.dumps(checkpoint));temporary.replace(old_path)
            result={**old,'g7_scores':scores,'training_cohort':task['cohort'],'world_results':worlds,'world_seeds':seeds,'estimates':aggregate(worlds,index),'teacher_combat_mcts':2000,'label_status':'PAIRED_STOCHASTIC_FUTURE_ESTIMATE','sampler_limit':'Approximate public-state-compatible stochastic transitions, not an exact seed posterior','uncertain':True,'mechanical_amendment':'recovery-v1-unchanged-native-shop-ties-10000-bound'}
            temporary=old_path.with_suffix('.tmp');temporary.write_text(json.dumps(result));temporary.replace(old_path);done.append(row['id'])
            if len(done)==len(targets):break
        ordinal+=1
        if index<0:gc.skip_reward_cards()
        else:execs[index](gc)
    assert set(done)==set(task['ids'])
    return {'source':task['file'],'cases':len(done),'cohort':task['cohort'],'replay_ties':len(ties)}

def main(index,downloads):
    import shutil
    tasks=json.loads((OUT/'tasks.json').read_text())[index];wanted={i for task in tasks for i in task['ids']};start=time.time()
    previous=downloads/('next-train-label-'+str(index))
    assert previous.is_dir(), 'missing_previous_shard'
    reused=0
    for path in previous.rglob('labels/*.json'):
        if path.stem not in wanted:continue
        row=json.loads(path.read_text());destination=OUT/'labels'/path.name
        existing=json.loads(destination.read_text()) if destination.exists() else None
        if existing:
            for key in ('state','actions','g7_index','armg_observation','armg_descriptors','provenance'):assert row[key]==existing[key]
            n=min(len(row['world_results']),len(existing.get('world_results',[])))
            assert row['world_results'][:n]==existing.get('world_results',[])[:n], 'resume_prefix_changed'
        if not existing or len(row['world_results'])>=len(existing.get('world_results',[])):
            shutil.copyfile(path,destination)
    pending=[]
    for task in tasks:
        ids=[]
        for ident in task['ids']:
            path=OUT/'labels'/(ident+'.json');row=json.loads(path.read_text()) if path.exists() else None
            if row and len(row.get('world_results',[]))==task['worlds'] and row.get('training_cohort')==task['cohort']:reused+=1
            else:ids.append(ident)
        if ids:pending.append({**task,'ids':ids})
    print(json.dumps({'shard':index,'reused_completed_cases':reused,'pending_cases':sum(len(t['ids']) for t in pending)}),flush=True)
    errors=[];results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in concurrent.futures.as_completed([pool.submit(job,t) for t in pending]):
            try:results.append(future.result())
            except Exception as exc:errors.append(str(exc));print(json.dumps({'shard':index,'error':str(exc)}),flush=True)
    for path in (OUT/'labels').glob('*.json'):
        if path.stem not in wanted:path.unlink()
    if errors:save('recovery-errors.json',errors);raise RuntimeError('recovery_failed:'+repr(errors))
    assert {p.stem for p in (OUT/'labels').glob('*.json')}==wanted
    for task in tasks:
        for ident in task['ids']:
            row=json.loads((OUT/'labels'/(ident+'.json')).read_text());assert len(row['world_results'])==task['worlds'] and row['training_cohort']==task['cohort']
    report={'status':'PASS','shard':index,'cases':len(wanted),'seconds':time.time()-start,'combat_mcts':2000,'source_results':results,'reused_completed_cases':reused,'mechanical_amendment':'recovery-v1-unchanged-native-shop-ties-10000-bound'}
    save('shard-summary.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_results'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True);parser.add_argument('--downloads',type=pathlib.Path,required=True);args=parser.parse_args();main(args.shard,args.downloads)
