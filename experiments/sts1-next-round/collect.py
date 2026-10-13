"""Fresh G7+MCTS2000 captures; no candidate decisions and no world labels."""
import argparse,concurrent.futures,hashlib,json,multiprocessing,pathlib,sys,time
from common import ROOT,PILOT,OUT,save
sys.path.insert(0,str(PILOT))

def scout(task):
    import torch;torch.set_num_threads(1)
    import slaythespire as sts
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_set_pauses
    from collect_v4 import full_state,G7
    from collect_v3 import battle
    from state_v3 import action_names
    seed=int(task['seed']);policy=ArmGNoncombatPolicy(root=PILOT/'armg',weight_path=G7)
    gc=sts.GameContext(sts.CharacterClass.IRONCLAD,seed,0);agent=sts.Agent();_set_pauses(agent)
    rows=[];journal=[];combats=0
    for step in range(600):
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        agent.playout(gc)
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        if gc.screen_state==sts.ScreenState.BATTLE:battle(gc,sts);combats+=1;continue
        kind,index,descs,execs,scores=policy.decide(gc,sts)
        if kind=='card' and len(descs)>=2:
            state=full_state(gc,policy,descs);actions=action_names(gc,policy,kind,descs)
            ident=hashlib.sha256(json.dumps({'state':state,'actions':actions},sort_keys=True).encode()).hexdigest()
            rows.append({'id':ident,'split':task['split'],'state':state,'actions':actions,'g7_index':index,'g7_scores':scores,'armg_observation':policy.module.obs_vec(gc),'armg_descriptors':descs,'provenance':{'seed':seed,'noncombat_ordinal':len(journal),'combat_before':combats},'label_status':'UNLABELLED'})
        journal.append({'kind':kind,'index':index,'floor':int(gc.floor_num)})
        if index<0:gc.skip_reward_cards()
        else:execs[index](gc)
    else:raise RuntimeError('scout_game_bound')
    return {'seed':seed,'split':task['split'],'outcome':str(gc.outcome),'floor':int(gc.floor_num),'combat_mcts':2000,'rows':rows,'journal':journal}

def main(index):
    tasks=json.loads((ROOT/'collection-shards.json').read_text())[index]
    start=time.time();destination=OUT/'scouts';destination.mkdir(parents=True,exist_ok=True);results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(scout,tasks):
            (destination/(str(result['seed'])+'.json')).write_text(json.dumps(result));results.append(result)
    report={'status':'PASS','shard':index,'source_runs':len(results),'states':sum(len(r['rows']) for r in results),'seconds':time.time()-start,'combat_mcts':2000}
    assert {str(r['seed']) for r in results}=={t['seed'] for t in tasks}
    save('collection-summary.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True);args=parser.parse_args();main(args.shard)
