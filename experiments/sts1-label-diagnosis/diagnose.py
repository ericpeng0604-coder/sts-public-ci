"""Mechanical failure diagnosis only; no model or calibration changes."""
import collections,json,math,pathlib,sys,time
ROOT=pathlib.Path(__file__).parent;PILOT=ROOT.parent/'sts1-pilot8';TRAIN=ROOT.parent/'sts1-next-train'
sys.path.insert(0,str(PILOT))

def historical(source,ordinal,kind,index,floor,descs,scores):
    from replay_v4 import recorded_index
    try:return recorded_index(source,ordinal,kind,index,floor,descs,scores)
    except RuntimeError:
        expected=source['journal'][ordinal] if ordinal<len(source['journal']) else None
        if expected and kind=='shop' and expected['kind']==kind and expected['floor']==floor:
            old=expected['index']
            if 0<=old<len(descs) and 0<=index<len(descs) and descs[old]==descs[index] and all(math.isfinite(scores[i]) for i in (old,index)) and abs(scores[old]-scores[index])<=1e-5:
                return old,{'seed':str(source['seed']),'ordinal':ordinal,'kind':kind,'actual':index,'recorded':old,'descriptor_equal':True,'score_gap':abs(scores[old]-scores[index])}
        raise

def finish_trace(gc,sts,policy):
    from roguelike_ai.sts1_phase3.simulator import _set_pauses
    from collect_v3 import battle
    agent=sts.Agent();_set_pauses(agent);history=collections.deque(maxlen=25)
    for step in range(600):
        if gc.outcome!=sts.GameOutcome.UNDECIDED:return None
        agent.playout(gc)
        if gc.outcome!=sts.GameOutcome.UNDECIDED:return None
        if gc.screen_state==sts.ScreenState.BATTLE:
            battle(gc,sts);continue
        kind,index,descs,execs,scores=policy.decide(gc,sts)
        before=policy.module.obs_vec(gc)
        if index<0:gc.skip_reward_cards()
        else:execs[index](gc)
        after=policy.module.obs_vec(gc)
        history.append({'step':step,'kind':kind,'index':index,'floor':int(gc.floor_num),'screen':str(gc.screen_state),'gold':int(gc.gold),'hp':int(gc.cur_hp),'event':str(gc.cur_event),'same_observation':before==after,'observation':after,'selected_descriptor':descs[index] if index>=0 else None})
    return {'reason':'game_bound','floor':int(gc.floor_num),'screen':str(gc.screen_state),'event':str(gc.cur_event),'history':list(history)}

def main():
    import torch;torch.set_num_threads(1)
    import slaythespire as sts
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_set_pauses
    from collect_v3 import battle
    from collect_v4 import full_state,G7
    from worlds_v4 import world_seed
    from state_v3 import action_names
    prepared=ROOT/'prepared';failed=ROOT/'failed';groups=json.loads((prepared/'tasks.json').read_text())
    completed={}
    for p in failed.rglob('labels/*.json'):
        row=json.loads(p.read_text());completed[row['id']]=max(completed.get(row['id'],0),len(row['world_results']))
    pending=[]
    for shard in (1,2):
        for task in groups[shard]:
            ids=[i for i in task['ids'] if completed.get(i,0)<task['worlds']]
            if ids:pending.append({**task,'ids':ids,'shard':shard})
    report={'mechanical_only':True,'pending_sources':len(pending),'pending_cases':sum(len(t['ids']) for t in pending),'sources':[]};start=time.time()
    print(json.dumps({'pending_sources':len(pending),'pending_cases':report['pending_cases']}),flush=True)
    for task in pending:
        path=(PILOT/'input-scouts' if task['source']=='pilot' else prepared/'scouts')/task['file'];source=json.loads(path.read_text())
        targets={r['provenance']['noncombat_ordinal']:r for r in source['rows'] if r['id'] in task['ids']}
        policy=ArmGNoncombatPolicy(root=PILOT/'armg',weight_path=G7);gc=sts.GameContext(sts.CharacterClass.IRONCLAD,source['seed'],0);agent=sts.Agent();_set_pauses(agent)
        ordinal=0;checked=0;ties=[];bounds=[];tested_worlds=False
        for step in range(600):
            if gc.outcome!=sts.GameOutcome.UNDECIDED:break
            agent.playout(gc)
            if gc.outcome!=sts.GameOutcome.UNDECIDED:break
            if gc.screen_state==sts.ScreenState.BATTLE:battle(gc,sts);continue
            kind,index,descs,execs,scores=policy.decide(gc,sts);index,tie=historical(source,ordinal,kind,index,int(gc.floor_num),descs,scores)
            if tie:ties.append(tie)
            if ordinal in targets:
                row=targets[ordinal];assert full_state(gc,policy,descs)==row['state'] and action_names(gc,policy,kind,descs)==row['actions']
                assert index==row['g7_index'] and policy.module.obs_vec(gc)==row['armg_observation'] and descs==row['armg_descriptors'];checked+=1
                # Only the bound-failure shard requires paired continuation diagnosis.
                if task['shard']==2 and not tested_worlds:
                    for windex in range(task['worlds']):
                        seed=world_seed(row['id'],windex);base=gc.sample_public_future(seed)
                        for option in range(len(row['actions'])):
                            branch=base.clone();k,ds,ex=policy.choices(branch);ex[option](branch)
                            bound=finish_trace(branch,sts,policy)
                            if bound:bounds.append({'id':row['id'],'world_index':windex,'world_seed':str(seed),'option':option,**bound})
                        if bounds:break
                    tested_worlds=True
            ordinal+=1
            if index<0:gc.skip_reward_cards()
            else:execs[index](gc)
        assert checked==len(targets)
        record={'seed':str(source['seed']),'shard':task['shard'],'replayed_states':checked,'ties':ties,'bounds':bounds};report['sources'].append(record)
        print(json.dumps({'seed':record['seed'],'replayed_states':checked,'ties':len(ties),'bounds':len(bounds),'bound_locations':[{'id':b['id'],'world_index':b['world_index'],'option':b['option'],'floor':b['floor'],'screen':b['screen'],'event':b['event'],'last_choices':[{'kind':h['kind'],'index':h['index'],'same_observation':h['same_observation']} for h in b['history'][-5:]]} for b in bounds]}),flush=True)
    report['seconds']=time.time()-start;(ROOT/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
