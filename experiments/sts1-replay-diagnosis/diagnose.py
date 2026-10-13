"""Replay historical actions for diagnosis only; never produce training labels."""
import concurrent.futures,json,multiprocessing,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]/'sts1-pilot8'
sys.path.insert(0,str(ROOT))

def check(name):
    import torch;torch.set_num_threads(1)
    import slaythespire as sts
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_set_pauses
    from collect_v3 import battle
    from collect_v4 import full_state,G7
    from state_v3 import action_names
    source=json.loads((ROOT/'input-scouts'/name).read_text())
    policy=ArmGNoncombatPolicy(root=ROOT/'armg',weight_path=G7)
    gc=sts.GameContext(sts.CharacterClass.IRONCLAD,source['seed'],0)
    agent=sts.Agent();_set_pauses(agent)
    targets={r['provenance']['noncombat_ordinal']:r for r in source['rows']}
    ordinal=0;checked=0;differences=[]
    for step in range(600):
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        agent.playout(gc)
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        if gc.screen_state==sts.ScreenState.BATTLE:battle(gc,sts);continue
        kind,index,descs,execs,scores=policy.decide(gc,sts)
        actual={'kind':kind,'index':index,'floor':int(gc.floor_num)}
        expected=source['journal'][ordinal] if ordinal<len(source['journal']) else None
        if actual!=expected:
            differences.append({'ordinal':ordinal,'actual':actual,'expected':expected,'descriptors':descs,'scores':scores})
            if expected is None or kind!=expected['kind'] or actual['floor']!=expected['floor']:
                return {'seed':source['seed'],'status':'PATH_MISMATCH','differences':differences,'checked':checked}
        if ordinal in targets:
            row=targets[ordinal];state=full_state(gc,policy,descs)
            state_diff={k:{'actual':state.get(k),'expected':row['state'].get(k)} for k in set(state)|set(row['state']) if state.get(k)!=row['state'].get(k)}
            if state_diff or action_names(gc,policy,kind,descs)!=row['actions']:
                return {'seed':source['seed'],'status':'STATE_MISMATCH','ordinal':ordinal,'differences':differences,'state_diff':state_diff,'checked':checked}
            checked+=1
        recorded=expected['index']
        if recorded==-1 and kind=='reward_empty':gc.skip_reward_cards()
        elif 0<=recorded<len(execs):execs[recorded](gc)
        else:return {'seed':source['seed'],'status':'ILLEGAL_RECORDED_ACTION','ordinal':ordinal}
        ordinal+=1
    else:return {'seed':source['seed'],'status':'BOUND'}
    return {'seed':source['seed'],'status':'PASS' if checked==len(targets) and ordinal==len(source['journal']) else 'INCOMPLETE','checked':checked,'expected_states':len(targets),'differences':differences}

if __name__=='__main__':
    groups=json.loads((ROOT/'shards.json').read_text())['scout_files']
    files=[f for i in (5,7) for f in groups[i]];start=time.time();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(check,files):
            results.append(result);print(json.dumps(result),flush=True)
    output={'diagnostic_only':True,'seconds':time.time()-start,'runs':results}
    (ROOT/'data').mkdir(exist_ok=True)
    (ROOT/'data/replay-diagnosis.json').write_text(json.dumps(output,indent=2))
    if any(r['status']!='PASS' for r in results):sys.exit(1)
