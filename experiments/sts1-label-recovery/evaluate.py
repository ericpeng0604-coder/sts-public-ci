import argparse,concurrent.futures,json,math,multiprocessing,pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).parent.parent/'sts1-next-train'))
from paths import ROOT,PILOT,NEXT,OUT,save,digest
sys.path.insert(0,str(PILOT))

def pair(seed_text):
    import torch;torch.set_num_threads(1)
    import slaythespire as sts
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,run_simulator_game
    from runtime import AdvantagePolicy
    seed=int(seed_text);runs={}
    for arm in ('g7','candidate'):
        kwargs={'root':PILOT/'armg','weight_path':PILOT/'g7.pt'}
        policy=ArmGNoncombatPolicy(**kwargs) if arm=='g7' else AdvantagePolicy(model_path=OUT/'candidate.pt',**kwargs)
        result=run_simulator_game(student=None,sts=sts,seed=seed,heldout_seeds=(seed,),armg_policy=policy,combat_mcts_sims=2000,max_game_steps=10000,max_battle_steps=1200,evidence_path=None)
        keep=['result','outcome','final_floor','combat_policy','student_action_count','illegal_action_count','crash_count','timeout_count','communication_error_count']
        runs[arm]={k:result[k] for k in keep};runs[arm]['override_count']=len(getattr(policy,'overrides',[]))
    return {'seed':seed_text,'runs':runs}

def shard(index):
    report=json.loads((OUT/'training-report.json').read_text());assert report['selected'] is not None
    frozen=json.loads((OUT/'model-freeze.json').read_text());assert frozen['mechanical_amendment_sha256']==digest(pathlib.Path(__file__).parent/'amendment.json');assert digest(OUT/'candidate.pt')==frozen['candidate_sha256'] and digest(PILOT/'g7.pt')==frozen['g7_sha256']
    protocol=json.loads((ROOT/'protocol.json').read_text());assert protocol['evaluation_pairs']==1200
    reserved=json.loads((NEXT/'cohort-manifest.json').read_text())['cohorts']['evaluation_reserved'];seeds=reserved[:1200][index::8]
    start=time.time();rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(pair,seeds):rows.append(row)
    assert len(rows)==150 and [r['seed'] for r in rows]==seeds
    assert digest(OUT/'candidate.pt')==frozen['candidate_sha256'] and digest(PILOT/'g7.pt')==frozen['g7_sha256']
    save('evaluation-pairs-'+str(index)+'.json',rows);save('evaluation-summary-'+str(index)+'.json',{'shard':index,'pairs':len(rows),'seconds':time.time()-start,'frozen_sha256':frozen['candidate_sha256']})
    print(json.dumps({'shard':index,'pairs':len(rows),'seconds':time.time()-start}),flush=True)

def finalize(downloads):
    training=json.loads(next(downloads.rglob('training-report.json')).read_text())
    if training['selected'] is None:
        save('final-report.json',{'verdict':'HOLD_G7_NO_CALIBRATED_ADVANTAGE','formal_evaluation_executed':False,'production_replaced':False,'known_simulator_courier_pricing_bug':True,'real_game_performance_claim_supported':False,'training':training});print(json.dumps({'verdict':'HOLD_G7_NO_CALIBRATED_ADVANTAGE','formal_evaluation_executed':False}),flush=True);return
    frozen=json.loads(next(downloads.rglob('model-freeze.json')).read_text());rows=[]
    summaries=[json.loads(p.read_text()) for p in downloads.rglob('evaluation-summary-*.json')]
    assert len(summaries)==8 and {s['shard'] for s in summaries}==set(range(8)) and all(s['frozen_sha256']==frozen['candidate_sha256'] for s in summaries)
    for p in downloads.rglob('evaluation-pairs-*.json'):rows.extend(json.loads(p.read_text()))
    reserved=json.loads((NEXT/'cohort-manifest.json').read_text())['cohorts']['evaluation_reserved'][:1200]
    assert len(rows)==1200 and {r['seed'] for r in rows}==set(reserved)
    safety=['illegal_action_count','crash_count','timeout_count','communication_error_count']
    safe=all(result['result']=='PASS_SIMULATOR_COMPLETE_RUN' and all(result[k]==0 for k in safety) and result['combat_policy']=='mcts_2000' and result['student_action_count']==0 for row in rows for result in row['runs'].values())
    better=sum(r['runs']['candidate']['outcome']=='victory' and r['runs']['g7']['outcome']!='victory' for r in rows)
    worse=sum(r['runs']['g7']['outcome']=='victory' and r['runs']['candidate']['outcome']!='victory' for r in rows)
    n=better+worse;pvalue=sum(math.comb(n,k) for k in range(better,n+1))/2**n if n else 1.
    totals={arm:{'wins':sum(r['runs'][arm]['outcome']=='victory' for r in rows),'mean_floor':sum(r['runs'][arm]['final_floor'] for r in rows)/len(rows)} for arm in ('g7','candidate')}
    pass_gate=safe and better>worse and pvalue<=.05 and totals['candidate']['mean_floor']>=totals['g7']['mean_floor']
    report={'verdict':'FRESH_SIMULATOR_RESEARCH_GATE_PASS' if pass_gate else 'HOLD_G7','pairs':1200,'totals':totals,'candidate_only_wins':better,'parent_only_wins':worse,'net_wins':better-worse,'win_rate_delta':(better-worse)/1200,'one_sided_exact_p':pvalue,'safety_pass':safe,'formal_evaluation_executed':True,'production_replaced':False,'real_client_game_verified':False,'private_history_exclusion_proven':False,'candidate_sha256':frozen['candidate_sha256']}
    report['known_simulator_courier_pricing_bug']=True;report['real_game_performance_claim_supported']=False;report['mechanical_amendment']='recovery-v1-unchanged-native-shop-ties-10000-bound'
    save('final-report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['shard','finalize']);parser.add_argument('--shard',type=int);parser.add_argument('--downloads',type=pathlib.Path);args=parser.parse_args()
    if args.stage=='shard':shard(args.shard)
    else:finalize(args.downloads)
