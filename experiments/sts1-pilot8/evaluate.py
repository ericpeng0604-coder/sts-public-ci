"""Frozen, bounded 300-pair full-run simulator check; no adaptive retesting."""
import argparse,concurrent.futures,hashlib,json,math,multiprocessing,os,pathlib,time,uuid
ROOT=pathlib.Path(__file__).parent
REPO=ROOT.parents[1]
G7=ROOT/'g7.pt'
G7_SHA='8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run_pair(payload):
    seed,root,model,trace=payload
    import torch;torch.set_num_threads(1)
    import slaythespire as sts
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,run_simulator_game
    from runtime_v4 import AdvantagePolicy
    outputs={}
    for arm in ['g7','human_reward']:
        kwargs={'root':ROOT/'armg','weight_path':G7}
        p=ArmGNoncombatPolicy(**kwargs) if arm=='g7' else AdvantagePolicy(model_path=model,**kwargs)
        path=pathlib.Path(root)/'traces'/f'{seed}-{arm}.jsonl'
        r=run_simulator_game(student=None,sts=sts,seed=seed,heldout_seeds=(seed,),armg_policy=p,combat_mcts_sims=2000,
                             max_game_steps=600,max_battle_steps=1200,evidence_path=path if trace else None)
        r['arm']=arm;r['candidate_considered']=getattr(p,'considered',0)
        r['candidate_overrides']=getattr(p,'overrides',[]);r['candidate_abstentions']=getattr(p,'abstentions',{})
        r['public_state_coverage']=getattr(p,'state_coverage',{})
        r['candidate_public_states']=getattr(p,'decision_states',[])
        outputs[arm]=r
    return {'seed':seed,'runs':outputs}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pairs',type=int,default=300);args=parser.parse_args()
    model=ROOT/'data/candidate.pt';out=ROOT/'data/full-run';out.mkdir(exist_ok=False);(out/'traces').mkdir()
    assert sha(G7)==G7_SHA
    exclusion=set(json.loads((ROOT/'seed-exclusions.json').read_text()))
    for path in (ROOT/'data/labels').glob('*.json'):
        exclusion.update(json.loads(path.read_text()).get('world_seeds',[]))
    nonce=uuid.uuid4().hex;seeds=[]
    for i in range(args.pairs):
        seed=(int(hashlib.sha256((nonce+':'+str(i)).encode()).hexdigest()[:16],16)&((1<<62)-1))+(1<<40)
        assert seed not in exclusion and seed not in seeds;seeds.append(seed)
    code_hashes={p.name:sha(p) for p in [ROOT/'evaluate.py',ROOT/'runtime_v4.py',ROOT/'train.py',ROOT/'model_v4.py',ROOT/'collect_v4.py',ROOT/'state_v3.py',ROOT/'collect_v3.py']}
    manifest={'scope':'research_full_run_not_formal_promotion','created_unix':time.time(),'seeds':seeds,'pairs':args.pairs,
              'model_sha256':sha(model),'g7_sha256':G7_SHA,'code_hashes':code_hashes,
              'simulator_commit':'7476a81954020087da31d41d16fddf475746ec2d','runtime_binary_sha256':sha(ROOT/'native/slaythespire.cpython-312-x86_64-linux-gnu.so'),'mcts_sims':2000,'ascension':0,
              'candidate_scope':'calibration_eligible_noncombat_kinds','workers':2,'max_game_steps':600,'max_battle_steps':1200,
              'known_excluded_seed_count':len(exclusion),'private_exclusion_manifests':'UNAVAILABLE_NOT_PROVEN',
              'seed_method':'pre-run UUID nonce + SHA256, large 62-bit domain; known IDs excluded',
              'adoption_gate':'positive paired net, one-sided exact win sign p<=0.05, mean floor not lower, all complete and all safety counters zero',
              'no_retraining_after_results':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks=[(s,str(out),str(model),True) for s in seeds]
        for row in pool.map(run_pair,tasks):
            rows.append(row);(out/'pairs.jsonl').open('a').write(json.dumps(row)+'\n')
            print(json.dumps({'completed_pairs':len(rows),'parent_wins':sum(x['runs']['g7']['outcome']=='victory' for x in rows),
                              'candidate_wins':sum(x['runs']['human_reward']['outcome']=='victory' for x in rows)}),flush=True)
    assert sha(model)==manifest['model_sha256'] and sha(G7)==G7_SHA
    for p in [ROOT/'evaluate.py',ROOT/'runtime_v4.py',ROOT/'train.py',ROOT/'model_v4.py',ROOT/'collect_v4.py',ROOT/'state_v3.py',ROOT/'collect_v3.py']:assert sha(p)==code_hashes[p.name]
    safety=['illegal_action_count','crash_count','timeout_count','communication_error_count']
    safe=all(r['result']=='PASS_SIMULATOR_COMPLETE_RUN' and all(r[k]==0 for k in safety) and r['combat_policy']=='mcts_2000' and r['student_action_count']==0 for x in rows for r in x['runs'].values())
    better=sum(x['runs']['human_reward']['outcome']=='victory' and x['runs']['g7']['outcome']!='victory' for x in rows)
    worse=sum(x['runs']['g7']['outcome']=='victory' and x['runs']['human_reward']['outcome']!='victory' for x in rows)
    n=better+worse;p=sum(math.comb(n,k) for k in range(better,n+1))/2**n if n else 1.0
    totals={a:{'wins':sum(x['runs'][a]['outcome']=='victory' for x in rows),'mean_floor':sum(x['runs'][a]['final_floor'] for x in rows)/len(rows)} for a in ['g7','human_reward']}
    summary={'pairs':len(rows),'safety_pass':safe,'totals':totals,'candidate_only_wins':better,'parent_only_wins':worse,'net':better-worse,'one_sided_sign_p':p,
             'candidate_override_count':sum(len(x['runs']['human_reward']['candidate_overrides']) for x in rows),
             'seeds_with_overrides':sum(bool(x['runs']['human_reward']['candidate_overrides']) for x in rows),
             'model_and_code_freeze_pass':True,'verdict':'RESEARCH_GATE_PASS' if safe and better>worse and p<=0.05 and totals['human_reward']['mean_floor']>=totals['g7']['mean_floor'] else 'HOLD_G7',
             'production_replaced':False,'real_game':'NOT_RUN','private_seed_exclusion':'NOT_VERIFIED'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
