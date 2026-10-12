"""Portable, hash-pinned public GitHub Actions pilot orchestration."""
import argparse,collections,concurrent.futures,hashlib,json,multiprocessing,pathlib,time
ROOT=pathlib.Path(__file__).parent;DATA=ROOT/'data'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    m=json.loads((ROOT/'package-manifest.json').read_text())
    for name,h in m['file_sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('package_hash_mismatch:'+name)
    DATA.mkdir(exist_ok=True)
    (DATA/'scouts').mkdir(exist_ok=True)
    rows=[]
    for src in sorted((ROOT/'input-scouts').glob('*.json')):
        source=json.loads(src.read_text());rows.extend(source['rows'])
        (DATA/'scouts'/src.name).write_text(src.read_text())
    (DATA/'public-states.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (DATA/'labels').mkdir(exist_ok=True)
    rows=[json.loads(l) for l in (DATA/'public-states.jsonl').read_text().splitlines()]
    assert len(rows)==240 and len({r['id'] for r in rows})==240
    assert collections.Counter(r['split'] for r in rows)=={'train':144,'validation':80,'development':16}
    return rows

def preflight():
    rows=prepare();import torch;torch.set_num_threads(1)
    import slaythespire as sts
    assert torch.__version__=='2.14.1+cpu'
    assert sha(ROOT/'g7.pt')==json.loads((ROOT/'package-manifest.json').read_text())['g7_sha256']
    from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy,_set_pauses
    from collect_v4 import full_state,G7
    from collect_v3 import battle
    from model_v4 import inputs,make_model
    from state_v3 import action_names
    policy=ArmGNoncombatPolicy(root=ROOT/'armg',weight_path=G7)
    selected=json.loads(next((DATA/'scouts').glob('*.json')).read_text());byordinal={r['provenance']['noncombat_ordinal']:r for r in selected['rows']}
    gc=sts.GameContext(sts.CharacterClass.IRONCLAD,selected['seed'],0);agent=sts.Agent();_set_pauses(agent);ordinal=0;checks=0
    for _ in range(600):
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        agent.playout(gc)
        if gc.outcome!=sts.GameOutcome.UNDECIDED:break
        if gc.screen_state==sts.ScreenState.BATTLE:battle(gc,sts);continue
        kind,index,descs,execs,scores=policy.decide(gc,sts)
        assert {'kind':kind,'index':index,'floor':int(gc.floor_num)}==selected['journal'][ordinal]
        if ordinal in byordinal:
            row=byordinal[ordinal];assert full_state(gc,policy,descs)==row['state']
            for seed in range(8):
                w=gc.sample_public_future(9174211362+seed);k,ds,_=policy.choices(w)
                assert full_state(w,policy,ds)==row['state'];assert action_names(w,policy,k,ds)==row['actions'];checks+=1
            r={**row,'g7_scores':list(scores)};x=torch.tensor(inputs(r,json.loads((ROOT/'human-prior.json').read_text())),dtype=torch.float32)
            net=make_model(x.shape[1],torch.load(G7,weights_only=True,map_location='cpu'));assert bool(torch.isfinite(net(x)).all())
        ordinal+=1
        if index<0:gc.skip_reward_cards()
        else:execs[index](gc)
    else:raise RuntimeError('preflight_replay_bound')
    assert checks==8*len(selected['rows'])
    result={'status':'PASS','replayed_seed':selected['seed'],'public_world_checks':checks,'torch':torch.__version__,'native_sha256':sha(ROOT/'native/slaythespire.cpython-312-x86_64-linux-gnu.so'),'scope':'platform/replay compatibility, not proof of win-rate gain'}
    (DATA/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def shard(index):
    rows=prepare();from pilot_jobs_v4 import label_run
    groups=json.loads((ROOT/'shards.json').read_text())['scout_files'];assert 0<=index<len(groups)
    files=[DATA/'scouts'/name for name in groups[index]]
    wanted={r['id'] for p in files for r in json.loads(p.read_text())['rows']}
    # Remove reused labels assigned to other shards before uploading evidence.
    for p in (DATA/'labels').glob('*.json'):
        if p.stem not in wanted:p.unlink()
    start=time.time();done=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for count in pool.map(label_run,map(str,files)):
            done+=1;print(json.dumps({'shard':index,'runs_completed':done,'labels':len(list((DATA/'labels').glob('*.json'))),'seconds':time.time()-start}),flush=True)
    actual={p.stem for p in (DATA/'labels').glob('*.json')};assert actual==wanted
    result={'status':'PASS','shard':index,'labels':len(actual),'worlds_per_option':8,'combat_mcts':2000,'elapsed_seconds':time.time()-start}
    (DATA/'shard-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def aggregate(downloads):
    rows=prepare();wanted={r['id']:r for r in rows};actual={};summaries=list(downloads.rglob('shard-summary.json'))
    assert len(summaries)==8,'all eight shard summaries required'
    assert {json.loads(p.read_text())['shard'] for p in summaries}==set(range(8))
    for p in downloads.rglob('labels/*.json'):
        r=json.loads(p.read_text());ident=r['id']
        if ident in actual:raise RuntimeError('duplicate_shard_case:'+ident)
        assert ident in wanted;original=wanted[ident]
        for k in ['state','actions','split','g7_index']:assert r[k]==original[k]
        assert len(r['world_results'])==8 and r['teacher_combat_mcts']==2000
        assert all(len(w)==len(r['actions']) for w in r['world_results'])
        actual[ident]=r
    assert set(actual)==set(wanted),'missing pilot labels'
    for ident,r in actual.items():(DATA/'labels'/(ident+'.json')).write_text(json.dumps(r))
    result={'status':'PASS','labels':len(actual),'shards':8,'worlds_per_option':8,'all_states_and_groups_match':True,'production_replaced':False}
    (DATA/'aggregate-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','shard','aggregate']);p.add_argument('--index',type=int);p.add_argument('--downloads',type=pathlib.Path);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='preflight':preflight()
    elif a.stage=='shard':shard(a.index)
    elif a.stage=='aggregate':aggregate(a.downloads)
