"""Validate diagnostic artifacts and publish independent next-stage state lists."""
import argparse,collections,hashlib,json,pathlib
from common import ROOT,PILOT,OUT,save,band,select_states

def main(downloads):
    manifest=json.loads((ROOT/'cohort-manifest.json').read_text());cohorts=manifest['cohorts']
    sets={k:set(v) for k,v in cohorts.items()}
    assert all(not sets[a]&sets[b] for a in sets for b in sets if a!=b)
    old_sources={p.stem for p in (PILOT/'input-scouts').glob('*.json')}
    assert not old_sources&(sets['train']|sets['calibration']|sets['evaluation_reserved'])
    summaries=list(downloads.rglob('collection-summary.json'));assert len(summaries)==8
    summary_data=[json.loads(p.read_text()) for p in summaries]
    assert {r['shard'] for r in summary_data}==set(range(8)) and all(r['status']=='PASS' for r in summary_data)
    sources={};pools=collections.defaultdict(list)
    for path in downloads.rglob('scouts/*.json'):
        source=json.loads(path.read_text());seed=str(source['seed']);split=source['split']
        assert seed not in sources and split in ('train','calibration') and seed in sets[split]
        assert source['combat_mcts']==2000 and seed not in sets['evaluation_reserved']
        for row in source['rows']:
            assert row['split']==split and str(row['provenance']['seed'])==seed and row['label_status']=='UNLABELLED'
            assert len(row['actions'])==len(row['armg_descriptors']) and 0<=row['g7_index']<len(row['actions'])
        sources[seed]=source;pools[split].extend(source['rows'])
    assert set(sources)==sets['train']|sets['calibration']
    old_ids={r['id'] for p in (PILOT/'input-scouts').glob('*.json') for r in json.loads(p.read_text())['rows']}
    # Identical states are removed from calibration rather than shared across splits.
    train_pool=[r for r in pools['train'] if r['id'] not in old_ids]
    train=select_states(train_pool,manifest['targets']['train_states'],True)
    blocked=old_ids|{r['id'] for r in pools['train']}
    calibration=select_states([r for r in pools['calibration'] if r['id'] not in blocked],manifest['targets']['calibration_states'],False)
    assert len(train)==400 and len(calibration)==200,'Insufficient coverage: report shortfall rather than reusing consumed runs.'
    assert len({r['provenance']['seed'] for r in calibration})>=15
    selected_by_seed=collections.defaultdict(list)
    for row in train+calibration:selected_by_seed[str(row['provenance']['seed'])].append(row)
    for seed,rows in selected_by_seed.items():save('selected-scouts/'+seed+'.json',{**sources[seed],'rows':rows})
    for split,rows in [('train',train),('calibration',calibration)]:save(split+'-state-list.json',[{'id':r['id'],'seed':str(r['provenance']['seed']),'ordinal':r['provenance']['noncombat_ordinal'],'act':r['state']['act'],'hp_band':band(r),'status':'UNLABELLED'} for r in rows])
    save('evaluation-reservation.json',{'seeds':cohorts['evaluation_reserved'],'executed':False,'sample_size':'not yet chosen; preregister after power analysis before frozen candidate evaluation'})
    refinements=[json.loads(p.read_text()) for p in downloads.rglob('refinement-summary.json')]
    assert len(refinements)==8 and {r['shard'] for r in refinements}==set(range(8))
    assert all(r['status']=='PASS' and r['combat_mcts']==2000 for r in refinements)
    refined=[json.loads(p.read_text()) for p in downloads.rglob('refined-labels/*.json')]
    assert len(refined)==8 and len({r['id'] for r in refined})==8
    for row in refined:
        original=json.loads(next(downloads.rglob('original-labels/'+row['id']+'.json')).read_text())
        assert row['world_results'][:8]==original['world_results'] and row['world_seeds'][:8]==original['world_seeds']
        assert len(row['world_results']) in (16,32) and row['all_options_share_worlds']
        assert all(len(w)==len(row['actions']) for w in row['world_results'])
    refinement_report={'states':8,'old_vs_new':[{'id':r['id'],'worlds':len(r['world_results']),'old_best_gain':max(e['win_advantage'] for e in json.loads(next(downloads.rglob('original-labels/'+r['id']+'.json')).read_text())['estimates']),'new_best_gain':max(e['win_advantage'] for e in r['estimates']),'uncertain':r['uncertain']} for r in refined],'new_rollouts':sum(r['new_rollouts'] for r in refinements),'slowest_shard_seconds':max(r['seconds'] for r in refinements),'estimated_64_case_additional_rollouts':8*sum(r['new_rollouts'] for r in refinements),'limits':'Extrapolation from eight selected cases; not a guaranteed runtime or independent win-rate validation.'}
    save('refinement-cost-report.json',refinement_report)
    diagnosis=json.loads(next(downloads.rglob('diagnosis.json')).read_text())
    summary={'status':'PASS','diagnosed_old_states':diagnosis['states'],'new_source_runs':len(sources),'new_captured_states':sum(len(s['rows']) for s in sources.values()),'new_train_states':len(train),'new_calibration_states':len(calibration),'train_source_runs':len({r['provenance']['seed'] for r in train}),'calibration_source_runs':len({r['provenance']['seed'] for r in calibration}),'train_cells':dict(collections.Counter(f"act{r['state']['act']}-{band(r)}" for r in train)),'calibration_cells':dict(collections.Counter(f"act{r['state']['act']}-{band(r)}" for r in calibration)),'refined_states':8,'reserved_evaluation_seeds':2000,'evaluation_executed':False,'new_model_trained':False,'production_replaced':False,'next_action':'Use cost and feature diagnosis to freeze the next labeling/training protocol; calibrate once on untouched source-run cohort.'}
    save('stage1-summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--downloads',type=pathlib.Path,required=True);args=parser.parse_args();main(args.downloads)
