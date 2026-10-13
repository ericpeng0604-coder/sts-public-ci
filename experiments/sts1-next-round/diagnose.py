"""Diagnose the actual public pilot, then freeze a bounded refinement shortlist."""
import argparse,collections,hashlib,json,pathlib,sys
from common import ROOT,PILOT,OUT,save,band,stable
sys.path.insert(0,str(PILOT))

def main(downloads):
    import torch;torch.set_num_threads(1)
    from model_v4 import inputs,make_model
    originals={r['id']:r for p in (PILOT/'input-scouts').glob('*.json') for r in json.loads(p.read_text())['rows']}
    rows=[];seen=set()
    for path in downloads.rglob('labels/*.json'):
        row=json.loads(path.read_text())
        if row['id'] in seen:raise RuntimeError('duplicate_label:'+row['id'])
        original=originals[row['id']]
        for key in ('state','actions','split','g7_index','armg_observation','armg_descriptors'):assert row[key]==original[key]
        assert len(row['world_results'])==8 and row['teacher_combat_mcts']==2000
        seen.add(row['id']);rows.append(row)
    assert seen==set(originals) and len(rows)==240
    paths=list(downloads.rglob('candidate.pt'));assert len(paths)==1
    payload=torch.load(paths[0],weights_only=True,map_location='cpu')
    assert payload['gate'] is None and payload['combat_mcts']==2000
    net=make_model(payload['input_dim']);net.load_state_dict(payload['state_dict']);net.eval()
    cases=[];groups=collections.defaultdict(list);saturation=score_count=collisions=all_defeats=flat_wins=0;collision_examples=[]
    for row in sorted(rows,key=lambda r:r['id']):
        encoded=inputs(row,payload['prior'])
        with torch.no_grad():scores=net(torch.tensor(encoded,dtype=torch.float32)).squeeze(-1).tolist()
        parent=row['g7_index'];candidate=max(range(len(scores)),key=lambda i:scores[i]);gap=scores[candidate]-scores[parent]
        wins=[sum(w[a]['outcome']=='victory' for w in row['world_results']) for a in range(len(row['actions']))]
        all_defeats+=int(max(wins)==0);flat_wins+=int(len(set(wins))==1)
        for score in row['g7_scores']:score_count+=1;saturation+=int(abs(score)>=10)
        for a in range(len(encoded)):
            for b in range(a):
                if encoded[a]==encoded[b] and row['actions'][a]!=row['actions'][b]:
                    collisions+=1;collision_examples.append({'id':row['id'],'actions':[row['actions'][a],row['actions'][b]],'descriptors_equal':row['armg_descriptors'][a]==row['armg_descriptors'][b]})
        case={'id':row['id'],'seed':str(row['provenance']['seed']),'act':row['state']['act'],'hp_band':band(row),'split':row['split'],'parent':parent,'candidate':candidate,'predicted_gain':gap,'estimated_gain':row['estimates'][candidate]['win_advantage'],'standard_error':row['estimates'][candidate]['standard_error'],'all_options_same_win_count':len(set(wins))==1,'all_options_defeat':max(wins)==0,'actions':row['actions'],'priority':int(candidate!=parent)*4+int(row['state']['act']==3)+int(band(row)=='low')+int(len(set(wins))==1)}
        cases.append(case);groups[(case['act'],case['hp_band'])].append(case)
    # All old splits are now diagnostic/training material, never fresh calibration.
    ranked=sorted(cases,key=lambda c:(-c['priority'],-c['predicted_gain'],c['id']))
    selected=[];run_counts=collections.Counter()
    for case in ranked:
        if run_counts[case['seed']]>=2:continue
        selected.append(case);run_counts[case['seed']]+=1
        if len(selected)==64:break
    assert len(selected)==64
    pilot=[];cells=set();pilot_seeds=set()
    for case in ranked:
        cell=(case['act'],case['hp_band'])
        if cell in cells or case['seed'] in pilot_seeds:continue
        pilot.append(case);cells.add(cell);pilot_seeds.add(case['seed'])
        if len(pilot)==8:break
    for case in ranked:
        if len(pilot)==8:break
        if case['seed'] not in pilot_seeds:pilot.append(case);pilot_seeds.add(case['seed'])
    assert len(pilot)==8
    save('refinement-shortlist.json',{'case_ids':[c['id'] for c in selected],'pilot_case_ids':[c['id'] for c in pilot],'use':'diagnosis/training only; old validation is not fresh'})
    for index in range(8):save(f'refinement-shards/{index}.json',[pilot[index]['id']])
    report={'states':240,'legal_options':sum(len(r['actions']) for r in rows),'old_worlds_per_option':8,'candidate_disagreements':sum(c['candidate']!=c['parent'] for c in cases),'all_options_defeat_cases':all_defeats,'all_options_same_win_count_cases':flat_wins,'clipped_g7_score_values':saturation,'g7_score_values':score_count,'distinct_action_feature_collisions':collisions,'old_calibration_is_consumed':True,'groups':{f'act{act}-{hp}':{'states':len(v),'disagreements':sum(c['candidate']!=c['parent'] for c in v),'mean_estimated_gain_on_disagreement':sum(c['estimated_gain'] for c in v if c['candidate']!=c['parent'])/max(1,sum(c['candidate']!=c['parent'] for c in v))} for (act,hp),v in sorted(groups.items())},'limits':['8-world means are noisy','zero observed paired variance is not proof of certainty','large distribution-free intervals cannot resolve small gains at 8-32 worlds','these cases may guide training but cannot validate a retuned candidate']}
    report['candidate_sha256']=__import__('hashlib').sha256(paths[0].read_bytes()).hexdigest()
    report['feature_collision_examples']=collision_examples
    save('diagnosis.json',report);save('case-diagnostics.json',cases)
    saved=OUT/'original-labels';saved.mkdir(parents=True,exist_ok=True)
    for row in rows:(saved/(row['id']+'.json')).write_text(json.dumps(row))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--downloads',type=pathlib.Path,required=True);args=parser.parse_args();main(args.downloads)
