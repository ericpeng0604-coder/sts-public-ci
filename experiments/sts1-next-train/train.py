"""Frozen 2-model/12-gate calibration on a new source-run cohort."""
import argparse,collections,json,math,pathlib,random,sys
from paths import ROOT,PILOT,NEXT,OUT,save,digest
from features import inputs,build_net,predict

def lower(selected):
    groups=collections.defaultdict(list)
    for row,index in selected:groups[str(row['provenance']['seed'])].append(row['estimates'][index]['win_advantage'])
    values=[sum(v)/len(v) for v in groups.values()]
    if len(values)<15 or len(selected)<50:return {'lower':None,'runs':len(values),'decisions':len(selected)}
    rng=random.Random(2026101301);bootstrap=6000
    distribution=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(bootstrap))
    return {'lower':distribution[int(bootstrap*.05/12)],'mean':sum(values)/len(values),'runs':len(values),'decisions':len(selected),'method':'source-run cluster bootstrap; 12 preregistered model/gate combinations; approximate-world evidence'}

def main(downloads):
    import torch;torch.set_num_threads(1)
    config=json.loads((ROOT/'protocol.json').read_text())
    initial={p.stem:json.loads(p.read_text()) for p in (OUT/'labels').glob('*.json')};assert len(initial)==240
    summaries=[json.loads(p.read_text()) for p in downloads.rglob('shard-summary.json')];assert len(summaries)==8 and {s['shard'] for s in summaries}==set(range(8)) and all(s['status']=='PASS' for s in summaries)
    replacements={}
    for path in downloads.rglob('labels/*.json'):
        row=json.loads(path.read_text());assert row['id'] not in replacements;replacements[row['id']]=row
    expected=json.loads((OUT/'new-state-ids.json').read_text());new_ids=set(expected['train'])|set(expected['calibration'])
    assert new_ids<=set(replacements)
    combined={**initial,**replacements};assert len(combined)==840
    references=dict(initial)
    for path in (OUT/'scouts').glob('*.json'):
        for row in json.loads(path.read_text())['rows']:references[row['id']]=row
    assert set(combined)==set(references)
    from worlds_v4 import world_seed,aggregate
    for ident,row in combined.items():
        for key in ('state','actions','g7_index','armg_observation','armg_descriptors','provenance'):assert row[key]==references[ident][key]
        assert row['world_seeds']==[world_seed(ident,i) for i in range(len(row['world_results']))]
        recalculated=aggregate(row['world_results'],row['g7_index'])
        for actual,recomputed in zip(row['estimates'],recalculated):
            for key in ('win_advantage','paired_variance','standard_error'):assert actual[key]==recomputed[key]
    training=[combined[i] for i in sorted(set(initial)|set(expected['train']))];calibration=[combined[i] for i in sorted(expected['calibration'])]
    assert len(training)==640 and len(calibration)==200
    assert not {r['provenance']['seed'] for r in training}&{r['provenance']['seed'] for r in calibration}
    reservations=set(json.loads((NEXT/'cohort-manifest.json').read_text())['cohorts']['evaluation_reserved'])
    for row in combined.values():
        assert row['teacher_combat_mcts']==2000 and len(row['world_results']) in (8,16,32)
        assert all(len(w)==len(row['actions']) for w in row['world_results'])
        assert not set(map(str,row['world_seeds']))&reservations
        if row['id'] in set(expected['calibration']):assert len(row['world_results'])==16 and row['training_cohort']=='calibration'
    prior=json.loads((PILOT/'human-prior.json').read_text());base=torch.load(PILOT/'g7.pt',weights_only=True,map_location='cpu')
    reports=[];payloads={};noise_reports={}
    for architecture in config['architectures']:
        torch.manual_seed(2026101301);xs=[];ys=[];variances=[]
        for row in training:
            xs.extend(inputs(row,prior,architecture));ys.extend(e['win_advantage'] for e in row['estimates'])
            n=len(row['world_results']);variances.extend(0. if e['index']==row['g7_index'] else min(1.,max(e['paired_variance'],1/(4*n))) for e in row['estimates'])
        x=torch.tensor(xs,dtype=torch.float32);y=torch.tensor(ys,dtype=torch.float32);noise=torch.tensor(variances,dtype=torch.float32)
        net=build_net(x.shape[1],base,architecture);optimizer=torch.optim.Adam(net.parameters(),lr=.0003,weight_decay=.0001)
        for epoch in range(40):
            for batch in torch.randperm(len(x)).split(128):
                mean,variance=predict(net,x[batch],architecture);loss=torch.nn.functional.mse_loss(mean,y[batch])
                if variance is not None:loss=loss+.1*torch.nn.functional.mse_loss(variance/.25,noise[batch]/.25)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
        net.eval();cal=[]
        with torch.no_grad():
            scores={};noise_errors=[];estimated_se=[]
            for row in calibration:
                mean,variance=predict(net,torch.tensor(inputs(row,prior,architecture),dtype=torch.float32),architecture);scores[row['id']]=mean
                if variance is not None:
                    for a,estimate in enumerate(row['estimates']):
                        value=float(variance[a]);noise_errors.append((value-estimate['paired_variance'])**2);estimated_se.append(math.sqrt(value/len(row['world_results'])))
            noise_reports[architecture]={'outcome_variance_rmse':math.sqrt(sum(noise_errors)/len(noise_errors)),'mean_predicted_sampling_se_at16worlds':sum(estimated_se)/len(estimated_se),'not_a_confidence_bound':True} if noise_errors else None
            for threshold in config['thresholds']:
                selected=[]
                for row in calibration:
                    mean=scores[row['id']];parent=row['g7_index'];index=int(mean.argmax());gain=float(mean[index]-mean[parent])
                    if index!=parent and gain>=threshold:selected.append((row,index))
                cal.append({'architecture':architecture,'threshold':threshold,**lower(selected)})
        payload={'schema':'armg-nonlinear-relative-advantage-v5','architecture':architecture,'state_dict':net.state_dict(),'input_dim':x.shape[1],'prior':prior,'gate':None,'combat_mcts':2000,'g7_sha256':digest(PILOT/'g7.pt'),'uncertainty_head':'paired-rollout outcome variance with finite-sample floor; not epistemic certainty or a confidence bound' if architecture=='armg_v5_card_identity' else None}
        payloads[architecture]=payload;torch.save(payload,OUT/(architecture+'.pt'));reports.extend(cal)
    good=[r for r in reports if r['lower'] is not None and r['lower']>0]
    chosen=max(good,key=lambda r:(r['lower'],-r['threshold'])) if good else None
    if chosen:
        payload=payloads[chosen['architecture']];payload['gate']=chosen['threshold'];torch.save(payload,OUT/'candidate.pt')
    result={'status':'PASS','training_cases':640,'fresh_calibration_cases':200,'epochs':40,'architectures':config['architectures'],'calibration':reports,'noise_prediction_diagnostics':noise_reports,'selected':chosen,'verdict':'FREEZE_FOR_FRESH_EVAL' if chosen else 'HOLD_G7_NO_CALIBRATED_ADVANTAGE','production_replaced':False,'formal_evaluation_pairs_if_eligible':config['evaluation_pairs'],'calibration_consumed':True,'no_more_tuning_on_this_calibration_cohort':True}
    save('training-report.json',result)
    save('model-freeze.json',{'candidate_sha256':digest(OUT/'candidate.pt') if chosen else None,'source_sha256':config['file_sha256'],'protocol_sha256':digest(ROOT/'protocol.json'),'g7_sha256':digest(PILOT/'g7.pt'),'calibration_gate_frozen':chosen,'evaluation_pairs':config['evaluation_pairs']})
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--downloads',type=pathlib.Path,required=True);args=parser.parse_args();main(args.downloads)
