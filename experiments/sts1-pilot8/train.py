"""Train original ArmG-sized nonlinear paired-win advantage predictor."""
import collections,hashlib,json,pathlib,random,math
from model_v4 import inputs,make_model
ROOT=pathlib.Path(__file__).parent;OUT=ROOT/'data'

def human_prior():
    return json.loads((ROOT/"human-prior.json").read_text())

def group_lower(rows,selected,bootstrap=3000):
    groups=collections.defaultdict(list)
    for row,a in selected:
        groups[row['provenance']['seed']].append(row['estimates'][a]['win_advantage'])
    values=[sum(v)/len(v) for v in groups.values()]
    if len(values)<15 or len(selected)<50:return {'lower':None,'runs':len(values),'decisions':len(selected)}
    rng=random.Random(2026101204);samples=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(bootstrap))
    return {'lower':samples[int(bootstrap*.05/6)],'mean':sum(values)/len(values),'runs':len(values),'decisions':len(selected),'method':'run-cluster bootstrap; Bonferroni 6 prereg thresholds; diagnostic approximate-world evidence'}

def main():
    import torch;torch.set_num_threads(1);torch.manual_seed(2026101204)
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'labels').glob('*.json'))]
    expected=sum(1 for _ in (OUT/'public-states.jsonl').open())
    if len(rows)!=expected:raise RuntimeError('all_collected_states_must_have_world_labels')
    prior=human_prior();parts={s:[r for r in rows if r['split']==s] for s in ['train','validation','development']}
    assert not ({r['provenance']['seed'] for r in parts['train']}&{r['provenance']['seed'] for r in parts['validation']})
    xs=[];ys=[]
    for r in parts['train']:
        xs.extend(inputs(r,prior));ys.extend(e['win_advantage'] for e in r['estimates'])
    x=torch.tensor(xs,dtype=torch.float32);y=torch.tensor(ys,dtype=torch.float32)
    base=torch.load(ROOT/'g7.pt',weights_only=True,map_location='cpu');net=make_model(x.shape[1],base)
    optimizer=torch.optim.Adam(net.parameters(),lr=.0003,weight_decay=.0001);curve=[]
    for epoch in range(40):
        order=torch.randperm(len(x));losses=[]
        for batch in order.split(128):
            pred=net(x[batch]).squeeze(-1);loss=torch.nn.functional.mse_loss(pred,y[batch]);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step();losses.append(float(loss))
        curve.append(sum(losses)/len(losses))
    net.eval();thresholds=[.02,.05,.10,.15,.25,.40];cal=[]
    with torch.no_grad():
        for threshold in thresholds:
            selected=[]
            for r in parts['validation']:
                scores=net(torch.tensor(inputs(r,prior),dtype=torch.float32)).squeeze(-1);parent=r['g7_index'];a=int(scores.argmax());delta=float(scores[a]-scores[parent])
                if a!=parent and delta>=threshold:selected.append((r,a))
            cal.append({'threshold':threshold,**group_lower(parts['validation'],selected)})
    good=[c for c in cal if c['lower'] is not None and c['lower']>0];gate=min(c['threshold'] for c in good) if good else None
    payload={'schema':'armg-nonlinear-card-advantage-v4','state_dict':net.state_dict(),'input_dim':x.shape[1],'prior':prior,'gate':gate,
             'combat_mcts':2000,'g7_sha256':hashlib.sha256((ROOT/'g7.pt').read_bytes()).hexdigest(),'sampler_limit':'approximate stochastic continuation; not exact seed posterior'}
    torch.save(payload,OUT/'candidate.pt')
    report={'counts':{k:len(v) for k,v in parts.items()},'epochs':40,'training_mse_curve':curve,'calibration':cal,'gate':gate,'verdict':'FREEZE_FOR_FRESH_RESEARCH_EVAL' if gate is not None else 'HOLD_G7_NO_CALIBRATED_ADVANTAGE','evaluation_run':False,'human_prior':'source-balanced training histories only; preference evidence, not terminal win labels','production_replaced':False}
    (OUT/'training-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
