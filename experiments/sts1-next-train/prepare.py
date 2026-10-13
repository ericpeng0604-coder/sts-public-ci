import argparse,collections,json,pathlib,shutil
from paths import ROOT,PILOT,NEXT,OUT,save

def main(downloads):
    summary=json.loads(next(downloads.rglob('stage1-summary.json')).read_text());assert summary['status']=='PASS' and summary['new_train_states']==400 and summary['new_calibration_states']==200
    sources=list(downloads.rglob('selected-scouts/*.json'));assert len(sources)==80
    old={json.loads(p.read_text())['id']:json.loads(p.read_text()) for p in downloads.rglob('original-labels/*.json')};assert len(old)==240
    for p in downloads.rglob('refined-labels/*.json'):
        r=json.loads(p.read_text());assert r['id'] in old;old[r['id']]=r
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'labels').mkdir(exist_ok=True);(OUT/'scouts').mkdir(exist_ok=True)
    for ident,row in old.items():save('labels/'+ident+'.json',{**row,'training_cohort':'consumed_pilot'})
    tasks=[];new_rows=[]
    for path in sources:
        source=json.loads(path.read_text());save('scouts/'+path.name,source);cohort=source['split']
        assert cohort in ('train','calibration')
        budget=8 if cohort=='train' else 16
        tasks.append({'source':'new','file':path.name,'ids':[r['id'] for r in source['rows']],'worlds':budget,'cohort':cohort,'cost':sum((58-r['state']['floor'])*len(r['actions'])*budget for r in source['rows'])})
        new_rows.extend(source['rows'])
    assert len(new_rows)==600 and len({r['id'] for r in new_rows})==600 and not set(old)&{r['id'] for r in new_rows}
    old_shortlist=json.loads(next(downloads.rglob('refinement-shortlist.json')).read_text())['case_ids']
    pending={i for i in old_shortlist if len(old[i]['world_results'])<32};assert len(old_shortlist)==64
    for path in (PILOT/'input-scouts').glob('*.json'):
        source=json.loads(path.read_text());rows=[r for r in source['rows'] if r['id'] in pending]
        if rows:tasks.append({'source':'pilot','file':path.name,'ids':[r['id'] for r in rows],'worlds':32,'cohort':'consumed_pilot','cost':sum((58-r['state']['floor'])*len(r['actions'])*24 for r in rows)})
    groups=[[] for _ in range(8)];costs=[0]*8
    for task in sorted(tasks,key=lambda t:(-t['cost'],t['file'])):
        index=min(range(8),key=lambda i:costs[i]);groups[index].append(task);costs[index]+=task['cost']
    assert sum(len(t['ids']) for g in groups for t in g)==600+len(pending)
    save('tasks.json',groups);save('new-state-ids.json',{'train':[r['id'] for r in new_rows if r['split']=='train'],'calibration':[r['id'] for r in new_rows if r['split']=='calibration']})
    save('prepare-summary.json',{'status':'PASS','old_training_cases':240,'new_training_cases':400,'fresh_calibration_cases':200,'old_cases_to_refine':len(pending),'train_worlds':8,'calibration_worlds':16,'old_refinement_worlds':32,'shard_cost_units':costs})
    print(json.dumps(json.loads((OUT/'prepare-summary.json').read_text())),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--downloads',type=pathlib.Path,required=True);args=parser.parse_args();main(args.downloads)
