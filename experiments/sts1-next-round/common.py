import collections,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).parent
PILOT=ROOT.parent/'sts1-pilot8'
OUT=ROOT/'data'

def save(name,data):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')

def band(row):
    state=row['state'];ratio=state['hp']/max(1,state['max_hp'])
    return 'low' if ratio<.35 else 'medium' if ratio<.7 else 'high'

def stable(row):return hashlib.sha256(('next-round-select:'+row['id']).encode()).hexdigest()

def select_states(rows,target,balanced):
    pool=sorted(rows,key=stable);selected=[];seen=set();runs=collections.Counter();cells=collections.Counter()
    # Every eligible source run gets a state before adding repeated states.
    for row in pool:
        seed=row['provenance']['seed']
        if row['id'] not in seen and seed not in runs and len(selected)<target:
            selected.append(row);seen.add(row['id']);runs[seed]+=1;cells[(row['state']['act'],band(row))]+=1
    while len(selected)<target:
        available=[r for r in pool if r['id'] not in seen and runs[r['provenance']['seed']]<12]
        if not available:break
        if balanced:
            row=min(available,key=lambda r:(cells[(r['state']['act'],band(r))],runs[r['provenance']['seed']],stable(r)))
        else:row=available[0]
        selected.append(row);seen.add(row['id']);runs[row['provenance']['seed']]+=1;cells[(row['state']['act'],band(row))]+=1
    return selected
