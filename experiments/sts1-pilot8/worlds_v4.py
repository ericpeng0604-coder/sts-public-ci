import hashlib,json,math,pathlib,time
from collect_v4 import full_state,G7
from collect_v3 import finish,battle
from state_v3 import action_names

def world_seed(case_id,index):
    return int(hashlib.sha256(('v4-world:'+case_id+':'+str(index)).encode()).hexdigest()[:15],16)

def aggregate(worlds,parent):
    n=len(worlds);counts=len(worlds[0]);out=[]
    for a in range(counts):
        deltas=[float(w[a]['outcome']=='victory')-float(w[parent]['outcome']=='victory') for w in worlds]
        mean=sum(deltas)/n;var=sum((x-mean)**2 for x in deltas)/(n-1) if n>1 else None
        # Distribution-free simultaneous bounds for [-1,1] paired deltas.
        radius=math.sqrt(2*math.log(2*counts/0.05)/n)
        out.append({'index':a,'worlds':n,'win_advantage':mean,'paired_variance':var,'standard_error':math.sqrt(var/n) if var is not None else None,
                    'conservative_lower':max(-1,mean-radius),'conservative_upper':min(1,mean+radius),
                    'mean_floor_delta':sum(w[a]['floor']-w[parent]['floor'] for w in worlds)/n})
    return out

def label(original,policy,sts,row,max_worlds=16):
    worlds=[];seeds=[];actions=row['actions'];parent=row['g7_index']
    _,pindex,_,_,gscores=policy.decide(original,sts)
    if pindex!=parent:raise RuntimeError('teacher_parent_mismatch')
    row={**row,'g7_scores':list(gscores)}
    for j in range(max_worlds):
        seed=world_seed(row['id'],j);base=original.sample_public_future(seed)
        kind,descs,_=policy.choices(base)
        if full_state(base,policy,descs)!=row['state'] or action_names(base,policy,kind,descs)!=actions:raise RuntimeError('world_public_state_or_action_changed')
        outcomes=[]
        for a in range(len(actions)):
            gc=base.clone();kind,descs,execs=policy.choices(gc)
            if action_names(gc,policy,kind,descs)!=actions:raise RuntimeError('paired_world_action_mismatch')
            execs[a](gc);outcomes.append(finish(gc,sts,policy))
        kind,descs,_=policy.choices(base)
        if full_state(base,policy,descs)!=row['state']:raise RuntimeError('branch_changed_world_root')
        worlds.append(outcomes);seeds.append(seed)
        if j+1==8:
            estimates=aggregate(worlds,parent)
            best=max(estimates,key=lambda r:r['win_advantage'])
            # Close/uncertain alternatives receive 16; otherwise stop at prereg 8.
            if best['conservative_lower']>0:break
    estimates=aggregate(worlds,parent)
    return {**row,'label_status':'PAIRED_STOCHASTIC_FUTURE_ESTIMATE','sampler':'conditional-transition-model-v1',
            'sampler_limit':'Visible state and encounter history preserved; independent future RNG streams, conditional unseen queue generation, uniform remaining relic order. Approximate stochastic continuation, NOT exact original-seed conditional posterior.',
            'teacher_combat_mcts':2000,'world_seeds':seeds,'world_results':worlds,'estimates':estimates,
            'uncertain':not any(x['conservative_lower']>0 for x in estimates if x['index']!=parent)}
