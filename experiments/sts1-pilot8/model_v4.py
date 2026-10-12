"""Public-only ArmG-shaped nonlinear advantage scorer and bounded prior."""
import collections,json,pathlib,math
CAPABILITIES={
 'frontload':['Bash','Strike_R','Anger','Carnage','Bludgeon','Hemokinesis','Perfected Strike','Uppercut'],
 'aoe':['Cleave','Whirlwind','Immolate','Thunderclap'],
 'block':['Defend_R','Shrug It Off','Flame Barrier','Ghostly Armor','True Grit','Impervious','Power Through'],
 'draw':['Pommel Strike','Shrug It Off','Battle Trance','Offering','Burning Pact','Dark Embrace','Evolve'],
 'energy':['Seeing Red','Offering','Bloodletting','Berserk'],
 'scaling':['Demon Form','Inflame','Limit Break','Spot Weakness','Rampage'],
 'sustain':['Feed','Reaper'],
 'mitigation':['Uppercut','Clothesline','Shockwave','Disarm']}

def prior_feature(prior,state,action,actions):
    a=action.split('+')[0];total=score=0
    for other in actions:
        b=other.split('+')[0]
        if a==b:continue
        key=json.dumps([state['act'],a,b],separators=(',',':'))
        wins,losses=prior.get(key,[0,0]);n=wins+losses
        if n:score+=(wins-losses)/math.sqrt(n);total+=1
    return max(-1,min(1,score/max(1,total)/5))

def extras(state,action,actions,parent,scores,prior):
    counts=collections.Counter(c.split('+')[0] for c in state['deck']);stem=action.split('+')[0]
    f=[state['hp']/max(1,state['max_hp']),min(state['gold'],999)/999,state['act']/3,state['floor']/57,
       sum(c.get('upgrade_count',0)>0 for c in state['deck_details'])/max(1,len(state['deck'])),
       len(state['relics'])/30,sum(p!='EMPTY_POTION_SLOT' for p in state['potions'])/3,
       float(action==actions[parent]),max(-10,min(10,float(scores[actions.index(action)])))/10,
       max(-10,min(10,float(scores[parent])))/10,prior_feature(prior,state,action,actions)]
    public=state['public_extra']
    f.extend([public['potion_capacity']/3,public['shop_remove_count']/10,
              float(public['blue_key']),float(public['green_key']),float(public['red_key']),
              sum(min(1000,max(0,c['misc'])) for c in state['deck_details'])/1000,
              state['map_position']['x']/6,state['map_position']['y']/14])
    for cards in CAPABILITIES.values():
        f.extend([sum(counts[c] for c in cards)/max(1,len(state['deck'])),float(stem in cards)])
    return f

def inputs(row,prior):
    # Seed/provenance/result/relic internal data never enter this encoding.
    return [row['armg_observation']+d+extras(row['state'],a,row['actions'],row['g7_index'],row['g7_scores'],prior)
            for d,a in zip(row['armg_descriptors'],row['actions'])]

def make_model(input_dim,base_state=None):
    import torch
    net=torch.nn.Sequential(torch.nn.Linear(input_dim,128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,1))
    if base_state:
        # Keep original obs+descriptor index ordering, append reviewed features.
        with torch.no_grad():
            net[0].weight.zero_();net[0].weight[:,:base_state['net.0.weight'].shape[1]]=base_state['net.0.weight'];net[0].bias.copy_(base_state['net.0.bias'])
            net[2].weight.copy_(base_state['net.2.weight']);net[2].bias.copy_(base_state['net.2.bias'])
            net[4].weight.zero_();net[4].bias.zero_()
    return net
