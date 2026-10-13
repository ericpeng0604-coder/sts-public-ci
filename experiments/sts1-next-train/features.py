"""ArmG v4 control and appended card identity/upgrade/score-gap features."""
import math,pathlib,sys
PILOT=pathlib.Path(__file__).parent.parent/'sts1-pilot8'
sys.path.insert(0,str(PILOT))
from model_v4 import inputs as original_inputs,make_model
from catalog import IDS
VOCAB=[c for c in IDS if c and c!='INVALID']+['SKIP','UNKNOWN_CARD']
VOCAB=list(dict.fromkeys(VOCAB));INDEX={c:i for i,c in enumerate(VOCAB)}

def split_card(name):
    stem,sep,upgrade=name.partition('+')
    return stem,int(upgrade) if sep and upgrade.isdigit() else 0

def inputs(row,prior,architecture):
    base=original_inputs(row,prior)
    if architecture=='armg_v4_control':return base
    if architecture!='armg_v5_card_identity':raise ValueError(architecture)
    result=[]
    for index,(action,encoded) in enumerate(zip(row['actions'],base)):
        stem,upgrade=split_card(action);identity=[0.]*len(VOCAB)
        identity[INDEX.get(stem,INDEX['UNKNOWN_CARD'])]=1.
        copies=sum(split_card(card)[0]==stem for card in row['state']['deck'])
        upgraded=sum(split_card(card)[0]==stem and split_card(card)[1]>0 for card in row['state']['deck'])
        score=float(row['g7_scores'][index]);parent=float(row['g7_scores'][row['g7_index']])
        assert math.isfinite(score) and math.isfinite(parent)
        result.append(encoded+identity+[min(upgrade,10)/10,copies/max(1,len(row['state']['deck'])),upgraded/max(1,len(row['state']['deck'])),math.tanh(score/50),math.tanh((score-parent)/10)])
    return result

def build_net(input_dim,base,architecture):
    import torch
    net=make_model(input_dim,base)
    if architecture=='armg_v5_card_identity':
        old=net[4];net[4]=torch.nn.Linear(128,2)
        with torch.no_grad():
            net[4].weight.zero_();net[4].bias.zero_();net[4].weight[0].copy_(old.weight[0]);net[4].bias[0].copy_(old.bias[0]);net[4].bias[1]=-3.
    return net

def predict(net,x,architecture):
    import torch
    raw=net(x)
    if architecture=='armg_v4_control':return raw.squeeze(-1),None
    return raw[:,0],torch.nn.functional.softplus(raw[:,1])
