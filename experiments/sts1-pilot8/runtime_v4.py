import pathlib
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy
from model_v4 import inputs,make_model
from collect_v4 import full_state
from state_v3 import action_names

class AdvantagePolicy(ArmGNoncombatPolicy):
    def __init__(self,*,model_path,**kwargs):
        super().__init__(**kwargs);self.payload=self.torch.load(model_path,weights_only=True,map_location='cpu')
        self.advantage=make_model(self.payload['input_dim']);self.advantage.load_state_dict(self.payload['state_dict']);self.advantage.eval()
        self.overrides=[];self.considered=0;self.abstentions={};self.state_coverage={};self.decision_states=[]
    def decide(self,gc,sts):
        kind,parent,descs,execs,scores=super().decide(gc,sts)
        if kind!='card' or len(descs)<2 or self.payload['gate'] is None:return kind,parent,descs,execs,scores
        self.considered+=1
        state=full_state(gc,self,descs);actions=action_names(gc,self,kind,descs)
        row={'state':state,'actions':actions,'g7_index':parent,'g7_scores':list(scores),'armg_observation':self.module.obs_vec(gc),'armg_descriptors':descs}
        with self.torch.no_grad():
            advantage=self.advantage(self.torch.tensor(inputs(row,self.payload['prior']),dtype=self.torch.float32)).squeeze(-1)
        index=int(advantage.argmax());delta=float(advantage[index]-advantage[parent])
        if index==parent or delta<self.payload['gate']:return kind,parent,descs,execs,scores
        self.overrides.append({'floor':state['floor'],'parent':actions[parent],'candidate':actions[index],'predicted_relative_advantage':delta})
        return kind,index,descs,execs,scores
