import sys
from paths import PILOT
sys.path.insert(0,str(PILOT))
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy
from collect_v4 import full_state
from state_v3 import action_names
from features import inputs,build_net,predict

class AdvantagePolicy(ArmGNoncombatPolicy):
    def __init__(self,*,model_path,**kwargs):
        super().__init__(**kwargs);self.payload=self.torch.load(model_path,weights_only=True,map_location='cpu')
        self.advantage=build_net(self.payload['input_dim'],None,self.payload['architecture']);self.advantage.load_state_dict(self.payload['state_dict']);self.advantage.eval()
        self.overrides=[];self.considered=0;self.abstentions={};self.state_coverage={};self.decision_states=[]
    def decide(self,gc,sts):
        kind,parent,descs,execs,scores=super().decide(gc,sts)
        if kind!='card' or len(descs)<2 or self.payload['gate'] is None:return kind,parent,descs,execs,scores
        self.considered+=1;state=full_state(gc,self,descs);actions=action_names(gc,self,kind,descs)
        row={'state':state,'actions':actions,'g7_index':parent,'g7_scores':scores,'armg_observation':self.module.obs_vec(gc),'armg_descriptors':descs}
        with self.torch.no_grad():mean,variance=predict(self.advantage,self.torch.tensor(inputs(row,self.payload['prior'],self.payload['architecture']),dtype=self.torch.float32),self.payload['architecture'])
        index=int(mean.argmax());gain=float(mean[index]-mean[parent])
        if index==parent or gain<self.payload['gate']:return kind,parent,descs,execs,scores
        self.overrides.append({'floor':state['floor'],'parent':actions[parent],'candidate':actions[index],'predicted_gain':gain,'predicted_outcome_variance':float(variance[index]) if variance is not None else None})
        return kind,index,descs,execs,scores
