import pathlib,sys,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).parent.parent/'sts1-pilot8'))
from mechanical import recorded_index,finish

class MechanicalTests(unittest.TestCase):
    def source(self,kind='shop'):
        return {'seed':123,'journal':[{'kind':kind,'index':1,'floor':37}]}
    def test_equivalent_shop_inputs_restore_recorded_legal_action(self):
        index,tie=recorded_index(self.source(),0,'shop',0,37,[[1],[1]],[.1,.100000001])
        self.assertEqual(index,1);self.assertEqual(tie['reason'],'equivalent_shop_inputs_float32_tie')
    def test_different_shop_item_encoding_rejected(self):
        with self.assertRaises(RuntimeError):recorded_index(self.source(),0,'shop',0,37,[[1],[2]],[.1,.1])
    def test_scores_outside_tie_tolerance_rejected(self):
        with self.assertRaises(RuntimeError):recorded_index(self.source(),0,'shop',0,37,[[1],[1]],[.1,.11])
    def test_card_teacher_never_relaxed(self):
        with self.assertRaises(RuntimeError):recorded_index(self.source('card'),0,'card',0,37,[[1],[1]],[.1,.1])
    def test_progressing_run_beyond_600_steps_completes(self):
        gc=types.SimpleNamespace(outcome=0,screen_state=0,gold=701,floor_num=47,cur_hp=80,max_hp=80)
        class Agent:
            def playout(self,gc):pass
        class Policy:
            def step(self,gc,sts):
                gc.gold-=1
                if gc.gold==0:gc.outcome=1
        sts=types.SimpleNamespace(Agent=Agent,GameOutcome=types.SimpleNamespace(UNDECIDED=0,PLAYER_VICTORY=1),ScreenState=types.SimpleNamespace(BATTLE=1))
        with patch('roguelike_ai.sts1_phase3.simulator._set_pauses'):
            result=finish(gc,sts,Policy())
        self.assertEqual(result['outcome'],'victory');self.assertEqual(gc.gold,0)

if __name__=='__main__':unittest.main()
