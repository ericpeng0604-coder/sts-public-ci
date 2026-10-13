import math,unittest
from replay_v4 import recorded_index

class ReplayChecks(unittest.TestCase):
    def setUp(self):self.source={'seed':1,'journal':[{'kind':'map','index':1,'floor':31}]}
    def test_exact_record(self):self.assertEqual(recorded_index(self.source,0,'map',1,31,[[1],[1]],[1.,1.]),(1,None))
    def test_equivalent_float32_tie(self):
        index,detail=recorded_index(self.source,0,'map',0,31,[[1],[1]],[1.000001,1.])
        self.assertEqual(index,1);self.assertEqual(detail['reason'],'equivalent_map_inputs_float32_tie')
    def test_meaningful_mismatches_fail_closed(self):
        for args in [('map',0,31,[[1],[2]],[1.,1.]),('map',0,31,[[1],[1]],[1.,1.1]),('map',0,32,[[1],[1]],[1.,1.]),('card',0,31,[[1],[1]],[1.,1.]),('map',0,31,[[1],[1]],[math.nan,math.nan])]:
            with self.subTest(args=args),self.assertRaises(RuntimeError):recorded_index(self.source,0,*args)
    def test_missing_record_rejected(self):
        with self.assertRaises(RuntimeError):recorded_index(self.source,1,'map',0,31,[[1]],[1.])
    def test_illegal_record_rejected(self):
        self.source['journal'][0]['index']=2
        with self.assertRaises(RuntimeError):recorded_index(self.source,0,'map',0,31,[[1],[1]],[1.,1.])

if __name__=='__main__':unittest.main()
