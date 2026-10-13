import json,pathlib,unittest
from paths import ROOT,PILOT
from features import inputs,build_net,predict,split_card

class FeaturesChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior=json.loads((PILOT/'human-prior.json').read_text())
        cls.row=next(r for p in (PILOT/'input-scouts').glob('*.json') for r in json.loads(p.read_text())['rows'] if r['id']=='f18c5ff182728203fe6271e6782aaa2acd3d89dd9870af2689b20191d48eddf1')
        cls.row={**cls.row,'g7_scores':[50.]*len(cls.row['actions'])}
    def test_distinct_colorless_cards(self):
        row=self.row;a=row['actions'].index('Violence');b=row['actions'].index('Jack Of All Trades')
        old=inputs(row,self.prior,'armg_v4_control');new=inputs(row,self.prior,'armg_v5_card_identity')
        self.assertEqual(old[a],old[b]);self.assertNotEqual(new[a],new[b])
    def test_no_seed_or_outcome_features(self):
        before=inputs(self.row,self.prior,'armg_v5_card_identity')
        after=inputs({**self.row,'provenance':{'seed':0},'outcome':'victory','world_results':[]},self.prior,'armg_v5_card_identity')
        self.assertEqual(before,after)
    def test_upgrade_parsing(self):
        self.assertEqual(split_card('Bash+2'),('Bash',2));self.assertEqual(split_card('SKIP'),('SKIP',0))
    def test_shapes_and_variance(self):
        import torch
        for architecture in ('armg_v4_control','armg_v5_card_identity'):
            x=torch.tensor(inputs(self.row,self.prior,architecture),dtype=torch.float32);net=build_net(x.shape[1],None,architecture);net.eval()
            mean,variance=predict(net,x,architecture)
            self.assertEqual(tuple(mean.shape),(len(self.row['actions']),));self.assertTrue(bool(torch.isfinite(mean).all()))
            if variance is not None:self.assertTrue(bool((variance>0).all()));self.assertTrue(bool(torch.isfinite(variance).all()))
            clone=build_net(x.shape[1],None,architecture);clone.load_state_dict(net.state_dict());clone.eval();self.assertTrue(torch.equal(predict(clone,x,architecture)[0],mean))

if __name__=='__main__':unittest.main()
