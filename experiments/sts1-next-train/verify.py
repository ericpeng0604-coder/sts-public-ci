import json,pathlib
from paths import ROOT,PILOT,OUT,digest

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for name,expected in protocol['file_sha256'].items():assert digest(ROOT/name)==expected,'own_source_hash:'+name
    for name,expected in protocol['pilot_sha256'].items():assert digest(PILOT/name)==expected,'pilot_source_hash:'+name
    assert len(protocol['architectures'])==2 and len(protocol['thresholds'])==6 and protocol['multiple_comparisons']==12
    assert protocol['evaluation_pairs']==1200 and protocol['combat_mcts']==2000
    print(json.dumps({'status':'PASS','models':2,'gates_per_model':6,'formal_pairs_if_calibrated':1200,'production_adoption':False}),flush=True)

if __name__=='__main__':main()
