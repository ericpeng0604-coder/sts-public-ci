import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).parent
def main():
    config=json.loads((ROOT/'amendment.json').read_text())
    for name,expected in config['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, 'recovery_hash:'+name
    assert config['combat_mcts']==2000 and config['game_step_bound']==10000
    assert config['native_engine_changed'] is False and config['resample_failed_worlds'] is False
    assert config['model_hyperparameters_changed'] is False and config['calibration_results_seen'] is False
    print(json.dumps({'mechanical_amendment':'PASS','known_simulator_courier_pricing_bug':True,'real_game_performance_claim_supported':False}),flush=True)
if __name__=='__main__':main()
