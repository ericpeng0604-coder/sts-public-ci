import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).parent
OUT=ROOT.parent/'sts1-next-train/data'
path=OUT/'model-freeze.json'
row=json.loads(path.read_text())
row['mechanical_amendment_sha256']=hashlib.sha256((ROOT/'amendment.json').read_bytes()).hexdigest()
row['mechanical_source_sha256']=json.loads((ROOT/'amendment.json').read_text())['source_sha256']
row['known_simulator_courier_pricing_bug']=True
row['real_game_performance_claim_supported']=False
path.write_text(json.dumps(row,indent=2)+'\n')
