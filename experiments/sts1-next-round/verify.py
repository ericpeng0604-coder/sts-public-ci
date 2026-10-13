import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).parent

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for name,expected in protocol['file_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'source_hash_mismatch:'+name
    cohorts=json.loads((ROOT/'cohort-manifest.json').read_text())['cohorts'];sets={k:set(v) for k,v in cohorts.items()}
    assert len(sets['train'])==48 and len(sets['calibration'])==32 and len(sets['evaluation_reserved'])==2000
    assert all(len(v)==len(cohorts[k]) for k,v in sets.items())
    assert all(not sets[a]&sets[b] for a in sets for b in sets if a!=b)
    old=ROOT.parent/'sts1-pilot8';excluded=set(map(str,json.loads((old/'seed-exclusions.json').read_text())))
    excluded.update(p.stem for p in (old/'input-scouts').glob('*.json'))
    assert all(not v&excluded for v in sets.values())
    shards=json.loads((ROOT/'collection-shards.json').read_text());assert len(shards)==8
    actual=[task['seed'] for group in shards for task in group]
    assert len(actual)==80 and len(set(actual))==80 and set(actual)==sets['train']|sets['calibration']
    assert all(t['seed'] in sets[t['split']] for group in shards for t in group)
    print(json.dumps({'status':'PASS','cohorts_disjoint':True,'new_source_runs':80,'formal_evaluation_seeds_not_executed':2000}),flush=True)

if __name__=='__main__':main()
