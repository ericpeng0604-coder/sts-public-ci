import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).parent
PILOT=ROOT.parent/'sts1-pilot8'
NEXT=ROOT.parent/'sts1-next-round'
OUT=ROOT/'data'
def save(name,data):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
