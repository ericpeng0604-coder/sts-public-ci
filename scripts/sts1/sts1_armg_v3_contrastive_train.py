#!/usr/bin/env python3
"""Outcome-aware ArmG v3: winner imitation plus conservative negative preference on losses."""
from __future__ import annotations
import argparse,hashlib,importlib,json,os,sys
from pathlib import Path
def split(n): return "val" if int(hashlib.sha256(n.encode()).hexdigest()[:8],16)%5==0 else "train"
def load(root):
 games=[]
 for p in sorted(root.rglob("seed-*.ndjson")):
  try: rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  except: continue
  s=next((r for r in reversed(rows) if r.get("type")=="summary"),{})
  ds=[r for r in rows if r.get("type")=="armg_noncombat_decision_v3" and r.get("selected_index",-1)>=0 and len(r.get("candidate_desc_368",[]))>1]
  if ds and s.get("outcome") in ("victory","defeat"): games.append((p.stem,s,ds))
 return games
def main():
 ap=argparse.ArgumentParser()
 for n in ("evidence","armg-root","base-weight","output"): ap.add_argument("--"+n,type=Path,required=True)
 ap.add_argument("--epochs",type=int,default=6);ap.add_argument("--lr",type=float,default=5e-5);ap.add_argument("--anchor",type=float,default=2e-4);ap.add_argument("--loss-weight",type=float,default=.15);a=ap.parse_args()
 os.environ["STS_BOT_DIR"]=str(a.armg_root);sys.path.insert(0,str(a.armg_root));torch=importlib.import_module("torch");m=importlib.import_module("armG_train");m.card_idx=lambda n:m._vocab.get(n,m.VOCAB_CAP-1)
 state=torch.load(a.base_weight,weights_only=True,map_location="cpu");net=m.Scorer((128,128));net.load_state_dict(state);base=[p.detach().clone() for p in net.parameters()]
 games=load(a.evidence); train=[g for g in games if split(g[0])=="train"];val=[g for g in games if split(g[0])=="val"]
 if not train or not val: raise SystemExit(f"insufficient split games train={len(train)} val={len(val)}")
 opt=torch.optim.Adam(net.parameters(),lr=a.lr);ce=torch.nn.CrossEntropyLoss()
 def decision_loss(r,outcome,grad):
  obs=torch.tensor(r["obs_412"],dtype=torch.float32);descs=r["candidate_desc_368"];idx=int(r["selected_index"])
  with torch.set_grad_enabled(grad):
   logits=net.score(obs,descs).reshape(1,-1)
   if outcome=="victory": loss=ce(logits,torch.tensor([idx]))
   else:
    # Conservative contrast: on failed runs, only penalize excessive confidence in the taken action.
    probs=torch.softmax(logits,1); p=probs[0,idx].clamp(max=.999999); loss=a.loss_weight*(-torch.log1p(-p))
   if grad: loss=loss+a.anchor*sum((p-b).pow(2).mean() for p,b in zip(net.parameters(),base))
  return loss
 best=None;bestvl=1e9;hist=[]
 for ep in range(1,a.epochs+1):
  net.train();tl=tn=0
  for _,s,ds in train:
   for r in ds: opt.zero_grad();z=decision_loss(r,s["outcome"],True);z.backward();opt.step();tl+=float(z.detach());tn+=1
  net.eval();vl=vn=0
  for _,s,ds in val:
   for r in ds: z=decision_loss(r,s["outcome"],False);vl+=float(z.detach());vn+=1
  row={"epoch":ep,"train_loss":tl/max(1,tn),"val_loss":vl/max(1,vn)};hist.append(row);print(json.dumps(row),flush=True)
  if row["val_loss"]<bestvl:bestvl=row["val_loss"];best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};bestep=ep
 a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(best,a.output)
 rep={"schema_version":"sts1-armg-v3-outcome-contrast-v1","games":len(games),"wins":sum(s.get("outcome")=="victory" for _,s,_ in games),"losses":sum(s.get("outcome")=="defeat" for _,s,_ in games),"train_games":len(train),"val_games":len(val),"best_epoch":bestep,"best_val_loss":bestvl,"history":hist}
 a.output.with_suffix(".json").write_text(json.dumps(rep,indent=2)+"\n");print("ARMG_V3_TRAIN_RESULT",json.dumps(rep))
if __name__=="__main__":main()
