#!/usr/bin/env python3
"""Winner-weighted warm start for ArmG v2 using exact rich telemetry vectors."""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, sys
from pathlib import Path

def load_rows(root: Path):
    games=[]
    for p in sorted(root.glob("seed-*.ndjson")):
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        summary=next((r for r in reversed(rows) if r.get("type")=="summary"),{})
        if summary.get("outcome")!="victory":
            continue
        decisions=[r for r in rows if r.get("type")=="armg_noncombat_decision_v3" and r.get("selected_index", -1)>=0]
        if decisions:
            games.append((p.stem,decisions))
    return games

def split_seed(name: str) -> str:
    return "val" if int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%5==0 else "train"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",type=Path,required=True)
    ap.add_argument("--armg-root",type=Path,required=True)
    ap.add_argument("--base-weight",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--anchor",type=float,default=1e-4)
    a=ap.parse_args()
    os.environ["STS_BOT_DIR"]=str(a.armg_root)
    sys.path.insert(0,str(a.armg_root))
    torch=importlib.import_module("torch"); m=importlib.import_module("armG_train")
    m.card_idx=lambda name:m._vocab.get(name,m.VOCAB_CAP-1)
    net=m.Scorer((128,128)); base=m.Scorer((128,128))
    state=torch.load(a.base_weight,weights_only=True,map_location="cpu")
    net.load_state_dict(state); base.load_state_dict(state); base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    games=load_rows(a.evidence)
    train=[d for n,ds in games if split_seed(n)=="train" for d in ds]
    val=[d for n,ds in games if split_seed(n)=="val" for d in ds]
    if not train or not val: raise SystemExit(f"need winner games in both splits; games={len(games)} train={len(train)} val={len(val)}")
    opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    ce=torch.nn.CrossEntropyLoss()
    base_params=[p.detach().clone() for p in base.parameters()]
    def one(r,grad):
        obs=torch.tensor(r["obs_412"],dtype=torch.float32)
        desc=r["candidate_desc_368"]; target=torch.tensor([int(r["selected_index"])])
        with torch.set_grad_enabled(grad):
            logits=net.score(obs,desc).reshape(1,-1)
            loss=ce(logits,target)
            if grad:
                reg=sum((p-b).pow(2).mean() for p,b in zip(net.parameters(),base_params))
                loss=loss+a.anchor*reg
        return loss, int(logits.argmax(1).item()==target.item())
    history=[]
    for epoch in range(1,a.epochs+1):
        net.train(); total=correct=0; loss_sum=0.0
        for r in train:
            opt.zero_grad(); loss,ok=one(r,True); loss.backward(); opt.step()
            total+=1; correct+=ok; loss_sum+=float(loss.detach())
        net.eval(); vt=vc=0; vl=0.0
        for r in val:
            loss,ok=one(r,False); vt+=1; vc+=ok; vl+=float(loss.detach())
        row={"epoch":epoch,"train_loss":loss_sum/total,"train_acc":correct/total,"val_loss":vl/vt,"val_acc":vc/vt}
        history.append(row); print(json.dumps(row,sort_keys=True),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True); torch.save(net.state_dict(),a.output)
    report={"schema_version":"sts1-armg-v2-winner-warmstart-v1","winner_games":len(games),"train_decisions":len(train),"val_decisions":len(val),"epochs":a.epochs,"lr":a.lr,"anchor":a.anchor,"history":history,"output":str(a.output)}
    a.output.with_suffix(".json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("ARMG_V2_TRAIN_RESULT",json.dumps(report,sort_keys=True))
if __name__=="__main__": raise SystemExit(main())
