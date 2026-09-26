#!/usr/bin/env python3
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

CAPS={
 "strength":{"Inflame","Spot Weakness","Demon Form","Limit Break","Flex"},
 "aoe":{"Cleave","Whirlwind","Immolate","Thunderclap","Reaper"},
 "draw":{"Battle Trance","Pommel Strike","Shrug It Off","Burning Pact","Offering"},
 "energy":{"Offering","Bloodletting","Seeing Red"},
 "exhaust":{"Fiend Fire","Burning Pact","True Grit","Second Wind","Corruption","Feel No Pain","Dark Embrace"},
 "block":{"Shrug It Off","Impervious","Flame Barrier","Power Through","Ghostly Armor","Entrench"},
 "frontload":{"Carnage","Bludgeon","Perfected Strike","Hemokinesis","Uppercut","Wild Strike","Clothesline"},
}

def rows(path):
 for p in sorted(path.glob("seed-*.ndjson")):
  rs=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  summ=next((x for x in reversed(rs) if x.get("type")=="summary"),{})
  yield p.stem,rs,summ

def card_name(x):
 c=x.get("card") or {}
 return c.get("name") or c.get("id") or x.get("card_name")

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("evidence",type=Path); ap.add_argument("--output",type=Path,default=Path("teacher-analysis.json")); a=ap.parse_args()
 groups=defaultdict(list)
 for seed,rs,s in rows(a.evidence):
  floor=int(s.get("final_floor",s.get("floor",0)) or 0); outcome=s.get("outcome")
  g="winner" if outcome=="victory" else ("near_win" if floor>=45 else "loss")
  plays=Counter(); choices=Counter(); deck=Counter(); turns=Counter()
  for r in rs:
   if r.get("type")=="combat_play_trace_v2":
    n=card_name(r)
    if n: plays[n]+=1
    turns[(r.get("floor"),r.get("turn"))]+=1
   elif r.get("type")=="armg_noncombat_decision_v2":
    sem=r.get("selected_semantics") or {}
    choices[json.dumps(sem,sort_keys=True)]+=1
    for c in r.get("deck_before") or []:
     n=c.get("name") if isinstance(c,dict) else None
     if n: deck[n]+=1
  names=set(deck)|set(plays)
  caps={k:sum((deck[n] or plays[n]) for n in v if n in names) for k,v in CAPS.items()}
  groups[g].append({"seed":seed,"floor":floor,"plays":plays,"deck":deck,"caps":caps,"decisions":len([r for r in rs if r.get("type")=="combat_play_trace_v2"])})
 def agg(xs):
  n=len(xs); pc=Counter(); dc=Counter(); cc=Counter()
  for x in xs: pc.update(x["plays"]); dc.update(x["deck"]); cc.update(x["caps"])
  return {"runs":n,"mean_floor":sum(x["floor"] for x in xs)/n if n else 0,"mean_combat_decisions":sum(x["decisions"] for x in xs)/n if n else 0,
   "top_played_cards":pc.most_common(25),"top_deck_cards":dc.most_common(25),"capability_signal_per_run":{k:v/n for k,v in cc.items()} if n else {}}
 out={"schema_version":"sts1-teacher-archetype-analysis-v1","groups":{k:agg(v) for k,v in groups.items()},
      "note":"Descriptive winner/near-win/loss comparison. Capability signals are diagnostics, not hard-coded archetype labels."}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
