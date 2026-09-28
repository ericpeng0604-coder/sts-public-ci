#!/usr/bin/env python3
import argparse,json
p=argparse.ArgumentParser();p.add_argument("--error",required=True);p.add_argument("--output",required=True);a=p.parse_args()
e=open(a.error,errors="ignore").read()
if "stale shard" in e:
 d={"action":"SET_CHECKPOINT_ID","value":"armg-original-15k-v11-r1","retry":True,"champion_mutation":False}
elif "No module named 'slaythespire'" in e:
 d={"action":"REPAIR_SIMULATOR_PATH","retry":True,"champion_mutation":False}
else:
 d={"action":"STOP_UNKNOWN","retry":False,"champion_mutation":False}
open(a.output,"w").write(json.dumps(d,indent=2)+"\n");print("AUTO_REPAIR_DECISION",json.dumps(d))
