#!/usr/bin/env python3
"""Pure decision controller for the STS1 self-improve diagnostic loop."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
d=json.loads(a.input.read_text())
div=d.get("divergence_pct")
if div is None: decision="TRAIN_CANDIDATE"; reason="margin measured; candidate/divergence not yet measured"
elif div < 5: decision="RETRAIN"; reason="decision divergence below 5%"
elif div <= 15: decision="PAIRED_AB_50"; reason="controlled divergence in 5-15% gate"
else: decision="HOLD_SMOKE"; reason="divergence above 15%; protect Champion"
out={"schema":"sts1-self-improve-controller-v1","decision":decision,"reason":reason,"champion_mutation":False,"inputs":d}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print("LOOP_CONTROLLER_PASS",json.dumps(out),flush=True)
