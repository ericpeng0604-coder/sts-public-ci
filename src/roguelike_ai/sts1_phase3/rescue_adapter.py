"""Local similarity Rescue Adapter layered on top of frozen ArmG G7."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, SimulatorRunError


class ArmGLocalRescueAdapterPolicy(ArmGNoncombatPolicy):
    """Frozen ArmG plus a calibrated, local nearest-prototype override.

    The base ArmG weights are never changed.  A candidate action is overridden
    only when its (obs_412, desc_368) vector falls inside a prototype-specific
    radius that was calibrated against frozen G7 winner replay.
    """

    def __init__(
        self,
        *,
        root: Path,
        weight_path: Path,
        adapter_path: Path,
    ) -> None:
        super().__init__(root=root, weight_path=weight_path)
        if not adapter_path.is_file():
            raise SimulatorRunError(f"Local Rescue Adapter missing: {adapter_path}")
        payload=json.loads(adapter_path.read_text(encoding="utf-8"))
        if payload.get("schema_version")!="sts1-local-rescue-adapter-v31":
            raise SimulatorRunError("Local Rescue Adapter schema mismatch")
        self.obs_scale=[float(x) for x in payload["obs_scale"]]
        self.desc_scale=[float(x) for x in payload["desc_scale"]]
        if len(self.obs_scale)!=412 or len(self.desc_scale)!=368:
            raise SimulatorRunError("Local Rescue Adapter feature scale mismatch")
        self.prototypes=[]
        for i,row in enumerate(payload.get("prototypes") or []):
            obs=[float(x) for x in row["obs"]]
            desc=[float(x) for x in row["teacher_desc"]]
            radius=float(row["radius"])
            if len(obs)!=412 or len(desc)!=368 or radius<0:
                raise SimulatorRunError(f"invalid Local Rescue Adapter prototype {i}")
            self.prototypes.append({
                "seed":int(row["seed"]),
                "floor":int(row.get("floor",0) or 0),
                "kind":str(row["kind"]),
                "obs":obs,
                "teacher_desc":desc,
                "radius":radius,
                "source_type":str(row.get("source_type","")),
            })
        if not self.prototypes:
            raise SimulatorRunError("Local Rescue Adapter has no prototypes")
        self.adapter_hits=0
        self.adapter_overrides=0
        self.last_adapter_match=None

    @staticmethod
    def _sq_scaled(a:list[float],b:list[float],scale:list[float])->float:
        if len(a)!=len(b) or len(a)!=len(scale):
            return float("inf")
        total=0.0
        for x,y,s in zip(a,b,scale):
            z=(float(x)-float(y))/float(s)
            total += z*z
        return total

    def _distance(
        self,
        proto:dict[str,Any],
        obs:list[float],
        desc:list[float],
    )->float:
        total=self._sq_scaled(proto["obs"],obs,self.obs_scale)
        total+=self._sq_scaled(proto["teacher_desc"],desc,self.desc_scale)
        return math.sqrt(total/(len(self.obs_scale)+len(self.desc_scale)))

    def decide(self,gc:Any,sts:Any):
        kind,selected,descs,execs,scores=super().decide(gc,sts)
        self.last_adapter_match=None
        if selected<0 or len(descs)<2:
            return kind,selected,descs,execs,scores

        snap=self.training_vector_snapshot(gc,descs)
        obs=[float(x) for x in snap["obs_412"]]
        vecs=[[float(x) for x in row] for row in snap["candidate_desc_368"]]

        best=None
        for proto_idx,proto in enumerate(self.prototypes):
            if proto["kind"]!=str(kind):
                continue
            radius=float(proto["radius"])
            for cand_idx,desc in enumerate(vecs):
                distance=self._distance(proto,obs,desc)
                if distance<=radius+1e-12:
                    key=(distance,proto_idx,cand_idx)
                    if best is None or key<best[0]:
                        best=(key,proto,cand_idx,distance)
        if best is None:
            return kind,selected,descs,execs,scores

        _,proto,cand_idx,distance=best
        self.adapter_hits += 1
        if int(cand_idx)!=int(selected):
            self.adapter_overrides += 1
        self.last_adapter_match={
            "prototype_seed":proto["seed"],
            "prototype_floor":proto["floor"],
            "kind":str(kind),
            "distance":float(distance),
            "radius":float(proto["radius"]),
            "base_index":int(selected),
            "adapter_index":int(cand_idx),
        }
        return kind,int(cand_idx),descs,execs,scores
