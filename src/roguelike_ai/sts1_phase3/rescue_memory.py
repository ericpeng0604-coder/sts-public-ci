"""Exact, auditable Rescue Memory layered on top of frozen ArmG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, SimulatorRunError


class ArmGRescueMemoryPolicy(ArmGNoncombatPolicy):
    """Frozen ArmG plus exact verified non-combat decision memories.

    Exact matching is intentionally conservative. It is a diagnostic/safe
    integration layer: no fuzzy generalization is claimed.
    """

    def __init__(
        self,
        *,
        root: Path,
        weight_path: Path,
        memory_path: Path,
        obs_tolerance: float = 1e-7,
        desc_tolerance: float = 1e-7,
    ) -> None:
        super().__init__(root=root, weight_path=weight_path)
        if not memory_path.is_file():
            raise SimulatorRunError(f"Rescue Memory file missing: {memory_path}")
        rows=[]
        for line_no,line in enumerate(memory_path.read_text(encoding="utf-8").splitlines(),1):
            if not line.strip():
                continue
            row=json.loads(line)
            obs=[float(x) for x in row["obs"]]
            descs=[[float(x) for x in d] for d in row["descs"]]
            teacher=int(row["teacher_best_index"])
            if not descs or not 0 <= teacher < len(descs):
                raise SimulatorRunError(f"invalid Rescue Memory row {line_no}")
            rows.append({
                "seed":int(row["seed"]),
                "floor":int(row.get("floor",0) or 0),
                "kind":str(row["kind"]),
                "obs":obs,
                "descs":descs,
                "teacher_best_index":teacher,
                "current_armg_index":int(row["current_armg_index"]),
                "type":str(row.get("type","")),
            })
        if not rows:
            raise SimulatorRunError("Rescue Memory is empty")
        self.memory_rows=rows
        self.obs_tolerance=float(obs_tolerance)
        self.desc_tolerance=float(desc_tolerance)
        self.memory_hits=0
        self.last_memory_match=None

    @staticmethod
    def _max_abs(a:list[float],b:list[float])->float:
        if len(a)!=len(b):
            return float("inf")
        return max((abs(float(x)-float(y)) for x,y in zip(a,b)),default=0.0)

    def _matches(self,row:dict[str,Any],kind:str,obs:list[float],descs:list[list[float]])->bool:
        if row["kind"]!=kind or len(row["descs"])!=len(descs):
            return False
        if self._max_abs(row["obs"],obs)>self.obs_tolerance:
            return False
        for x,y in zip(row["descs"],descs):
            if self._max_abs(x,y)>self.desc_tolerance:
                return False
        return True

    def decide(self,gc:Any,sts:Any):
        kind,selected,descs,execs,scores=super().decide(gc,sts)
        self.last_memory_match=None
        if selected < 0 or len(descs)<2:
            return kind,selected,descs,execs,scores
        snap=self.training_vector_snapshot(gc,descs)
        obs=list(snap["obs_412"])
        vecs=list(snap["candidate_desc_368"])
        for row in self.memory_rows:
            if self._matches(row,kind,obs,vecs):
                teacher=int(row["teacher_best_index"])
                self.memory_hits += 1
                self.last_memory_match={
                    "seed":row["seed"],
                    "floor":row["floor"],
                    "kind":kind,
                    "base_index":int(selected),
                    "teacher_index":teacher,
                }
                return kind,teacher,descs,execs,scores
        return kind,selected,descs,execs,scores
