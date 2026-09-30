from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT=Path(__file__).parents[1]


def _load_module(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path)
    assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc=_load_module("v21_bc",ROOT/"scripts/sts1/sts1_build_rescue_bc_v21.py")
gate=_load_module("v21_gate",ROOT/"scripts/sts1/sts1_build_rescue_gate_v21.py")


def _teacher(seed:int=1):
    return {
        "schema_version":"sts1-armg-strategy-branch-dataset-v1",
        "seed":seed,
        "floor":21,
        "kind":"event",
        "obs":[0.0,1.0],
        "descs":[[0.0],[1.0]],
        "current_armg_index":0,
        "teacher_best_index":1,
        "target_probs":[0.0,1.0],
        "confidence_weight":1.0,
        "teacher_consensus_fraction":1.0,
        "teacher_margin":10.0,
        "priority":4.5,
    }


def test_verified_rescue_loader_deduplicates(tmp_path:Path):
    row=_teacher()
    p=tmp_path/"r.jsonl"
    p.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n",encoding="utf-8")
    rows=bc._load_rescues(p)
    assert len(rows)==1
    assert rows[0]["teacher_best_index"]==1


def test_verified_rescue_loader_rejects_unverified(tmp_path:Path):
    row=_teacher()
    row["confidence_weight"]=0.75
    p=tmp_path/"r.jsonl"
    p.write_text(json.dumps(row)+"\n",encoding="utf-8")
    with pytest.raises(RuntimeError,match="confidence"):
        bc._load_rescues(p)


def test_paired_sign_pvalue():
    assert gate._sign_pvalue(0,0)==1.0
    assert gate._sign_pvalue(3,0)==pytest.approx(0.125)
    assert gate._sign_pvalue(4,0)==pytest.approx(0.0625)
    assert gate._sign_pvalue(2,2)==pytest.approx(0.6875)


def test_gate_win_definition():
    assert gate._is_win({"outcome":"victory"})
    assert not gate._is_win({"outcome":"defeat"})
