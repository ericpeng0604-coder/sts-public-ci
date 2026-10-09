from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_pinned_native_binding_exposes_read_only_five_slot_potion_snapshot():
    module_path = os.environ.get("STS1_POTION_BINDING_MODULE")
    if not module_path:
        pytest.skip("set STS1_POTION_BINDING_MODULE to the pinned native extension under test")

    code = r"""
import importlib.util
import json
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("slaythespire", path)
if spec is None or spec.loader is None:
    raise RuntimeError("native binding module could not be loaded")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
context = module.GameContext(module.CharacterClass.IRONCLAD, 0, 0)
slots = list(context.potions)
assert slots == ["EMPTY_POTION_SLOT"] * 5, slots
try:
    context.potions = ["FIRE_POTION"] * 5
except AttributeError:
    pass
else:
    raise AssertionError("native potion inventory property must be read-only")
print(json.dumps({"potion_slots": slots, "episodes_started": 0}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(Path(module_path).resolve())],
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert '"episodes_started": 0' in completed.stdout
