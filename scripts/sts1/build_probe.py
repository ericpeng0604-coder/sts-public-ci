#!/usr/bin/env python3
"""Diagnostic-only simulator build with unbuffered compiler output."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "external" / "sts_lightspeed" / "source"
BUILD = SIM / "build-probe"
if BUILD.exists():
    shutil.rmtree(BUILD)
subprocess.run([
    "cmake", "-S", str(SIM), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release", f"-DPYTHON_EXECUTABLE={sys.executable}",
], check=True)
subprocess.run([
    "cmake", "--build", str(BUILD), "--target", "slaythespire", "--parallel", "4",
], check=True)
print("BUILD_PROBE_PASS")
