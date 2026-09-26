#!/usr/bin/env python3
"""B3 `run-spec` K=1..8 self-consistency cell for one arm (M1-PREREG.md B3 step 1).

Usage (as the collector's --execute child):
  m1-spec-cell.py --arms-lock m1-prereg/b3-arms.lock.json --arm worker16 --binary run-spec --artifact FILE
Builds the arm's env exactly as the B3 runner does (common env, prompt env, arm env, unset list)
and execs run-spec in place, so the collector's capture is run-spec's own output and exit code.
The verdict line is run-spec's `=== SELF-CONSISTENCY PASS ===` (or FAIL).
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("runner", HERE / "m1-spill-runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
for name in ("--arms-lock", "--arm", "--binary", "--artifact"):
    ap.add_argument(name, required=True)
a = ap.parse_args()
lock = json.loads(Path(a.arms_lock).read_text())
arm = next(x for x in lock["arms"] if x["name"] == a.arm)
env = R.arm_env(lock, arm)
print(f"[m1-spec-cell] arm={a.arm} env=" + json.dumps({k: env[k] for k in sorted(env) if k.startswith("MEMRA_")}), flush=True)
os.execve(a.binary, [a.binary, a.artifact], env)
