#!/usr/bin/env python3
"""Diagnostic, never scored: is the OWED 17 build with the door off the same worker16 I/O program
as the B3 build? Alternates worker16 visits on the two frozen run-gen binaries (cold start each,
the B3 arm env verbatim) and records each visit's `[spill-pread]` totals line and token ids.
Run under the canonical lock inside the capped scope. Usage: m1-doorless-diag.py --out DIR --pairs N
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("runner", HERE / "m1-spill-runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
ART = "/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
BINS = {"b3": "/home/avifenesh/spill-f-5090/bin/run-gen", "f17": "/home/avifenesh/spill-f-5090/bin17/run-gen"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lock = json.loads((HERE / "m1-prereg/b3-arms.lock.json").read_text())
    arm = next(x for x in lock["arms"] if x["name"] == "worker16")
    env = R.arm_env(lock, arm)
    oracle = json.loads((HERE / "m1-prereg/f17-oracle-tokens.json").read_text())
    rows = []
    for k in range(a.pairs):
        for name in (("b3", "f17") if k % 2 == 0 else ("f17", "b3")):
            R.CACHE.cold([ART])
            log = out / f"p{k + 1}-{name}.log"
            t0 = time.monotonic()
            with log.open("xb") as f:
                rc = subprocess.run([BINS[name], ART], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
            text = log.read_text(errors="replace")
            parsed = R.parse_log(text)
            drop = parsed["drop"]
            row = {"pair": k + 1, "bin": name, "rc": rc, "seconds": round(time.monotonic() - t0, 1),
                   "tok_s": float(parsed["generated"][2]) if parsed["generated"] else None,
                   "tokens_equal_oracle": parsed["tokens"] == oracle,
                   "reads": int(drop[0]) if drop else None, "fallbacks": int(drop[4]) if drop else None,
                   "buffer_waits": int(drop[5]) if drop else None, "ring_full": int(drop[6]) if drop else None}
            rows.append(row)
            print("M1-DIAG " + json.dumps(row), flush=True)
    (out / "diag.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
