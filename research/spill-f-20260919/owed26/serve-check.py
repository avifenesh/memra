#!/usr/bin/env python3
"""OWED 26 G2 serving-shape check (unscored): the fix build's worker arms read no block through
mmap. Visits `worker2`, `worker16` (B3 lock rows) and `bypass-mapped` (F lock row) on the frozen fix
run-gen, cold start each, two passes in alternating order; each visit's totals line and token ids.
PASS when every visit has zero fallbacks and zero demand-wait timeouts, prints the new counters,
and generates tokens equal to the byte oracle. Run under the canonical lock inside the capped scope.
Usage: serve-check.py --binary RUN_GEN --out DIR
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("runner", HERE / "m1-spill-runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
ART = "/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
NEW = re.compile(r"demand_waits=(\d+) demand_wait_ns=(\d+) prefetch_cancels=(\d+) demand_wait_timeouts=(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    b3 = json.loads((HERE / "m1-prereg/b3-arms.lock.json").read_text())
    f17 = json.loads((HERE / "m1-prereg/f17-arms.lock.json").read_text())
    arms = [(b3, next(x for x in b3["arms"] if x["name"] == "worker2")),
            (b3, next(x for x in b3["arms"] if x["name"] == "worker16")),
            (f17, next(x for x in f17["arms"] if x["name"] == "bypass-mapped"))]
    oracle = json.loads((HERE / "m1-prereg/f17-oracle-tokens.json").read_text())
    rows = []
    for k in range(2):
        for lock, arm in (arms if k == 0 else arms[::-1]):
            R.CACHE.cold([ART])
            log = out / f"p{k + 1}-{arm['name']}.log"
            gpu = R.GPU.GpuSampler(out / f"p{k + 1}-{arm['name']}.gpu.csv").start()
            t0 = time.monotonic()
            with log.open("xb") as f:
                rc = subprocess.run([a.binary, ART], stdout=f, stderr=subprocess.STDOUT,
                                    env=R.arm_env(lock, arm)).returncode
            telemetry = gpu.stop()
            text = log.read_text(errors="replace")
            parsed = R.parse_log(text)
            drop, new = parsed["drop"], NEW.search(text)
            row = {"pass": k + 1, "arm": arm["name"], "rc": rc, "seconds": round(time.monotonic() - t0, 1),
                   "tok_s": float(parsed["generated"][2]) if parsed["generated"] else None,
                   "tokens_equal_oracle": parsed["tokens"] == oracle,
                   "fallbacks": int(drop[4]) if drop else None,
                   "demand_waits": int(new[1]) if new else None, "demand_wait_ns": int(new[2]) if new else None,
                   "prefetch_cancels": int(new[3]) if new else None,
                   "demand_wait_timeouts": int(new[4]) if new else None, "gpu_telemetry": telemetry}
            row["ok"] = bool(rc == 0 and row["tokens_equal_oracle"] and row["fallbacks"] == 0 and new
                             and row["demand_wait_timeouts"] == 0)
            rows.append(row)
            print("OWED26-SERVE " + json.dumps({k2: v for k2, v in row.items() if k2 != "gpu_telemetry"}), flush=True)
    ok = all(r["ok"] for r in rows)
    (out / "serve-check.json").write_text(json.dumps({"pass": ok, "visits": rows}, indent=1) + "\n")
    print(f"OWED26-SERVE-CHECK {'PASS' if ok else 'FAIL'} visits={len(rows)}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
