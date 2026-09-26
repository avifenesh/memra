#!/usr/bin/env python3
"""The 5090 half's queue (M1-PREREG.md sections D, E, F): both doors' correctness gates first
(GPU ownership cell, f17 smoke and gate, the handoff 1 GiB pairs), then the timing regimes.

Each step is one `m1-5090-rounds.py` invocation (idle-gated per round cell) or one GPU unit cell
under the canonical lock. A step's exit code and times go to `QUEUE.jsonl`; a failing step stops
the queue, with its log kept. The bounded step's MemoryMax comes from the anon-peak step's
`bounded_memory_max_bytes` (bounded amendment). Usage: m1-5090-queue.py --receipts DIR [--from STEP]
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUNDS = HERE / "m1-5090-rounds.py"
CAP = str(20 * 1024 ** 3)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def steps(rec, first_rounds="1-10", bounded_bytes=None, over=None):
    over = over or {}
    def rounds(regime, out, *extra):
        return [sys.executable, str(ROUNDS), "--regime", regime, "--out", str(rec / out), *extra]
    return [
        ("capped", lambda: rounds("capped", "capped", "--memory-max", CAP, "--rounds", first_rounds)),
        ("anonpeak", lambda: rounds("anonpeak", "anonpeak", "--memory-max", CAP)),
        ("mapped-gpu-cell", lambda: rounds("gpucell", "mapped-gpu-cell", "--memory-max", CAP)),
        ("f17-smoke", lambda: rounds("f17", "f17-smoke", "--memory-max", CAP, "--rounds", "1", "--smoke")),
        ("f17-smoke-gate", lambda: [sys.executable, str(HERE / "m1-b3-pool.py"), str(rec / "f17-smoke"),
                                    "--bypass-check", "--fallback-unclean", "--require-correct"]),
        ("owed26-cells", lambda: rounds("owed26cells", "owed26-cells", "--memory-max", CAP)),
        ("owed26-serve", lambda: rounds("owed26serve", "owed26-serve", "--memory-max", CAP)),
        ("handoff-1g", lambda: rounds("handoff", "handoff-1g", "--memory-max", CAP, "--rounds", over.get("handoff-1g", "1-10"),
                                      "--size-bytes", str(1 << 30), "--host-mb", "4096", "--tenant-pct", "100")),
        ("bounded", lambda: rounds("bounded", "bounded", "--memory-max", str(bounded_bytes or bounded_max(rec)), "--rounds", "1-10")),
        ("g2", lambda: rounds("g2", "g2", "--memory-max", CAP)),
        ("f17", lambda: rounds("f17", "f17", "--memory-max", CAP, "--rounds", "1-10")),
        ("handoff-8g", lambda: rounds("handoff", "handoff-8g", "--memory-max", CAP, "--rounds", "1-10",
                                      "--size-bytes", str(8 << 30), "--host-mb", "12288", "--tenant-pct", "100")),

    ]


def bounded_max(rec):
    runs = sorted((rec / "anonpeak").glob("round-01*/visits/anon-peak.json"))
    if not runs:
        raise SystemExit("REFUSED: no anon-peak result; the bounded regime cannot be sized")
    result = json.loads(runs[-1].read_text())
    if not (result["all_exit_zero"] and result["all_correct"]):
        raise SystemExit("REFUSED: the anon-peak cell did not pass")
    return int(result["bounded_memory_max_bytes"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--from", dest="start", default="capped")
    ap.add_argument("--capped-rounds", default="1-10",
                    help="resume: the capped step's rounds (a reboot-interrupted round is banked, then rerun)")
    ap.add_argument("--bounded-max", type=int, help="the registered bounded MemoryMax (bounded sizing record)")
    ap.add_argument("--step-rounds", action="append", default=[],
                    help="resume a rounds step mid-way, e.g. handoff-1g=3-10")
    a = ap.parse_args()
    rec = Path(a.receipts)
    rec.mkdir(parents=True, exist_ok=True)
    plan = steps(rec, a.capped_rounds, a.bounded_max, dict(x.split("=", 1) for x in a.step_rounds))
    names = [n for n, _ in plan]
    with (rec / "QUEUE.jsonl").open("a") as q:
        for name, argv in plan[names.index(a.start):]:
            cmd = argv()
            t0 = time.monotonic()
            q.write(json.dumps({"utc": now(), "step": name, "event": "start", "argv": cmd}) + "\n")
            q.flush()
            with (rec / f"{name}.queue.log").open("ab") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
            q.write(json.dumps({"utc": now(), "step": name, "event": "end", "rc": rc,
                                "seconds": round(time.monotonic() - t0, 1)}) + "\n")
            q.flush()
            print(f"M1-5090-QUEUE step={name} rc={rc}", flush=True)
            if rc != 0:
                return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
