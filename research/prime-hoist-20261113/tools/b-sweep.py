#!/usr/bin/env python3
"""memra#1258 margin sweep: cold DFlash primes one at a time beside a decoding keeper (fresh
/metrics, a warm pool the previous prime left), each placed by a ballast at the gate's required
plus a margin; reads the margin the gate actually saw ([diag1255] gate-pass, MiB) and whether the
prime ran out of memory. Works on the diag build (gate-pass lines) only.

usage: b-sweep.py --bin BIN --model GGUF --draft DIR --out DIR [--margins 0,25,50,75,100,125,150]
       [--words 3582] [--reps 1] [--seed0 400] [--trim 0|1] [--call-trim 0|1]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import threading
import time
from pathlib import Path

GATE = Path.home() / ".cache/memra-1258/harness/same-pass-admission-gate.py"
spec = importlib.util.spec_from_file_location("gate", GATE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
MIB = 1 << 20
RE_PASS = re.compile(r"\[diag1255\] gate-pass id=(\S+) prompt=(\d+) path=(\w+) cost=(\d+)MB draft=(\d+)MB "
                     r"reserve=(\d+)MB required=(\d+)MB required_eager=(\d+)MB eager_arm=(\w+) booked=(\d+)MB "
                     r"limiting_free=(\d+)MB active=(\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--margins", default="0,25,50,75,100,125,150")
    ap.add_argument("--words", type=int, default=3582)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=400,
                    help="the first row's prompt is seed0 + 1 (b-fresh.sh: one row per boot, one prompt each)")
    ap.add_argument("--port", type=int, default=18181)
    ap.add_argument("--trim", choices=("0", "1"), default="0",
                    help="MEMRA_DIAG_TRIM_BEFORE_PRIME: drain the pool's cached bytes before each cold prime")
    ap.add_argument("--call-trim", choices=("0", "1"), default="0",
                    help="MEMRA_DIAG_CALL_TRIM (diag5): the per-call driver-headroom trim prototype")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "serve.log"
    server = g.Server(args.bin, args.model, args.draft, args.port, 32768, log)
    _popen = g.subprocess.Popen

    def popen(cmd, *a, **kw):
        env = kw.get("env")
        if env is not None and cmd == [args.bin]:
            env["MEMRA_DIAG_TRIM_BEFORE_PRIME"] = args.trim
            env["MEMRA_DIAG_CALL_TRIM"] = args.call_trim
            env.pop("MEMRA_SPEC_GATE_LOW", None)
        return _popen(cmd, *a, **kw)

    g.subprocess.Popen = popen
    rows = []
    ballast = None
    try:
        server.boot()
        g.subprocess.Popen = _popen
        # Learn the required (MiB, as the gate-pass line prints it) for this length.
        start = len(log.read_text(errors="replace").splitlines())
        server.complete(g.prose(301, args.words), 16)
        time.sleep(1.0)
        p = [m.groups() for l in g.log_lines(log, start) if (m := RE_PASS.search(l))][-1]
        required = int(p[6]) * MIB
        keeper: dict = {}
        def start_keeper() -> threading.Thread:
            t = threading.Thread(target=lambda: keeper.update(server.complete(g.keeper_prompt(), 20000)),
                                 daemon=True)
            t.start()
            time.sleep(6.0)
            return t

        # A counting keeper on DFlash decodes ~100 tok/s, so 20,000 tokens last about 3 minutes:
        # a finished keeper is replaced, and the server's stop ends the last one.
        kt = start_keeper()
        ballast = g.Ballast(args.out / "ballast.log")
        seed = args.seed0
        for rep in range(args.reps):
            for margin in [int(x) for x in args.margins.split(",")]:
                seed += 1
                if not kt.is_alive():
                    # The ballast leaves no room for a new keeper's admission: drop it first.
                    ballast.set(0)
                    kt = start_keeper()
                target = required + margin * MIB
                # Two corrections on fresh snapshots: the keeper and the last prime move the reading.
                for _ in range(2):
                    here = g.effective_free(server.fresh_metrics())
                    ballast.set(max(0, ballast.total + here - target))
                start = len(log.read_text(errors="replace").splitlines())
                r = server.complete(g.prose(seed, args.words), 16)
                time.sleep(0.5)
                lines = g.log_lines(log, start)
                passes = [m.groups() for l in lines if (m := RE_PASS.search(l))]
                first = next((q for q in passes if int(q[1]) > 1000), None)
                if first is not None and first[2] == "spec" and int(first[6]) * MIB != required:
                    # The gate learns (a fixed residual high-water after a plain pass, say):
                    # place the next primes against what it charges now.
                    print(f"required moved {required >> 20} -> {first[6]} MiB", flush=True)
                    required = int(first[6]) * MIB
                row = {
                    "rep": rep, "margin_target_mib": margin, "status": r["status"],
                    "prompt_tokens": r["prompt_tokens"],
                    "path": first[2] if first else None,
                    "margin_seen_mib": (int(first[10]) - int(first[6])) if first else None,
                    "eager": first[8] if first else None,
                    "oom_lines": sum(1 for l in lines if "out of memory" in l),
                    "trim_fires": sum(1 for l in lines if l.startswith("[diag1258] call-trim") and " fire " in l),
                    "ring_tail": [l for l in lines if l.startswith("[diag1255-ring]")][-3:],
                }
                rows.append(row)
                with (args.out / "rows.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"rep {rep} m{margin}: status {r['status']} path {row['path']} seen {row['margin_seen_mib']} "
                      f"eager {row['eager']} oom {row['oom_lines']}", flush=True)
                if not kt.is_alive():
                    print(f"keeper finished ({keeper.get('status')}, {keeper.get('completion_tokens')} tokens); "
                          "starting another", flush=True)
                    ballast.set(0)
                    kt = start_keeper()
    finally:
        if ballast is not None:
            ballast.stop()
        server.stop()


if __name__ == "__main__":
    main()
