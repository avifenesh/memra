#!/usr/bin/env python3
"""memra#1258: the boot calibration probe against a served lone cold prime of the probe's shape,
in one boot of the diag build with MEMRA_ALLOC_TRACE=1 ([dflash-alloc] phase readings inside
dspark_prime_start and around the first burst, for the probe and for the served prime alike) plus
the [diag1255] snapshots around the served prime's chunks. Idle card, no ballast.

usage: b-probe.py --bin BIN --model GGUF --draft DIR --out DIR [--words 2556] [--trim 0|1]
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

GATE = Path.home() / ".cache/memra-1255/wt/tools/same-pass-admission-gate.py"
spec = importlib.util.spec_from_file_location("gate", GATE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--words", default="2556,2556")
    ap.add_argument("--trim", choices=("0", "1"), default="0")
    ap.add_argument("--port", type=int, default=18180)
    ap.add_argument("--seeds", default="", help="prose seeds, one per --words entry (default 61 + 6i)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "serve.log"
    server = g.Server(args.bin, args.model, args.draft, args.port, 32768, log)
    _popen = g.subprocess.Popen

    def popen(cmd, *a, **kw):
        env = kw.get("env")
        if env is not None and cmd == [args.bin]:
            env["MEMRA_ALLOC_TRACE"] = "1"
            env["MEMRA_DIAG_TRIM_BEFORE_PRIME"] = args.trim
            env.pop("MEMRA_SPEC_GATE_LOW", None)
        return _popen(cmd, *a, **kw)

    g.subprocess.Popen = popen
    try:
        server.boot()
        g.subprocess.Popen = _popen
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else []
        for i, words in enumerate(int(w) for w in args.words.split(",")):
            seed = seeds[i] if seeds else 61 + 6 * i
            start = len(log.read_text(errors="replace").splitlines())
            r = server.complete(g.prose(seed, words), 64)
            time.sleep(1.0)
            print(f"seed {seed}: status {r['status']} prompt {r['prompt_tokens']}", flush=True)
            for line in g.log_lines(log, start):
                if line.startswith(("[dflash-alloc]", "[diag1255]", "[admission] request cost", "[prime-yield]")):
                    print("  " + line[:260], flush=True)
        text = log.read_text(errors="replace").splitlines()
        print("boot:", flush=True)
        for line in text:
            if line.startswith(("[admit-cal]", "[device-trim]")) or (line.startswith("[dflash-alloc]") and "[server] listening" not in line):
                print("  " + line[:300], flush=True)
            if line.startswith("[server] listening"):
                break
    finally:
        g.subprocess.Popen = _popen
        server.stop()


if __name__ == "__main__":
    main()
