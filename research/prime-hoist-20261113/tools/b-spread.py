#!/usr/bin/env python3
"""memra#1258: the first prime chunk's pool draw, many samples, beside a decoding keeper on a roomy
card (no ballast, so nothing runs out of memory and every prime is a sample). Per cold prime of
--words words: need = pool reserved after chunk 1 - pool used before it (MiB, from the [diag1255]
snapshots: the chunk's blocks at its peak, whether reused or newly mapped), and the [diag1258]
call-trim lines when MEMRA_DIAG_CALL_TRIM is set (diag6). Each row also carries the sha256 of the
request's 16-token greedy text (added 2026-11-14 for the 8k and GB10 park-floor cells).
usage: b-spread.py --bin BIN --model GGUF --draft DIR --out DIR --mode 0|1|2 [--n 20] [--words 3582]
"""
from __future__ import annotations

import argparse
import hashlib
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
RE_PASS = re.compile(r"\[diag1255\] gate-pass id=(\S+) prompt=(\d+) path=(\w+)")
RE_SNAP = re.compile(r"\[diag1255\] (dspark-prime-chunk-before|dspark-prime-chunk-after|dspark-prime-done) id=(\S+): "
                     r"driver_free=(\d+)MB pool_res=(\d+)MB pool_used=(\d+)MB pool_unused=(\d+)MB")
RE_YIELD = re.compile(r"^\[prime-yield\] .*wall_ms=([0-9.]+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=("0", "1", "2"), required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--words", type=int, default=3582)
    ap.add_argument("--seed0", type=int, default=700)
    ap.add_argument("--port", type=int, default=18183)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "serve.log"
    server = g.Server(args.bin, args.model, args.draft, args.port, 32768, log)
    _popen = g.subprocess.Popen

    def popen(cmd, *a, **kw):
        env = kw.get("env")
        if env is not None and cmd == [args.bin]:
            env["MEMRA_DIAG_CALL_TRIM"] = args.mode
            env.pop("MEMRA_SPEC_GATE_LOW", None)
        return _popen(cmd, *a, **kw)

    g.subprocess.Popen = popen
    try:
        server.boot()
        g.subprocess.Popen = _popen
        keeper: dict = {}

        def start_keeper() -> threading.Thread:
            t = threading.Thread(target=lambda: keeper.update(server.complete(g.keeper_prompt(), 20000)),
                                 daemon=True)
            t.start()
            time.sleep(6.0)
            return t

        kt = start_keeper()
        for i in range(args.n):
            if not kt.is_alive():
                kt = start_keeper()
            seed = args.seed0 + i
            start = len(log.read_text(errors="replace").splitlines())
            r = server.complete(g.prose(seed, args.words), 16)
            time.sleep(0.5)
            lines = g.log_lines(log, start)
            p = next((m.groups() for l in lines if (m := RE_PASS.search(l)) and int(m.group(2)) > 1000), None)
            snaps = [(m.group(1), [int(m.group(k)) for k in range(3, 7)]) for l in lines
                     if (m := RE_SNAP.search(l)) and p and m.group(2) == p[0]]
            before = next((s for t, s in snaps if t == "dspark-prime-chunk-before"), None)
            after = next((s for t, s in snaps if t == "dspark-prime-chunk-after"), None)
            # Every call: demand = pool reserved after it - pool used before it (the bytes the call
            # held or stranded at its peak), growth = what it newly mapped.
            calls = []
            for k, (t, snap) in enumerate(snaps):
                if t == "dspark-prime-chunk-before" and k + 1 < len(snaps) and snaps[k + 1][0] != "dspark-prime-chunk-before":
                    nxt = snaps[k + 1][1]
                    calls.append({"demand_mib": nxt[1] - snap[2], "growth_mib": nxt[1] - snap[1],
                                  "unused_at_start_mib": snap[3], "driver_free_at_start_mib": snap[0]})
            walls = [float(m.group(1)) for l in lines if (m := RE_YIELD.search(l))]
            trims = [l for l in lines if l.startswith("[diag1258] call-trim")]
            row = {
                "mode": args.mode, "i": i, "seed": seed, "status": r["status"],
                "prompt_tokens": p and int(p[1]), "path": p and p[2],
                "chunk1_before": before, "chunk1_after": after,
                "need_mib": (after[1] - before[2]) if (before and after) else None,
                "growth_mib": (after[1] - before[1]) if (before and after) else None,
                "unused_at_start_mib": before[3] if before else None,
                "calls": calls,
                "prime_wall_ms": round(sum(walls), 3), "chunks": len(walls),
                "trim_lines": trims,
                "oom_lines": sum(1 for l in lines if "out of memory" in l),
                # The 16 greedy tokens' text (temperature 0), so two arms' answers can be compared.
                "completion_tokens": r.get("completion_tokens"),
                "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest() if r.get("text") is not None else None,
            }
            with (args.out / "rows.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"i {i}: status {r['status']} prompt {row['prompt_tokens']} path {row['path']} need {row['need_mib']} "
                  f"growth {row['growth_mib']} unused0 {row['unused_at_start_mib']} calls "
                  f"{[(c['demand_mib'], c['growth_mib']) for c in calls]} trims {len(trims)} "
                  f"oom {row['oom_lines']} text {(row['text_sha256'] or 'none')[:16]}", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
