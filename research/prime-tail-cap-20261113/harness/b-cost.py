#!/usr/bin/env python3
"""memra#1258: cost of the per-call driver-headroom trim prototype (diag5, MEMRA_DIAG_CALL_TRIM).
One boot per arm (trim 0 = main's behaviour, trim 1 = the prototype). Per boot:
  roomy: idle card, no keeper, no ballast: cold prompts of --roomy-words (one per entry); the trim
         should not fire. Reads TTFT (client, streamed), the prime wall ([prime-yield] chunk walls)
         and the [diag1258] call-trim lines.
  tight: a decoding keeper beside (warm pool), the ballast placing effective free at the gate's
         required + --margin MiB for a --tight-words prompt, --reps cold primes.
usage: b-cost.py --bin BIN --model GGUF --draft DIR --out DIR --trim 0|1
       [--roomy-words 1250,1250,10250] [--tight-words 10250] [--margin 300] [--reps 3]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

GATE = Path.home() / ".cache/memra-1255/wt/tools/same-pass-admission-gate.py"
spec = importlib.util.spec_from_file_location("gate", GATE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
MIB = 1 << 20
RE_PASS = re.compile(r"\[diag1255\] gate-pass id=(\S+) prompt=(\d+) path=(\w+) cost=(\d+)MB draft=(\d+)MB "
                     r"reserve=(\d+)MB required=(\d+)MB required_eager=(\d+)MB eager_arm=(\w+) booked=(\d+)MB "
                     r"limiting_free=(\d+)MB active=(\d+)")
RE_YIELD = re.compile(r"^\[prime-yield\] .*wall_ms=([0-9.]+)")


def stream_ttft(server, prompt: str, max_tokens: int) -> dict:
    body = {"model": "gate", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "seed": 1234,
            "timeout_ms": 240000, "stream": True}
    req = urllib.request.Request(f"{server.base}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    status = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=300) as f:
            status = f.status
            for raw in f:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in ev:
                    err = ev["error"]
                ch = ev.get("choices") or []
                if ttft is None and ch and ch[0].get("text"):
                    ttft = time.monotonic() - t0
    except urllib.error.HTTPError as e:
        status, err = e.code, e.read().decode(errors="replace")[:300]
    return {"status": status, "ttft_s": ttft, "wall_s": time.monotonic() - t0, "error": err}


def read_request(lines: list[str]) -> dict:
    p = next((m.groups() for l in lines if (m := RE_PASS.search(l)) and int(m.group(2)) > 1000), None)
    walls = [float(m.group(1)) for l in lines if (m := RE_YIELD.search(l))]
    fires = [l for l in lines if l.startswith("[diag1258] call-trim") and " fire " in l]
    skips = [l for l in lines if l.startswith("[diag1258] call-trim") and " skip " in l]
    released = sum(int(re.search(r"released=(\d+)MB", l).group(1)) for l in fires)
    trim_ms = sum(float(re.search(r" ms=([0-9.]+)", l).group(1)) for l in fires)
    return {
        "prompt_tokens": int(p[1]) if p else None, "path": p[2] if p else None,
        "margin_seen_mib": int(p[10]) - int(p[6]) if p else None, "required_mib": int(p[6]) if p else None,
        "prime_wall_ms": round(sum(walls), 3), "chunks": len(walls),
        "prefill_tok_s": round(int(p[1]) / (sum(walls) / 1000), 1) if p and walls else None,
        "trim_fires": len(fires), "trim_skips": len(skips), "released_mib": released,
        "trim_ms": round(trim_ms, 3), "oom_lines": sum(1 for l in lines if "out of memory" in l),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--trim", choices=("0", "1", "2"), required=True)
    ap.add_argument("--roomy-words", default="1250,1250,10250")
    ap.add_argument("--tight-words", type=int, default=10250)
    ap.add_argument("--margin", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--port", type=int, default=18182)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "serve.log"
    server = g.Server(args.bin, args.model, args.draft, args.port, 32768, log)
    _popen = g.subprocess.Popen

    def popen(cmd, *a, **kw):
        env = kw.get("env")
        if env is not None and cmd == [args.bin]:
            env["MEMRA_DIAG_CALL_TRIM"] = args.trim
            env.pop("MEMRA_SPEC_GATE_LOW", None)
        return _popen(cmd, *a, **kw)

    g.subprocess.Popen = popen
    ballast = None
    seed = 500

    def record(phase: str, words: int, extra: dict) -> None:
        nonlocal seed
        seed += 1
        start = len(log.read_text(errors="replace").splitlines())
        r = stream_ttft(server, g.prose(seed, words), 16)
        time.sleep(0.5)
        row = {"trim": args.trim, "phase": phase, "words": words, **extra, **r,
               **read_request(g.log_lines(log, start))}
        with (args.out / "rows.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{phase} w{words}: status {row['status']} prompt {row['prompt_tokens']} ttft {row['ttft_s']} "
              f"prime {row['prime_wall_ms']} ms ({row['chunks']} chunks, {row['prefill_tok_s']} tok/s) "
              f"fires {row['trim_fires']} released {row['released_mib']} MiB trim {row['trim_ms']} ms "
              f"oom {row['oom_lines']}", flush=True)

    try:
        server.boot()
        g.subprocess.Popen = _popen
        for words in [int(x) for x in args.roomy_words.split(",")]:
            record("roomy", words, {})
        # Learn the required for the tight length (MiB, as the gate-pass line prints it).
        start = len(log.read_text(errors="replace").splitlines())
        server.complete(g.prose(301, args.tight_words), 16)
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
        for rep in range(args.reps):
            if not kt.is_alive():
                ballast.set(0)
                kt = start_keeper()
            target = required + args.margin * MIB
            for _ in range(2):
                here = g.effective_free(server.fresh_metrics())
                ballast.set(max(0, ballast.total + here - target))
            record("tight", args.tight_words, {"rep": rep, "margin_target_mib": args.margin})
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
