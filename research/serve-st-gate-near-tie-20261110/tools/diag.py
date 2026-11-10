#!/usr/bin/env python3
"""memra#1181: where do the CLI and the server part ways on the synthetic block-128 FP8
Qwen3-1.7B directory, and at what margin.

usage: diag.py BIN_DIR CHECKPOINT_DIR OUT_DIR     (run under a GPU lease)

Arms, all greedy, the gate's prompt, 64 tokens:
  cli            run-gen (tokenwise prime, eager decode)
  srv            server, MEMRA_SERVE_SPEC=0 (batched prime, batched B=1 decode): the gate's arm
  srv-b1fast     + MEMRA_SERVE_B1FAST=1 (batched prime, eager decode)
  srv-twprime    + MEMRA_CONFIDENCE_TRACE (tokenwise prime, batched B=1 decode)
  srv-both       both doors (tokenwise prime, eager decode: the CLI's program in the server)
Logits at the diverging step: run-gen teacher-forced on its own ids (the CLI program), and
verify-shape-bisect with the tile prime and with MEMRA_PRIME_TOKENWISE (eager decode in both).
"""
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import time
import urllib.request

BIN, ST, OUT = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
PROMPT = "What is the capital of France? Answer in one short sentence."
NGEN = 64
PORT = 18181
BASE = f"http://127.0.0.1:{PORT}"


def run(name, cmd, env):
    log = OUT / f"{name}.log"
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"{name}: exit {rc}", flush=True)
    return log.read_text(errors="replace")


def ids_of(text):
    lines = [l for l in text.splitlines() if l.startswith("tokens: ")]
    return [int(x) for x in re.findall(r"\d+", lines[-1])] if lines else []


def served(name, extra):
    log = open(OUT / f"{name}.server.log", "w")
    env = {**os.environ, "MEMRA_SERVE_SPEC": "0", "MEMRA_MODELS": f"st={ST}", "MEMRA_ADDR": f"127.0.0.1:{PORT}", **extra}
    p = subprocess.Popen([str(BIN / "memra-server")], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"{BASE}/health", timeout=2).read()
                break
            except Exception:
                if p.poll() is not None:
                    print(f"{name}: server exited {p.returncode}", flush=True)
                    return []
                time.sleep(1)
        body = json.dumps({"model": "st", "prompt": PROMPT, "chat": True, "max_tokens": NGEN, "temperature": 0}).encode()
        req = urllib.request.Request(f"{BASE}/v1/completions", data=body, headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=300).read())
        (OUT / f"{name}.response.json").write_text(json.dumps(out))
        return out.get("tokens") or []
    finally:
        p.terminate()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


cli = ids_of(run("cli", [str(BIN / "run-gen"), ST, "--prompt", PROMPT], {"MEMRA_CHAT": "1", "MEMRA_NGEN": str(NGEN)}))
print("cli ids:", cli, flush=True)
arms = {"srv": {}, "srv-b1fast": {"MEMRA_SERVE_B1FAST": "1"},
        "srv-twprime": {"MEMRA_CONFIDENCE_TRACE": str(OUT / "srv-twprime.trace.jsonl")},
        "srv-both": {"MEMRA_SERVE_B1FAST": "1", "MEMRA_CONFIDENCE_TRACE": str(OUT / "srv-both.trace.jsonl")}}
got = {"cli": cli}
for name, extra in arms.items():
    got[name] = served(name, extra)
    d = first_diff(cli, got[name])
    print(f"{name}: {len(got[name])} ids, first difference from cli at {d}"
          + (f": cli={cli[d:d + 5]} {name}={got[name][d:d + 5]}" if d is not None else ""), flush=True)
(OUT / "ids.json").write_text(json.dumps(got))

div = first_diff(cli, got["srv"])
if div is None:
    print("cli and srv agree on all ids: nothing to probe")
    sys.exit(0)
a, b = cli[div], got["srv"][div]
tape = OUT / "cli.ids"
tape.write_text(" ".join(map(str, cli)))
for g in (div - 1, div, div + 1):
    if g < 0 or g >= NGEN:
        continue
    f32 = OUT / f"cli-logits-{g}.f32"
    run(f"cli-forced-{g}", [str(BIN / "run-gen"), ST, "--prompt", PROMPT],
        {"MEMRA_CHAT": "1", "MEMRA_NGEN": str(NGEN), "MEMRA_FORCE_TOKENS_FILE": str(tape),
         "MEMRA_FORCE_LOGITS_AT": str(g), "MEMRA_FORCE_LOGITS_FILE": str(f32)})
    if f32.exists():
        raw = f32.read_bytes()
        row = struct.unpack(f"<{len(raw) // 4}f", raw)
        top = sorted(range(len(row)), key=lambda i: -row[i])[:4]
        print(f"cli program, step {g}: top " + ", ".join(f"{i}={row[i]:.4f}" for i in top)
              + f"; gap top1-top2 {row[top[0]] - row[top[1]]:.4f}; logit[{a}]={row[a]:.4f} logit[{b}]={row[b]:.4f} "
              f"(difference {row[a] - row[b]:+.4f})", flush=True)
prefix = [str(t) for t in cli[:div - 1]]
probe = str(cli[div - 1])
for name, extra in (("bisect-tile-prime", {}), ("bisect-tokenwise-prime", {"MEMRA_PRIME_TOKENWISE": "1"})):
    text = run(name, [str(BIN / "verify_shape_bisect"), ST, *prefix, "--", probe, str(a), str(b)],
               {"MEMRA_PROMPT": PROMPT, "MEMRA_CHAT": "1", **extra})
    for line in text.splitlines():
        if line.startswith(("loaded", "production B=1", "verify T=1")) or "watch" in line:
            print(f"  {name}: {line[:300]}", flush=True)
print("DONE", flush=True)
