#!/usr/bin/env python3
"""Boot cells for the spec-cost cap and admission-floor fixes (RTX 5090).

usage: boot_cells.py OUTDIR CELL:ARM:BIN [CELL:ARM:BIN ...]

CELL is one of
  orn         Ornith-1.5-35B-A3B NVFP4+Q5K with its embedded MTP head, MEMRA_CTX=4096, nothing else
              (qwen35_moe: the MoE FFN refuses the exact-16 tier, so the decode wave cap is 8)
  q36-spec    Qwen3.6-35B-A3B UD-IQ4_XS + its owntrim NextN draft, MEMRA_CTX=4096 (the same cap 8)
  q9-pinned   Qwen3.5-9B + owntrim draft, MEMRA_CTX=8192, MEMRA_SPEC_K=3: the wave probe runs and
              the spec-cost calibration is skipped, so nothing trims between them and admit-cal
  q9-default  the same without the pin (the calibration runs and trims): the floor control
ARM names the binary (base, land or fix). Each boot writes OUTDIR/<cell>-<arm>-<n>.log (every line
prefixed with seconds since spawn) and sends one short greedy chat request; one JSON row per boot
goes to OUTDIR/rows.jsonl and one CELL line to stdout and OUTDIR/cells.txt.

Pass rules: every cell must reach "[server] listening", answer on a port this process's child
owns, admit the request and log no panic. The q9 cells also need the admission calibration's
measured transient at or under the 1536 MB static floor, which is what the default boot of the
9B measures (1083 MB on the 5090 before this lane).
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
# Model root: the directory that holds the checkpoints below (MODELS=<dir>).
M = os.environ.get("MODELS", "<models>")
ORN = f"{M}/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf"
Q36 = f"{M}/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
Q36D = f"{M}/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf"
Q9 = f"{M}/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
Q9D = f"{M}/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"
PORT = int(os.environ.get("CELL_PORT", "18391"))
BASE = f"http://127.0.0.1:{PORT}"
STATIC_FLOOR_MB = 1536
ENVS = {
    "orn": ("orn", {"MEMRA_MODELS": f"orn={ORN}", "MEMRA_CTX": "4096"}),
    "q36-spec": ("q36", {"MEMRA_MODELS": f"q36={Q36}+{Q36D}", "MEMRA_CTX": "4096"}),
    "q9-pinned": ("q9", {"MEMRA_MODELS": f"q9={Q9}+{Q9D}", "MEMRA_CTX": "8192", "MEMRA_SPEC_K": "3"}),
    "q9-default": ("q9", {"MEMRA_MODELS": f"q9={Q9}+{Q9D}", "MEMRA_CTX": "8192"}),
}
PROMPT = "Explain how GPS finds a position in a few short paragraphs."
FAILS = []


def listeners():
    return subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout


def port_lines():
    return [l for l in listeners().splitlines() if re.search(rf"[:.]{PORT} ", l)]


def boot(cell, arm, binary, n):
    model, extra = ENVS[cell]
    tag = f"{cell}-{arm}-{n}"
    log_path = os.path.join(OUT, f"{tag}.log")
    busy = port_lines()
    if busy:
        raise SystemExit(f"port {PORT} is already LISTENing before {tag}: {busy}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    env.update({"MEMRA_COMPAT": "openai", "MEMRA_ADDR": f"127.0.0.1:{PORT}"})
    env.update(extra)
    if "MEMRA_GPU_LOCK" in os.environ:
        env["MEMRA_GPU_LOCK"] = os.environ["MEMRA_GPU_LOCK"]
    t0 = time.time()
    p = subprocess.Popen([binary], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                         text=True, errors="replace")
    lines = []
    logf = open(log_path, "w")

    def pump():
        for line in p.stdout:
            stamp = f"[{time.time() - t0:8.2f}] {line}"
            lines.append(stamp)
            logf.write(stamp)
            logf.flush()

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    ready = False
    while time.time() - t0 < 900:
        if p.poll() is not None:
            break
        if any("[server] listening on" in l for l in lines):
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=3).read()
                ready = True
                break
            except Exception:
                pass
        time.sleep(1)
    ready_s = time.time() - t0
    got = "not ready" if p.poll() is None else f"exited {p.returncode} before listening"
    owned = None
    if ready:
        owned = any(f"pid={p.pid}," in l for l in port_lines())
        if not owned:
            got = f"port {PORT} answers but is not owned by pid {p.pid}"
            ready = False
    if ready:
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 32,
                "temperature": 0.0, "stream": False}
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
            msg = resp["choices"][0]["message"]
            got = f"admitted ({resp['usage']['prompt_tokens']} prompt tokens, " \
                  f"{resp['usage']['completion_tokens']} completion tokens, " \
                  f"{len((msg.get('reasoning') or '') + (msg.get('content') or ''))} chars)"
        except urllib.error.HTTPError as e:
            got = f"HTTP {e.code}: {e.read()[:200]!r}"
        except Exception as e:  # a worker panic mid-request drops the connection
            got = f"request failed: {e!r}"
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    th.join(10)
    logf.close()
    text = "".join(lines)

    def first(pattern):
        m = re.search(pattern, text)
        return m.group(0).strip() if m else None

    floor = first(r"transient floor \d+MB \(static[^\]]*\]")
    measured = re.search(r"measured (\d+)MB", floor or "")
    measured = int(measured.group(1)) if measured else None
    trims = [(why, json.loads(j)) for why, j in re.findall(r"\[device-trim\] reason=(\S+) (\{[^\n]*\})", text)]
    panics = [l.strip()[:300] for l in lines if "panicked at" in l or "PANIC in the GPU worker" in l]
    row = {"cell": cell, "arm": arm, "n": n, "binary": binary, "build": first(r"build: [^\n]*"),
           "exact16": first(r"\[exact16\][^\n]*"), "cap": first(r"decode wave cap \d+[^\n]*"),
           "probe": first(r"\[wave-probe\][^\n]*"), "spec_cost": first(r"\[spec-cost\][^\n]*"),
           "floor": floor, "measured_floor_mb": measured,
           "trims": [(why, j.get("reclaimed_bytes", 0) >> 20) for why, j in trims],
           "listening": first(r"\[server\] listening on [^\n]*"), "owned": owned, "panics": panics,
           "ready_s": round(ready_s, 1), "request": got, "log": log_path}
    with open(os.path.join(OUT, "rows.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    ok = got.startswith("admitted") and not panics
    if cell.startswith("q9"):
        ok = ok and measured is not None and measured <= STATIC_FLOOR_MB
    detail = panics[0] if panics else (floor and floor[:170])
    line = f"CELL {tag} {'PASS' if ok else 'FAIL'} | {got} | {row['spec_cost'] and row['spec_cost'][:200]} | {detail}"
    print(line, flush=True)
    with open(os.path.join(OUT, "cells.txt"), "a") as f:
        f.write(line + "\n")
    if not ok:
        FAILS.append(tag)


for spec in sys.argv[2:]:
    cell, arm, binary = spec.split(":", 2)
    n = 1
    while os.path.exists(os.path.join(OUT, f"{cell}-{arm}-{n}.log")):
        n += 1
    boot(cell, arm, binary, n)
sys.exit(1 if FAILS else 0)
