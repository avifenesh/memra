#!/usr/bin/env python3
"""Boot cells for the wave-probe pool-residue fix (RTX 5090).

usage: boot_cells.py OUTDIR CELL:ARM:BIN [CELL:ARM:BIN ...]

CELL is one of
  q38-dspark     Qwen3.8-27B GGUF + DFlash2, MEMRA_DSPARK_SPEC=1, MEMRA_CTX=8192, default session cap
  q38-dspark-s4  the same with MEMRA_MAX_SESSIONS=4 (the probe is skipped: the no-probe control)
  q38-plain      Qwen3.8-27B GGUF, MEMRA_CTX=8192, default session cap, no drafter
  q9             Qwen3.5-9B, MEMRA_MAX_SESSIONS=32, MEMRA_CTX=32768 (the probe's own 5090 receipt env)
ARM names the binary (base or fix). Each boot writes OUTDIR/<cell>-<arm>-<n>.log (every line
prefixed with seconds since spawn), a 250 ms nvidia-smi csv beside it, and one short chat
request; one JSON row per boot goes to OUTDIR/rows.jsonl and one CELL line to stdout.
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
# Model root: the directory that holds the two checkpoints below (MODELS=<dir>).
M = os.environ.get("MODELS", "<models>")
Q38 = f"{M}/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
DF2 = f"{M}/models/q38-gguf/Qwen3.8-27B-DFlash2-memra"
Q9 = f"{M}/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
PORT = 18297
BASE = f"http://127.0.0.1:{PORT}"
ENVS = {
    "q38-dspark": ("q38", {"MEMRA_MODELS": f"q38={Q38}", "MEMRA_CTX": "8192", "MEMRA_DSPARK_SPEC": "1",
                           "MEMRA_DSPARK_DRAFT": DF2}),
    "q38-dspark-s4": ("q38", {"MEMRA_MODELS": f"q38={Q38}", "MEMRA_CTX": "8192", "MEMRA_DSPARK_SPEC": "1",
                              "MEMRA_DSPARK_DRAFT": DF2, "MEMRA_MAX_SESSIONS": "4"}),
    "q38-plain": ("q38", {"MEMRA_MODELS": f"q38={Q38}", "MEMRA_CTX": "8192"}),
    "q9": ("q9", {"MEMRA_MODELS": f"q9={Q9}", "MEMRA_MAX_SESSIONS": "32", "MEMRA_CTX": "32768"}),
}
PROMPT = "Explain how GPS finds a position in a few short paragraphs."
FAILS = []


def smi_apps():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def boot(cell, arm, binary, n):
    model, extra = ENVS[cell]
    tag = f"{cell}-{arm}-{n}"
    log_path = os.path.join(OUT, f"{tag}.log")
    smi_path = os.path.join(OUT, f"{tag}.smi.csv")
    apps_before = smi_apps()
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    env.update({"MEMRA_COMPAT": "openai", "MEMRA_ADDR": f"127.0.0.1:{PORT}"})
    env.update(extra)
    if "MEMRA_GPU_LOCK" in os.environ:
        env["MEMRA_GPU_LOCK"] = os.environ["MEMRA_GPU_LOCK"]
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,"
                            "clocks.sm,temperature.gpu", "--format=csv", "-lms", "250"],
                           stdout=open(smi_path, "w"), stderr=subprocess.STDOUT)
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
    apps_ready = ""
    while time.time() - t0 < 900:
        if p.poll() is not None:
            break
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=3).read()
            ready = True
            break
        except Exception:
            time.sleep(1)
    ready_s = time.time() - t0
    got = "not ready"
    if ready:
        apps_ready = smi_apps()
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
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    th.join(10)
    logf.close()
    smi.terminate()
    smi.wait()
    text = "".join(lines)

    def first(pattern):
        m = re.search(pattern, text)
        return m.group(0).strip() if m else None

    probe = first(r"\[wave-probe\][^\n]*")
    probe_t = first(r"\[ *[0-9.]+\] \[wave-probe\]")
    cap = first(r"decode wave cap \d+")
    prefix = first(r"\[prefix-cache\] (on: budget [0-9]+MB|WARNING: derived budget is 0 B)")
    boot_free = first(r"boot driver free \d+ B")
    floor = first(r"transient floor \d+MB \(static[^\]]*\]")
    trims = re.findall(r"\[device-trim\] reason=(\S+) (\{[^\n]*\})", text)
    wave_trim = [json.loads(j) for why, j in trims if why == "wave-probe"]
    peak = max((float(l.split(",")[1].split()[0]) for l in open(smi_path).read().splitlines()[1:]
                if l.count(",") >= 5 and l.split(",")[1].strip().split()[0].isdigit()), default=None)
    row = {"cell": cell, "arm": arm, "n": n, "binary": binary, "build": first(r"build: [^\n]*"),
           "probe": probe, "probe_line_at_s": probe_t, "cap": cap, "prefix": prefix, "boot_free": boot_free,
           "floor": floor, "wave_trim": wave_trim, "ready_s": round(ready_s, 1), "request": got,
           "smi_peak_mem_mib": peak, "apps_before": apps_before, "apps_at_ready": apps_ready,
           "oom_lines": len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY|out of memory", text)), "log": log_path}
    with open(os.path.join(OUT, "rows.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    ok = got.startswith("admitted")
    if cell != "q9":
        ok = ok and prefix is not None and "] on: budget" in prefix and not prefix.endswith(" 0MB")
    line = f"CELL {tag} {'PASS' if ok else 'FAIL'} | {got} | {prefix} | {floor and floor[:150]} | {probe and probe[:220]}"
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
