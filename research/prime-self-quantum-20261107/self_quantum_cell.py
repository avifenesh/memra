#!/usr/bin/env python3
"""Served cell for memra#1085: a row's own pending prime capped its first speculative burst.

RTX 5090, Qwen3.8-27B NVFP4 + Q5_K GGUF with the DFlash2 drafter, context 8,192, two sessions,
prefix cache at its derived size, the restore switch at the device default (OFF on the 5090).
Since memra#1083 a cold DFlash prime of an off-grid prompt is cut on the prime grid, so it runs
as two walker chunks: it yields once (`[prime-yield] count=1`), and on the tick its last chunk
runs the row's own pending prime used to make it owe itself a peer quantum, capping its first
speculative step to one round (`[dspark-acc] ... burst=1/7`).

Each boot serves one short warm-up request (not scored), then PROMPTS distinct cold chat
prompts (about 3,000 tokens, off the 32-token grid, no shared prefix), one at a time, greedy,
96 tokens, thinking off, streamed: TTFT = first content chunk, total = end of stream. From the
server log, per request: the prime-yield count and the first `[dspark-acc]` burst.

usage: self_quantum_cell.py OUT BIN_BASE BIN_FIX [ORDER]
  ORDER: comma list of arms, one boot each, default base,fix,fix,base,base,fix,fix,base
Writes OUT/server-<n>-<arm>.log, OUT/runs.jsonl, OUT/summary.json, OUT/cells.txt; exit 1 on a
failed check (every prompt's greedy text byte-equal across all boots of both arms; every scored
request split its prime; the base's first burst capped at one round, the fix's not).
"""
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

OUT, BIN_BASE, BIN_FIX = sys.argv[1], sys.argv[2], sys.argv[3]
ORDER = (sys.argv[4] if len(sys.argv) > 4 else "base,fix,fix,base,base,fix,fix,base").split(",")
os.makedirs(OUT, exist_ok=True)
PORT = int(os.environ.get("CELL_PORT", "18173"))
PROMPTS = int(os.environ.get("CELL_PROMPTS", "5"))
BASE_URL = f"http://127.0.0.1:{PORT}"
Q38 = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
DF2 = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-DFlash2-memra"
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={Q38}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2", "MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": DF2}
FAILS = []


def system(k):
    # A distinct first sentence per prompt, so no prompt shares a cacheable prefix with another.
    return (f"Reviewer {k} of {PROMPTS}: you are checking batch {k * 7 + 3} of an inference "
            f"engine's serving notes. The operator pasted the notes below.\n\n"
            + "\n".join(
                f"Note {i}: the scheduler admits a request only when its KV booking fits the device "
                f"budget; a parked session keeps its prefix entry, and a later turn that extends the "
                f"same prompt restores that entry and primes only the new suffix. Item {i} of "
                f"checklist {k} asks whether the restored state is byte-identical to a cold prime."
                for i in range(1, 41)))


def body(k, max_tokens=96):
    return {"model": "q", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system(k)},
                         {"role": "user", "content": "Summarize the notes in three sentences."}]}


def cell(name, ok, why=""):
    line = f"CELL {name} {'PASS' if ok else 'FAIL'} {why}".rstrip()
    print(line, flush=True)
    with open(os.path.join(OUT, "cells.txt"), "a") as f:
        f.write(line + "\n")
    if not ok:
        FAILS.append(name)


def port_free():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


class Server:
    def __init__(self, tag, binary):
        if not port_free():
            raise SystemExit(f"port {PORT} already has a listener; refusing to boot {tag}")
        self.log = os.path.join(OUT, f"server-{tag}.log")
        e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        e.update(ENV)
        self.p = subprocess.Popen([binary], env=e, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"server {tag} exited {self.p.returncode}; see {self.log}")
            if "[server] listening" in self.text():
                try:
                    urllib.request.urlopen(BASE_URL + "/v1/models", timeout=3).read()
                    return
                except Exception:
                    pass
            time.sleep(2)
        self.stop()
        raise SystemExit(f"server {tag} not ready in 900 s")

    def text(self):
        return open(self.log, errors="replace").read()

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()


def streamed(b):
    """-> (text, ttft_s, total_s, usage). TTFT = first chunk carrying content."""
    b = dict(b, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE_URL + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    ttft, parts, usage = None, [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
    return "".join(parts), ttft, time.perf_counter() - t0, usage


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def run(n, arm):
    binary = BIN_BASE if arm == "base" else BIN_FIX
    s = Server(f"{n}-{arm}", binary)
    recs = []
    try:
        streamed(dict(body(0, 8), messages=[{"role": "user", "content": "Say hello."}]))  # warm-up
        for k in range(1, PROMPTS + 1):
            mark = len(s.text())
            text, ttft, total, usage = streamed(body(k))
            time.sleep(0.5)  # let the request's last log lines land
            seg = s.text()[mark:]
            yields = [int(m.group(1)) for m in re.finditer(r"\[prime-yield\] count=(\d+)", seg)]
            bursts = re.findall(r"\[dspark-acc\] ctx=(\d+) burst=(\d+)/(\d+)", seg)
            first = bursts[0] if bursts else None
            rec = {"boot": n, "arm": arm, "prompt": k, "prompt_tokens": (usage or {}).get("prompt_tokens"),
                   "completion_tokens": (usage or {}).get("completion_tokens"),
                   "ttft_s": round(ttft, 4), "total_s": round(total, 4), "text_sha": sha(text),
                   "prime_yields": yields, "first_burst": [int(x) for x in first[1:]] if first else None,
                   "bursts": len(bursts)}
            recs.append(rec)
            print(json.dumps(rec), flush=True)
            with open(os.path.join(OUT, "runs.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
    finally:
        s.stop()
    return recs


def main():
    allr = []
    for n, arm in enumerate(ORDER, 1):
        allr += run(n, arm)
    by = {a: [r for r in allr if r["arm"] == a] for a in ("base", "fix")}
    for k in range(1, PROMPTS + 1):
        shas = {r["text_sha"] for r in allr if r["prompt"] == k}
        cell(f"prompt{k}-bytes-equal", len(shas) == 1, f"sha {sorted(shas)}")
    cell("every-prime-split", all(r["prime_yields"] == [1] for r in allr),
         str(sorted({tuple(r["prime_yields"]) for r in allr})))
    cell("base-first-burst-capped", all(r["first_burst"] and r["first_burst"][0] <= 1 and r["first_burst"][1] <= 8
                                        for r in by["base"]), str([r["first_burst"] for r in by["base"]]))
    cell("fix-first-burst-full", all(r["first_burst"] and r["first_burst"][1] > 8 for r in by["fix"]),
         str([r["first_burst"] for r in by["fix"]]))
    summary = {}
    for a, rs in by.items():
        summary[a] = {
            "n": len(rs),
            "ttft_median_s": round(statistics.median(r["ttft_s"] for r in rs), 4),
            "total_median_s": round(statistics.median(r["total_s"] for r in rs), 4),
            "total_mean_s": round(statistics.fmean(r["total_s"] for r in rs), 4),
            "first_burst_accepted_median": statistics.median(r["first_burst"][0] for r in rs if r["first_burst"]),
            "first_burst_drafted_median": statistics.median(r["first_burst"][1] for r in rs if r["first_burst"]),
            "prompt_tokens": sorted({r["prompt_tokens"] for r in rs}),
        }
        per_prompt = {}
        for k in range(1, PROMPTS + 1):
            per_prompt[k] = round(statistics.median(r["total_s"] for r in rs if r["prompt"] == k), 4)
        summary[a]["total_median_by_prompt_s"] = per_prompt
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    sys.exit(1 if FAILS else 0)


main()
