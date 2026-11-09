#!/usr/bin/env python3
"""Would cold sampled long prompts do better at K=2 than K=3 on the RTX 5090? (memra#1093 follow-up)

The spec K table gives a cold prompt of 1,024 tokens or more the cold-long rung, K=3. A sampled
resume of a long conversation took K=2 until memra#1093 (#1141) gave it its cold twin's K=3, for
about 3% of its decode rate. This asks the other way round: would every sampled long request,
cold ones included, be faster at K=2? Measurement only; no default moves without a PRO 6000 A/B too.

Arms, one binary: `k3` the device default; `k2` MEMRA_SPEC_K_COLD_LONG=2 (the existing measurement
door; it moves the cold-long rung only). Boots alternate k3 k2 k2 k3, three times (both orders).
Every boot serves the same ten cold prompts, one at a time, each distinct from its first words (no
prefix-cache entry serves another), answers capped at 400 tokens to requests for about 300 words,
thinking off, streamed:

  t1-long    six T=1 prompts (seeds fixed per prompt): four of about 1,400 tokens, two of about 4,000
  greedy-long two greedy prompts of about 1,400 tokens: the output must match across arms (K does
             not move a greedy text), so only speed moves
  t1-short   two T=1 prompts of about 200 tokens: the cold-short rung, K=3 on both arms, the noise
             reference

Per request: the `[spec-k]` line (K and rung), TTFT (first content chunk), decode rate (completion
tokens over the time after TTFT), inter-chunk gaps (ITL), acceptance and the text's sha. GPU
telemetry at 250 ms per boot (nvidia-smi). Yields to a gpulock waiter whose reason says "landing"
between boots (a YIELD file in OUT does the same).

usage: cold_long_ab.py OUT BINARY [ORDER]
  BOOT_START=n numbers the boots from n (a driver runs one boot per lease and calls this once per
  boot); ORDER=- runs no boot and rebuilds summary.json from every row in OUT/runs.jsonl.
Writes OUT/server-<n>-<arm>.log, OUT/telemetry-<n>-<arm>.csv, OUT/runs.jsonl, OUT/summary.json.
"""
import hashlib
import json
import os
import pathlib
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BIN = sys.argv[2]
ORDER = (sys.argv[3] if len(sys.argv) > 3 else
         "k3,k2,k2,k3,k3,k2,k2,k3,k3,k2,k2,k3").split(",")
if ORDER == ["-"]:
    ORDER = []
BOOT_START = int(os.environ.get("BOOT_START", "1"))
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
ARMS = {"k3": {}, "k2": {"MEMRA_SPEC_K_COLD_LONG": "2"}}
KLINE = r"\[spec-k\] model=\"q\" tenant=\"[^\"]*\" K=(\d+) source=(\S+) prompt=(\d+) cached=(\d+)"


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic, n):
    return f"{topic.capitalize()} field notes. {notes(topic, n)}\n\nWrite about 300 words on what these notes say about {topic}."


def short_prompt(topic):
    return f"{topic.capitalize()} in brief: write about 300 words on how a {topic} survey is planned and run."


PROMPTS = [  # (id, class, content, sampling)
    ("rivers-1k", "t1-long", long_prompt("rivers", 45), {"temperature": 1.0, "seed": 101}),
    ("bridges-1k", "t1-long", long_prompt("bridges", 45), {"temperature": 1.0, "seed": 102}),
    ("glaciers-1k", "t1-long", long_prompt("glaciers", 45), {"temperature": 1.0, "seed": 103}),
    ("harbors-1k", "t1-long", long_prompt("harbors", 45), {"temperature": 1.0, "seed": 104}),
    ("volcanoes-4k", "t1-long", long_prompt("volcanoes", 130), {"temperature": 1.0, "seed": 105}),
    ("canyons-4k", "t1-long", long_prompt("canyons", 130), {"temperature": 1.0, "seed": 106}),
    ("islands-1k", "greedy-long", long_prompt("islands", 45), {"temperature": 0}),
    ("meadows-1k", "greedy-long", long_prompt("meadows", 45), {"temperature": 0}),
    ("comets-short", "t1-short", short_prompt("comet"), {"temperature": 1.0, "seed": 107}),
    ("tides-short", "t1-short", short_prompt("tide"), {"temperature": 1.0, "seed": 108}),
]


# PROMPT_SET=long (run 2, the follow-up on run 1's 4k lean): four T=1 prompts of about 4,000
# tokens, two of about 5,900, one greedy 4k control and the same short reference.
if os.environ.get("PROMPT_SET") == "long":
    PROMPTS = [
        ("rivers-4k", "t1-long", long_prompt("rivers", 130), {"temperature": 1.0, "seed": 201}),
        ("bridges-4k", "t1-long", long_prompt("bridges", 130), {"temperature": 1.0, "seed": 202}),
        ("glaciers-4k", "t1-long", long_prompt("glaciers", 130), {"temperature": 1.0, "seed": 203}),
        ("harbors-4k", "t1-long", long_prompt("harbors", 130), {"temperature": 1.0, "seed": 204}),
        ("volcanoes-6k", "t1-long", long_prompt("volcanoes", 200), {"temperature": 1.0, "seed": 205}),
        ("canyons-6k", "t1-long", long_prompt("canyons", 200), {"temperature": 1.0, "seed": 206}),
        ("islands-4k", "greedy-long", long_prompt("islands", 130), {"temperature": 0}),
        ("comets-short", "t1-short", short_prompt("comet"), {"temperature": 1.0, "seed": 107}),
        ("tides-short", "t1-short", short_prompt("tide"), {"temperature": 1.0, "seed": 108}),
    ]


def boot(log, env):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(env)
    p = subprocess.Popen([BIN], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise SystemExit(f"server exited {p.returncode}; see {log}")
        try:
            urllib.request.urlopen(URL + "/v1/models", timeout=3).read()
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise SystemExit("server not ready in 900 s")


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def stream(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times, text, usage = [], [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            obj = json.loads(line[5:])
            usage = obj.get("usage") or usage
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    times.append(time.perf_counter() - t0)
                    text.append(piece)
    return times, time.perf_counter() - t0, "".join(text), usage


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def landing_waiting():
    try:
        out = subprocess.run(["gpulock", "status"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    return any("waiting:" in ln and "landing" in ln.lower() for ln in out.splitlines())


rows = []
for n, arm in enumerate(ORDER, BOOT_START):
    if (OUT / "YIELD").exists() or landing_waiting():
        print(f"yield before boot {n} ({arm}): landing waiter or YIELD file", flush=True)
        (OUT / "YIELDED").write_text(f"before boot {n} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        break
    log = OUT / f"server-{n}-{arm}.log"
    tel = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
                            "--format=csv", "-lms", "250"], stdout=open(OUT / f"telemetry-{n}-{arm}.csv", "w"),
                           stderr=subprocess.DEVNULL)
    p = boot(log, dict(COMMON, **ARMS[arm]))
    try:
        for pid, cls, content, samp in PROMPTS:
            body = {"model": "q", "messages": [{"role": "user", "content": content}], "max_tokens": 400,
                    "chat_template_kwargs": {"enable_thinking": False}, **samp}
            before = log.stat().st_size
            times, total, text, usage = stream(body)
            time.sleep(0.3)
            with open(log, errors="replace") as f:
                f.seek(before)
                seg = f.read()
            k = re.findall(KLINE, seg)
            toks = (usage or {}).get("completion_tokens") or 0
            ttft = times[0] if times else None
            gaps = [b - a for a, b in zip(times, times[1:])]
            r = {"boot": n, "arm": arm, "prompt": pid, "class": cls, "k": k[0] if k else None,
                 "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                 "completion_tokens": toks, "ttft_s": round(ttft, 4) if ttft else None, "total_s": round(total, 4),
                 "decode_tok_s": round(toks / (total - ttft), 2) if ttft and total > ttft and toks else None,
                 "itl_ms": {"p50": round(1e3 * pct(gaps, 0.5), 2), "p95": round(1e3 * pct(gaps, 0.95), 2),
                            "p99": round(1e3 * pct(gaps, 0.99), 2)} if gaps else None,
                 "acceptance": ((usage or {}).get("spec") or {}).get("acceptance_rate"),
                 "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}
            rows.append(r)
            with open(OUT / "runs.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            print(arm, n, pid, r["k"], r["ttft_s"], r["decode_tok_s"], r["completion_tokens"], flush=True)
    finally:
        stop(p)
        tel.terminate()

rows = [json.loads(l) for l in open(OUT / "runs.jsonl")] if (OUT / "runs.jsonl").exists() else rows
summary = {"boots": sorted({(r["boot"], r["arm"]) for r in rows})}
for cls in ("t1-long", "greedy-long", "t1-short"):
    for sub in (("all", None), ("1k", "-1k"), ("4k", "-4k")):
        name, tag = sub
        for arm in ARMS:
            sel = [r for r in rows if r["arm"] == arm and r["class"] == cls and (tag is None or r["prompt"].endswith(tag))]
            if not sel:
                continue
            rates = [r["decode_tok_s"] for r in sel if r["decode_tok_s"]]
            ttfts = [r["ttft_s"] for r in sel if r["ttft_s"]]
            itl50 = [r["itl_ms"]["p50"] for r in sel if r["itl_ms"]]
            itl95 = [r["itl_ms"]["p95"] for r in sel if r["itl_ms"]]
            itl99 = [r["itl_ms"]["p99"] for r in sel if r["itl_ms"]]
            toks = sum(r["completion_tokens"] for r in sel)
            dec = sum(r["total_s"] - r["ttft_s"] for r in sel if r["ttft_s"])
            summary[f"{cls}/{name}/{arm}"] = {
                "n": len(sel), "k": sorted({f"{r['k'][0]}/{r['k'][1]}" for r in sel if r["k"]}),
                "ttft_s": {"p50": pct(ttfts, 0.5), "p95": pct(ttfts, 0.95), "min": min(ttfts), "max": max(ttfts)},
                "decode_tok_s": {"median": statistics.median(rates), "min": min(rates), "max": max(rates),
                                 "pooled": round(toks / dec, 2) if dec else None},
                "itl_ms_median_of": {"p50": statistics.median(itl50), "p95": statistics.median(itl95),
                                     "p99": statistics.median(itl99)},
                "acceptance_median": statistics.median(r["acceptance"] for r in sel if r["acceptance"] is not None),
                "completion_tokens": toks}
# Paired per prompt: the k2 / k3 ratio of each prompt's median decode rate.
pairs = {}
for pid, cls, _, _ in PROMPTS:
    m = {}
    for arm in ARMS:
        v = [r["decode_tok_s"] for r in rows if r["arm"] == arm and r["prompt"] == pid and r["decode_tok_s"]]
        m[arm] = statistics.median(v) if v else None
    shas = {arm: sorted({r["sha"] for r in rows if r["arm"] == arm and r["prompt"] == pid}) for arm in ARMS}
    pairs[pid] = {"class": cls, "k3": m["k3"], "k2": m["k2"],
                  "k2_over_k3": round(m["k2"] / m["k3"], 4) if m["k2"] and m["k3"] else None, "shas": shas}
summary["paired"] = pairs
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
