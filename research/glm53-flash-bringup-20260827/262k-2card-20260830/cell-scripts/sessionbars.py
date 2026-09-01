#!/usr/bin/env python3
"""Session bars at the deepest surviving rung (262k 2-card PINNED-RECIPE cell).

With the deep session LIVE (a fresh cold prime at the deepest rung, vendor-default
sampled = the product traffic shape, streaming, max_tokens sized so decode keeps the
session alive through the bars), add concurrent short sessions (8k-class, vendor,
max_tokens 48) and bank per-card VRAM + served/refused at each step:

  bar +1 : one short alongside the deep session          (2 live)
  bar +2 : two shorts concurrently alongside deep        (3 live)
  bar +3 : three shorts concurrently alongside deep      (4 live = MEMRA_MAX_SESSIONS)
  bar +4 : four shorts concurrently alongside deep       (5 > cap: the ADMISSION receipt -
           expected refusal shape banked, distinguishing admission from OOM)

Every bar records whether the deep stream was still decoding (deep_live) when the bar ran;
a bar that ran after deep finished is labeled and does not count as a concurrent receipt.
The deep row doubles as the cell's vendor-default sampled row at the deepest rung.
usage: sessionbars.py <outdir> <deep_chars> <short_chars> [deep_max_tokens=2048]
env: EP (default http://127.0.0.1:18600), MODEL
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

OUT = sys.argv[1]
DEEP_CHARS = int(sys.argv[2])
SHORT_CHARS = int(sys.argv[3])
DEEP_MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
EP = os.environ.get("EP", "http://127.0.0.1:18600")
MODEL = os.environ.get("MODEL", "zai/glm-5.3-flash")
CORPUS = "/root/out-262k-2c/corpus/corpus-1m.txt"
TEXT = open(CORPUS, encoding="utf-8", errors="strict").read()
os.makedirs(OUT, exist_ok=True)
ASK = ("\n\n---\nThe text above is a corpus of classic literature. In one short paragraph, "
       "name two of the works it contains and one theme they share.")
# The deep stream must stay LIVE through the bars: a short-paragraph ask would EOS in
# seconds. Long-form ask + max_tokens keeps the deep session decoding while shorts join.
DEEP_ASK = ("\n\n---\nThe text above is a corpus of classic literature. Walk through it "
            "in order and summarize every chapter or major section you can identify, one "
            "line each, until you run out of text. Do not stop early.")


def vram(label):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
    return {"label": label, "t": round(time.time() - T0, 1), "cards": out.splitlines()}


def stream(label, chars, max_tokens, mode, first_token_evt=None, done_evt=None, offset=0,
           ask=ASK):
    """mode: 'vendor' (no sampling params) or 'greedy'. offset slices a different corpus
    window so concurrent shorts are distinct real-text prompts, not clones."""
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": TEXT[offset:offset + chars] + ask}],
            "max_tokens": max_tokens,
            "reasoning_effort": "low",
            "stream": True,
            "stream_options": {"include_usage": True}}
    if mode == "greedy":
        body["temperature"] = 0.0
    req = urllib.request.Request(EP + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    row = {"label": label, "mode": mode, "chars": chars, "offset": offset,
           "max_tokens": max_tokens, "status": -1, "error": "", "usage": None,
           "prefill_s": None, "decode_tok_s": None, "wall_s": None, "output": "",
           "reasoning_head": ""}
    tfc, tlast, pieces, rpieces = None, None, [], []
    try:
        with urllib.request.urlopen(req, timeout=64800) as r:
            row["status"] = r.status
            for line in r:
                now = time.time()
                if not line.startswith(b"data:"):
                    continue
                frag = line[5:].strip()
                if not frag or frag == b"[DONE]":
                    continue
                try:
                    d = json.loads(frag)
                except Exception:
                    continue
                if d.get("usage"):
                    row["usage"] = d["usage"]
                e2 = (d.get("error") or {}).get("message")
                if e2:
                    row["error"] = e2
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    c = delta.get("content")
                    rc = delta.get("reasoning_content") or delta.get("reasoning")
                    if c or rc:
                        if tfc is None:
                            tfc = now
                            if first_token_evt:
                                first_token_evt.set()
                        tlast = now
                    if rc:
                        rpieces.append(rc)
                    if c:
                        pieces.append(c)
    except urllib.error.HTTPError as e:
        row["status"] = e.code
        row["error"] = e.read().decode(errors="replace")[:800]
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    row["wall_s"] = round(time.time() - t0, 2)
    row["output"] = "".join(pieces)[:600]
    row["reasoning_head"] = "".join(rpieces)[:200]
    pt = (row["usage"] or {}).get("prompt_tokens")
    ct = (row["usage"] or {}).get("completion_tokens")
    if tfc:
        row["prefill_s"] = round(tfc - t0, 3)
        row["prefill_tok_s"] = round(pt / (tfc - t0), 1) if pt else None
    if ct and ct > 1 and tfc and tlast and tlast > tfc:
        row["decode_tok_s"] = round((ct - 1) / (tlast - tfc), 2)
    if first_token_evt:
        first_token_evt.set()  # never leave the main thread waiting on a failed prime
    if done_evt:
        done_evt.set()
    RESULTS["rows"].append(row)
    print(f"  [{label}] status={row['status']} pt={pt} ct={ct} prefill={row['prefill_s']}s "
          f"err={row['error'][:120]!r}")
    return row


T0 = time.time()
RESULTS = {"ep": EP, "deep_chars": DEEP_CHARS, "short_chars": SHORT_CHARS,
           "deep_max_tokens": DEEP_MAXTOK, "rows": [], "vram": [], "bars": []}

RESULTS["vram"].append(vram("before-deep"))
ft, dd = threading.Event(), threading.Event()
deep_th = threading.Thread(target=stream, args=("deep-vendor", DEEP_CHARS, DEEP_MAXTOK,
                                                "vendor", ft, dd),
                           kwargs={"ask": DEEP_ASK})
deep_th.start()
print("deep prime launched; waiting for its first decoded token (prefill)...")
ft.wait()
RESULTS["vram"].append(vram("deep-first-token"))

for n in (1, 2, 3, 4):
    live_before = not dd.is_set()
    print(f"### bar +{n}: {n} short session(s) alongside deep (deep_live_before={live_before})")
    ths, evs = [], []
    for i in range(n):
        ev = threading.Event()
        th = threading.Thread(target=stream,
                              args=(f"bar{n}-short{i}", SHORT_CHARS, 48, "vendor"),
                              kwargs={"done_evt": ev, "offset": 2_000_000 + i * SHORT_CHARS})
        ths.append(th); evs.append(ev); th.start()
    for th in ths:
        th.join()
    live_after = not dd.is_set()
    RESULTS["vram"].append(vram(f"after-bar+{n}"))
    RESULTS["bars"].append({"bar": n, "deep_live_before": live_before,
                            "deep_live_after": live_after})
    if dd.is_set() and n < 4:
        print(f"  NOTE: deep stream finished during/before bar +{n}; later bars are "
              f"labeled deep_live=False and are not concurrent receipts")

deep_th.join()
RESULTS["vram"].append(vram("after-deep-done"))
json.dump(RESULTS, open(f"{OUT}/sessionbars.json", "w"), indent=1)
print("banked", f"{OUT}/sessionbars.json")
