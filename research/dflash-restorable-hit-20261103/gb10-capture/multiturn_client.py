#!/usr/bin/env python3
"""Multi-turn persistence of the cold-drafter conversion (GB10, GGUF DFlash2, doors on by device
default). Per session s, three raw /v1/completions turns, the gate's shape carried one turn further:
  x0: P_s served PLAIN (presence_penalty on a greedy request, max_tokens=1): a plain entry for P_s;
  t2: P_s + S_s (S longer than the drafter window), T=1 seed 1234: the veto fires on P_s's plain hit
      and the conversion re-arms a cold drafter at |P_s|;
  t3: t2's prompt + t2's text + a follow-up F_s: with the conversion capturing, t3 restores t2's
      tail-carrying entry; without it t3 hits P_s's plain entry again, is vetoed and converts again,
      re-priming S_s.
Records TTFT (streamed), decode tok/s and usage per turn.
usage: multiturn_client.py BASE MODEL SESSIONS OUT.jsonl TAG"""
import json
import sys
import time
import urllib.request

base, model, sessions, out, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]


def para(i, n=60):
    words = ("memory bandwidth decode kernel scheduler prefix cache draft tail verify accept "
             "token session window boundary restore prime chunk tensor stream").split()
    return " ".join(f"{words[(i * 7 + k * 3) % len(words)]}{(i * 31 + k) % 97}" for k in range(n))


def stream(body):
    body = dict(body, stream=True)
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, text, usage, n = None, "", None, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            v = json.loads(line[5:])
            piece = (v.get("choices") or [{}])[0].get("text") or ""
            if piece:
                n += 1
                if first is None:
                    first = time.time() - t0
            text += piece
            usage = v.get("usage") or usage
    total = time.time() - t0
    comp = (usage or {}).get("completion_tokens") or n
    dec = comp / (total - first) if first is not None and total > first else None
    return {"ttft_s": round(first, 3) if first is not None else None, "total_s": round(total, 3),
            "completion_tokens": comp, "decode_tps": round(dec, 2) if dec else None,
            "usage": usage}, text


with open(out, "a") as f:
    for s in range(sessions):
        P = f"Notes {tag}-{s}.\n" + "\n".join(para(s * 1000 + i) for i in range(70)) + "\n"
        S = "Session log.\n" + "\n".join(para(s * 1000 + 500 + i) for i in range(30)) + "\n"
        q = "\nQuestion: summarize the notes above in three sentences.\nAnswer:"
        F = "\nNow list two caveats, then one next step.\n" + "\n".join(para(s * 1000 + 900 + i) for i in range(3)) + "\n"
        rec, _ = stream({"model": model, "prompt": P, "max_tokens": 1, "temperature": 0, "presence_penalty": 1.5})
        f.write(json.dumps({"tag": tag, "session": s, "turn": "x0", **rec}) + "\n")
        rec, t2 = stream({"model": model, "prompt": P + S + q, "max_tokens": 256, "temperature": 1.0, "seed": 1234})
        f.write(json.dumps({"tag": tag, "session": s, "turn": "t2", **rec}) + "\n")
        rec, _ = stream({"model": model, "prompt": P + S + q + t2 + F, "max_tokens": 256, "temperature": 1.0, "seed": 1234})
        f.write(json.dumps({"tag": tag, "session": s, "turn": "t3", **rec}) + "\n")
        f.flush()
        print(tag, s, "done", flush=True)
