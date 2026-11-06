#!/usr/bin/env python3
"""fairness_cell.py: peer first-token time beside one long cold prime (memra#1073 review item 1,
the GB10 prime chunk default 8192 vs 4096). Against a running memra-server:

  t=0       one cold streaming chat request on the long document (--long-lines, ~131k tokens)
  t=+D, +D+S, +D+2S   cold streaming peers on short documents (--peer-lines, ~2k tokens),
            each with its own nonce so nothing hits the prefix cache
Every request: temperature 0, thinking off, max_tokens 32. Rows: send time, TTFT, total,
completion tokens, finish, text sha256. After all finish: /health worker.tick_max_ms.

  fairness_cell.py OUT.jsonl --tag ARM [--long-lines 4678] [--peer-lines 73] [--peers 3]
                   [--delay 10] [--spacing 5]
stdlib only."""
import argparse, hashlib, json, os, sys, threading, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pf_long_memra import doc, KW  # noqa: E402

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8090").rstrip("/")
MODEL = os.environ.get("BENCH_MODEL", "q38")


def stream(kind, idx, lines, nonce, t_ref, rows, lock):
    body = {"model": MODEL, "messages": [{"role": "user", "content": doc(lines, nonce)}], "max_tokens": 32,
            "temperature": 0.0, "presence_penalty": 0.0, "chat_template_kwargs": KW, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = []; finish = None; usage = None; err = None; code = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            code = r.status
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if "error" in d:
                    err = d["error"]
                    continue
                usage = d.get("usage") or usage
                for ch in d.get("choices", []):
                    finish = ch.get("finish_reason") or finish
                    piece = ch.get("delta", {}).get("content") or ""
                    if piece:
                        first = first or time.time()
                        text.append(piece)
    except Exception as e:  # a failed request is a result
        err = repr(e)
    t1 = time.time()
    s = "".join(text)
    row = {"kind": kind, "idx": idx, "nonce": nonce, "lines": lines, "http": code,
           "send_s": round(t0 - t_ref, 3), "ttft_s": round((first or t1) - t0, 3), "total_s": round(t1 - t0, 3),
           "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": (usage or {}).get("completion_tokens"),
           "finish": finish, "error": err, "text_sha256": hashlib.sha256(s.encode()).hexdigest(), "text": s}
    with lock:
        rows.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--long-lines", type=int, default=4678)
    ap.add_argument("--peer-lines", type=int, default=73)
    ap.add_argument("--peers", type=int, default=3)
    ap.add_argument("--delay", type=float, default=10.0)
    ap.add_argument("--spacing", type=float, default=5.0)
    a = ap.parse_args()
    rows, lock, threads = [], threading.Lock(), []
    t_ref = time.time()
    th = threading.Thread(target=stream, args=("long", 0, a.long_lines, "fairness-cell long v1", t_ref, rows, lock))
    th.start(); threads.append(th)
    for i in range(a.peers):
        time.sleep(max(0.0, t_ref + a.delay + i * a.spacing - time.time()))
        th = threading.Thread(target=stream, args=("peer", i, a.peer_lines, f"fairness-cell peer {i} v1", t_ref, rows, lock))
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=30) as r:
            health = json.loads(r.read())
        tick = (health.get("worker") or {}).get("tick_max_ms")
    except Exception as e:
        tick = repr(e)
    with open(a.out, "a") as f:
        for row in sorted(rows, key=lambda r: (r["kind"] != "long", r["idx"])):
            row["tag"] = a.tag
            row["tick_max_ms"] = tick
            f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("tag", "kind", "idx", "http", "send_s", "ttft_s", "total_s",
                                                  "prompt_tokens", "completion_tokens", "finish", "error")}), flush=True)
    print(json.dumps({"tag": a.tag, "tick_max_ms": tick}), flush=True)


if __name__ == "__main__":
    main()
