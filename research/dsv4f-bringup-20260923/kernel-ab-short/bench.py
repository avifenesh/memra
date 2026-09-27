#!/usr/bin/env python3
"""DSv4F serving cell: c workers x n requests each, streaming, usage-authoritative counts.

Per request: TTFT, E2E, TPOT, chunk ITLs, completion/prompt tokens, sha256 of the text.
Summary: p50/p95/p99 of TTFT/E2E/TPOT/ITL, request/s, aggregate completion tok/s, and
single-stream decode tok/s (1/TPOT median). One JSON summary line to --out, per-request
rows to --out.req.jsonl.
"""
import argparse, hashlib, json, statistics, threading, time, urllib.request

PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration string such as P3DT4H12M into total seconds. Include input validation and three unit tests.",
    "Explain how a write-ahead log gives a database crash consistency. Cover fsync ordering, checkpoints, and what happens on recovery, in about 300 words.",
    "You are a support agent. A customer says their invoice shows two charges for the same month. Draft a reply that asks for the two transaction ids and explains the refund timeline.",
    "Compare breadth-first and depth-first search for finding a shortest path in an unweighted graph. Give the complexity of each and one case where each is the better choice.",
    "Summarize the causes of the 1929 stock market crash in five bullet points, then give one lesson a modern retail investor could take from it.",
    "Translate this sentence into French, Spanish and German, then explain one grammatical difference between the three: 'The engineers finished the bridge two weeks early because the weather held.'",
    "Write a SQL query that returns, for each customer, their three most recent orders with the order total, using a window function. Then explain how an index would speed it up.",
    "Plan a three-day itinerary for a first visit to Kyoto in autumn, balancing temples, food and one day trip. Keep each day to five stops.",
]


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    k = (len(v) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


WORDS = ("amber basin cedar delta ember fjord glacier harbor island juniper kestrel lagoon meadow "
         "nectar orchard prairie quartz river summit tundra upland valley willow yarrow zephyr "
         "anchor beacon canyon dune estuary forest grove heath inlet jetty knoll ledge marsh "
         "narrows oasis peak quarry ridge shoal terrace uplift vista wharf").split()


def context(idx, n):
    """Deterministic n-word document, distinct per request and length so no prefix is shared."""
    s, out = 0x9E3779B9 ^ ((idx * 2654435761 + n * 40503) & 0xFFFFFFFF), []
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(WORDS[(s >> 8) % len(WORDS)] + ("." if i % 17 == 16 else ""))
    return " ".join(out)


def one(a, idx, seed):
    content = PROMPTS[idx % len(PROMPTS)]
    if a.context_words:
        content = ("Field notes follow.\n\n" + context(idx, a.context_words) +
                   "\n\nIgnore the notes above unless they help. " + content)
    body = {"model": a.model, "max_tokens": a.max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": content}]}
    if a.greedy:
        body["temperature"] = 0.0
    else:
        body["seed"] = seed
    if a.ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % a.port, data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer " + a.key})
    t0 = time.time()
    first, last, stamps, text, usage, finish = None, None, [], [], None, None
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as resp:
            for line in resp:
                line = line.decode("utf-8", "replace")
                if not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    d = ch.get("delta") or {}
                    piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
                    if piece:
                        now = time.time()
                        if first is None:
                            first = now
                        stamps.append(now)
                        text.append(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        end = time.time()
    except Exception as exc:
        return {"idx": idx, "error": str(exc)[:300], "e2e": time.time() - t0}
    txt = "".join(text)
    ct = (usage or {}).get("completion_tokens", 0)
    r = {"idx": idx, "seed": seed, "ttft": (first - t0) if first else None, "e2e": end - t0,
         "completion_tokens": ct, "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
         "finish": finish, "sha": hashlib.sha256(txt.encode()).hexdigest()[:16],
         "itl": [b - a_ for a_, b in zip(stamps, stamps[1:])], "t0": t0, "t_end": end}
    if first and ct > 1:
        r["tpot"] = (end - first) / (ct - 1)
    if a.keep_text:
        r["text"] = txt
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default="dsv4f")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--n", type=int, default=4, help="requests per worker")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--keep-text", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--label", default="")
    ap.add_argument("--context-words", type=int, default=0,
                    help="prepend a deterministic per-request document of this many words")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, lock = [], threading.Lock()

    def worker(w):
        for j in range(a.n):
            idx = w * a.n + j
            r = one(a, idx, 1000 + idx)
            r["worker"] = w
            with lock:
                rows.append(r)

    t0 = time.time()
    th = [threading.Thread(target=worker, args=(w,)) for w in range(a.conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    ok = [r for r in rows if "error" not in r]
    ms = lambda v: [x * 1000 for x in v if x is not None]
    ttft, e2e = ms(r.get("ttft") for r in ok), ms(r["e2e"] for r in ok)
    tpot = ms(r.get("tpot") for r in ok)
    itl = ms(x for r in ok for x in r["itl"])
    gen = sum(r["completion_tokens"] for r in ok)
    s = {"label": a.label, "conc": a.conc, "n_req": len(rows), "n_ok": len(ok),
         "n_err": len(rows) - len(ok), "greedy": a.greedy, "max_tokens": a.max_tokens,
         "wall_s": wall, "completion_tokens": gen,
         "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
         "agg_tok_s": gen / wall if wall else 0, "req_s": len(ok) / wall if wall else 0,
         "decode_tok_s_p50": 1000.0 / pct(tpot, 50) if tpot else None,
         "finish": sorted(set(str(r.get("finish")) for r in ok)),
         "shas": [r["sha"] for r in sorted(ok, key=lambda r: r["idx"])]}
    for name, v in (("ttft", ttft), ("e2e", e2e), ("tpot", tpot), ("itl", itl)):
        for p in (50, 95, 99):
            s["%s_ms_p%d" % (name, p)] = pct(v, p)
    with open(a.out, "a") as f:
        f.write(json.dumps(s) + "\n")
    with open(a.out + ".req.jsonl", "a") as f:
        for r in sorted(rows, key=lambda r: r["idx"]):
            r["label"] = a.label
            f.write(json.dumps(r) + "\n")
    print("CELL %s c=%d ok=%d/%d agg=%.2f tok/s decode_p50=%s tok/s ttft_p50=%s ms tpot_p50=%s ms errs=%s"
          % (a.label, a.conc, len(ok), len(rows), s["agg_tok_s"],
             "%.2f" % s["decode_tok_s_p50"] if s["decode_tok_s_p50"] else None,
             "%.0f" % s["ttft_ms_p50"] if s["ttft_ms_p50"] else None,
             "%.2f" % s["tpot_ms_p50"] if s["tpot_ms_p50"] else None,
             [r["error"][:80] for r in rows if "error" in r][:3]), flush=True)


if __name__ == "__main__":
    main()
