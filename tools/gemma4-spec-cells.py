#!/usr/bin/env python3
"""gemma4 served-spec cells (lane/gemma-batched stage 3, 2026-08-17).

Runs against an ALREADY-BOOTED spec-armed memra-server (MEMRA_GEMMA4_SPEC=K +
MEMRA_DRAFT + ranks — the shipping config). Per rep, interleaved:

  spec-c1     — solo greedy stream requests, prose + code classes: served DECODE
                tok/s = (tokens-1)/(t_last - t_first) via SSE (prime excluded, the
                bench receipts' convention) — the served-vs-bench delta source.
  batch-c8/16 — temp-0.7 load (non-greedy => plain batched by admission law):
                aggregate tok/s from usage blocks (the flip-cell convention).
  mixed       — one long greedy spec request admitted SOLO first, then a c8 batch
                burst fired while it streams: proves scheduler coexistence and
                reports both sides' rates under contention.

Emits one JSON line per (rep, cell) to --out. Ambiguity: the caller greps the server
log for [gspec-acc] growth per spec cell; this driver tags expected spec cells.
"""
import argparse, json, threading, time, urllib.request

PROSE = "Explain how ocean currents influence regional climates, in a detailed essay."
CODE = ("Write a Python class implementing an LRU cache with get/put in O(1), "
        "then explain each method briefly.")
BATCH_PROMPT = ("Summarize the operational state of a GPU serving cluster in exactly "
                "three sentences, then list four risks. " + "The quick brown fox jumps "
                "over the lazy dog while the seasoned engineer measures throughput. " * 6)


def sse_request(base, prompt, max_tokens, temperature=0.0, seed=None):
    """Stream one chat completion; return (n_tokens, t_first, t_last, wall, text_len)."""
    body = {"model": "g4", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    n, t_first, t_last, text_len = 0, None, None, 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = ev.get("choices", [{}])[0]
            delta = ch.get("delta", {})
            if "content" in delta or "reasoning" in delta:
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
                text_len += len(delta.get("content") or delta.get("reasoning") or "")
    return n, t_first and t_first - t0, t_last and t_last - t0, time.monotonic() - t0, text_len


def decode_rate(n, t_first, t_last):
    if n < 2 or t_first is None or t_last is None or t_last <= t_first:
        return None
    return (n - 1) / (t_last - t_first)


def usage_request(base, prompt, max_tokens, temperature, seed):
    body = {"model": "g4", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
            "stream": False}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return out.get("usage", {}).get("completion_tokens", 0), time.monotonic() - t0


def batch_cell(base, conc, requests_total, max_tokens=128, cold=False):
    """temp-0.7 concurrent STREAMED load: aggregate tok/s over the wall window +
    per-request TTFT p50 (the zoofusion re-bank banks TTFT per cell).
    cold=True prepends a unique per-request preamble so every prompt misses the
    prefix cache — the fusion/q6kb lanes' cold-prompt protocol (prefill-dominated)."""
    done, lock = [], threading.Lock()
    idx = [0]

    def worker(wid):
        while True:
            with lock:
                if idx[0] >= requests_total:
                    return
                i = idx[0]
                idx[0] += 1
            prompt = (f"Case {1000 + i}: consider deployment scenario number {i * 7 + 3} "
                      f"in region {i % 11}. " + BATCH_PROMPT) if cold else BATCH_PROMPT
            n, tf, _tl, wall, _ = sse_request(base, prompt, max_tokens, 0.7, 1000 + i)
            with lock:
                done.append((n, tf, wall))

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    total = sum(n for n, _, _ in done)
    tfs = sorted(t for _, t, _ in done if t is not None)
    ttft_p50 = tfs[len(tfs) // 2] if tfs else None
    return {"agg_tok_s": total / wall, "n_ok": len(done), "wall_s": wall,
            "completion_tokens_total": total, "ttft_p50_s": ttft_p50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8183")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec-tokens", type=int, default=256)
    ap.add_argument("--plain-only", action="store_true",
                    help="kill-switch boot reference: c1 plain prose/code stream cells only")
    ap.add_argument("--spec-only", action="store_true",
                    help="run only the spec-c1 prose/code cells (decomposition probes)")
    ap.add_argument("--tag", default="",
                    help="suffix appended to cell names, e.g. @q8embd (cross-config A/B)")
    args = ap.parse_args()
    out = open(args.out, "a")

    def emit(rec):
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(rec)

    for rep in range(1, args.reps + 1):
        if args.plain_only:
            # kill-switch reference boot: same prompts, plain path at c1.
            for cls, prompt in (("prose", PROSE), ("code", CODE)):
                n, tf, tl, wall, _ = sse_request(args.base, prompt, args.spec_tokens)
                emit({"cell": f"plain-c1-{cls}", "rep": rep, "n_tokens": n,
                      "ttft_s": tf, "decode_tok_s": decode_rate(n, tf, tl), "wall_s": wall})
            continue
        # spec-c1 prose + code (solo greedy stream -> spec route by admission law)
        for cls, prompt in (("prose", PROSE), ("code", CODE)):
            n, tf, tl, wall, _ = sse_request(args.base, prompt, args.spec_tokens)
            emit({"cell": f"spec-c1-{cls}{args.tag}", "rep": rep, "n_tokens": n,
                  "ttft_s": tf, "decode_tok_s": decode_rate(n, tf, tl), "wall_s": wall})
        if args.spec_only:
            continue
        # batch reconfirm c8 + c16 (temp 0.7 -> plain batched), cached protocol
        for conc in (8, 16):
            r = batch_cell(args.base, conc, conc * 3)
            r.update({"cell": f"batch-c{conc}", "rep": rep})
            emit(r)
        # cold-prompt c8 (the fusion/q6kb protocol: prefill-dominated, unique prompts)
        r = batch_cell(args.base, 8, 24, cold=True)
        r.update({"cell": "batch-c8-cold", "rep": rep})
        emit(r)
        # mixed: spec admitted solo FIRST, then c8 batch under it
        spec_res = {}

        def spec_side():
            n, tf, tl, wall, _ = sse_request(args.base, PROSE, 512)
            spec_res.update({"n_tokens": n, "ttft_s": tf,
                             "decode_tok_s": decode_rate(n, tf, tl), "wall_s": wall})

        th = threading.Thread(target=spec_side)
        th.start()
        time.sleep(2.0)  # spec request admitted solo; batch arrives under it
        b = batch_cell(args.base, 8, 16)
        th.join()
        emit({"cell": "mixed-spec-side", "rep": rep, **spec_res})
        emit({"cell": "mixed-batch-side", "rep": rep, **b})
    out.close()


if __name__ == "__main__":
    main()
