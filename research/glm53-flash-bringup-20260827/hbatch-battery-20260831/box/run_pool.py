#!/usr/bin/env python3
"""hbatch-battery pool driver (glm5 hyper-batch OFF vs ON, box stage). stdlib only.

Derived from 3way-decision-20260830/box/run_pool.py (same pools, same tape definition, same
streamed estimator) so rows are directly comparable to the banked 35.4 c=1 and 30.4 c=4
receipts. Added here: `--picks` (explicit mixed-pool index list), per-row burst timestamps,
`solo` (sequential tapes for the concurrent-vs-solo byte-identity bar), per-burst
distribution stats (per-session tok/s p50/p95, TTFT p50/max), and the decode-window
aggregate estimator alongside the burst estimator.

Subcommands:
  sample  --out DIR                        fresh-boot output sample (greedy 64, decode d00)
  timed   --out DIR                        c=1 baseline cell (flip-battery shape): streamed
           greedy decode pool (TTFT + decode tok/s) + l3 rows + deep TTFT + ONE vendor row
  conc    --out DIR --n N [--picks i,j,..] c=N concurrent streamed rows, one burst
  solo    --out DIR --picks i,j,..         same picks run SEQUENTIALLY (solo tapes)
  twin    --out DIR                        8-turn larger-prompt conversation, vendor mode
  compare --a DIR --b DIR                  byte-identity over shared *.txt tapes
"""
import argparse, hashlib, json, os, statistics, sys, threading, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
DECODE_POOL = "/root/memra/research/glm53-flash-bringup-20260827/decode-attribution-receipts/prompts.json"
L3_POOL = "/root/l3-ab-prompts.json"

# Deterministic mixed-pool pick lists per ladder rung (indices into load_pool("both"):
# 0..9 = decode pool d00..d09, 10..13 = l3 WARM/A4630/B5550/C6470).
# c=4 is BYTE-FOR-BYTE the 3way cell-5 pick list (pool[0], pool[6], pool[3], pool[11]) so the
# aggregate is directly comparable to the banked 30.4/30.5 receipt.
LADDER_PICKS = {
    1: [0],
    2: [0, 6],
    4: [0, 6, 3, 11],
    8: [0, 1, 2, 3, 4, 5, 6, 11],
    12: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}


def load_pool(which):
    items = []
    if which in ("decode", "both"):
        d = json.load(open(DECODE_POOL))
        for p in d["decode"]:
            items.append((f"d{p['idx']:02d}-{p['kind']}", p["kind"], p["text"]))
    if which in ("l3", "both"):
        d = json.load(open(L3_POOL))
        for k in ("WARM", "A4630", "B5550", "C6470"):
            items.append((f"l3-{k}", "l3deep", d[k]))
    return items


def post(body, timeout=600):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def one_shot(prompt, mode, max_tokens, timeout=600):
    body = {"model": MODEL, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    if mode == "greedy":
        body["temperature"] = 0
    t0 = time.monotonic()
    try:
        with post(body, timeout) as r:
            payload = json.loads(r.read())
        return payload, time.monotonic() - t0, None
    except urllib.error.HTTPError as e:
        return None, time.monotonic() - t0, f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001
        return None, time.monotonic() - t0, repr(e)


def streamed(messages, mode, max_tokens, timeout=600, epoch=None):
    """Returns dict with ttft_s, gen_wall_s, usage, text, reasoning, finish, err and
    (when epoch given) burst-relative t_req_s / t_first_s / t_last_s timestamps."""
    body = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if mode == "greedy":
        body["temperature"] = 0
    t0 = time.monotonic()
    t_first = t_last = None
    usage = None
    finish = None
    http_code = None
    text, reasoning = [], []
    err = None
    try:
        with post(body, timeout) as r:
            http_code = r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    got = False
                    rsn_delta = delta.get("reasoning") or delta.get("reasoning_content")
                    if rsn_delta:
                        reasoning.append(rsn_delta); got = True
                    if delta.get("content"):
                        text.append(delta["content"]); got = True
                    if got:
                        now = time.monotonic()
                        if t_first is None:
                            t_first = now
                        t_last = now
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read()[:300]!r}"
        http_code = e.code
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    if err is None and t_first is None:
        err = "no content chunks"
    out = {
        "ttft_s": (t_first - t0) if t_first else None,
        "gen_wall_s": (t_last - t_first) if t_first else None,
        "usage": usage, "text": "".join(text), "reasoning": "".join(reasoning),
        "finish": finish, "err": err, "http_code": http_code,
    }
    if epoch is not None:
        out["t_req_s"] = t0 - epoch
        out["t_first_s"] = (t_first - epoch) if t_first else None
        out["t_last_s"] = (t_last - epoch) if t_last else None
    return out


def cmd_sample(a):
    os.makedirs(a.out, exist_ok=True)
    tag, _, text = load_pool("decode")[0]
    payload, wall, err = one_shot(text, "greedy", 64)
    row = {"tag": tag, "wall_s": round(wall, 2), "err": err}
    if payload:
        msg = payload["choices"][0]["message"]
        row["content"] = msg.get("content")
        row["reasoning_content"] = msg.get("reasoning") or msg.get("reasoning_content")
        row["usage"] = payload.get("usage")
    json.dump(row, open(os.path.join(a.out, "fresh-boot-sample.json"), "w"), indent=1)
    body = (row.get("reasoning_content") or "") + (row.get("content") or "")
    ok = err is None and len(body.strip()) > 20
    print(f"[sample] ok={ok} wall={wall:.1f}s text={body[:120]!r}")
    return 0 if ok else 1


def cmd_timed(a):
    os.makedirs(a.out, exist_ok=True)
    out = {"pool_rows": [], "deep_ttft": [], "vendor_row": None}
    for tag, kind, text in load_pool("decode"):
        r = streamed([{"role": "user", "content": text}], "greedy", a.max_tokens)
        ct = (r["usage"] or {}).get("completion_tokens")
        toks = (ct - 1) / r["gen_wall_s"] if (ct and r["gen_wall_s"] and ct > 1) else None
        row = {"tag": tag, "kind": kind, "ttft_s": r["ttft_s"], "gen_wall_s": r["gen_wall_s"],
               "completion_tokens": ct, "decode_tok_s": toks, "finish": r["finish"], "err": r["err"]}
        out["pool_rows"].append(row)
        open(os.path.join(a.out, tag + ".txt"), "wb").write((r["reasoning"] + "\x00" + r["text"]).encode())
        print(f"[timed] {tag} ttft={r['ttft_s'] and round(r['ttft_s'],3)} tok/s={toks and round(toks,1)} err={r['err']}")
    for tag, kind, text in load_pool("l3"):
        r = streamed([{"role": "user", "content": text}], "greedy", a.max_tokens)
        ct = (r["usage"] or {}).get("completion_tokens")
        toks = (ct - 1) / r["gen_wall_s"] if (ct and r["gen_wall_s"] and ct > 1) else None
        row = {"tag": tag, "kind": kind, "ttft_s": r["ttft_s"], "gen_wall_s": r["gen_wall_s"],
               "completion_tokens": ct, "decode_tok_s": toks,
               "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
               "finish": r["finish"], "err": r["err"]}
        out["pool_rows"].append(row)
        open(os.path.join(a.out, tag + ".txt"), "wb").write((r["reasoning"] + "\x00" + r["text"]).encode())
        print(f"[timed l3] {tag} ttft={r['ttft_s'] and round(r['ttft_s'],3)} tok/s={toks and round(toks,1)} ptok={row['prompt_tokens']} err={r['err']}")
    for tag, kind, text in load_pool("l3"):
        if tag not in ("l3-WARM", "l3-A4630"):
            continue
        r = streamed([{"role": "user", "content": text}], "greedy", 64)
        out["deep_ttft"].append({"tag": tag, "ttft_s": r["ttft_s"],
                                 "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "err": r["err"]})
        print(f"[timed deep] {tag} ttft={r['ttft_s'] and round(r['ttft_s'],3)} err={r['err']}")
    # ONE vendor-default sampled row (serving law: the real traffic shape, no params)
    tag, kind, text = load_pool("decode")[4]
    r = streamed([{"role": "user", "content": text}], "vendor", a.max_tokens)
    ct = (r["usage"] or {}).get("completion_tokens")
    toks = (ct - 1) / r["gen_wall_s"] if (ct and r["gen_wall_s"] and ct > 1) else None
    short = bool(ct and ct < 128)  # 3way measurement-trap floor: short sampled rows excluded from tok/s
    out["vendor_row"] = {"tag": tag, "ttft_s": r["ttft_s"], "decode_tok_s": toks,
                         "completion_tokens": ct, "finish": r["finish"], "err": r["err"],
                         "short_row_excluded_from_tok_s": short}
    print(f"[timed vendor] {tag} tok/s={toks and round(toks,1)} ct={ct} short_excluded={short} err={r['err']}")
    med = statistics.median([x["decode_tok_s"] for x in out["pool_rows"] if x["kind"] != "l3deep" and x["decode_tok_s"]])
    out["decode_pool_median_tok_s"] = med
    print(f"[timed] decode-pool median tok/s = {med:.2f}")
    json.dump(out, open(os.path.join(a.out, "timed.json"), "w"), indent=1)
    return 0


def _burst_stats(rows, wall, n, mode, label):
    ok = [r for r in rows if r and not r["err"]]
    tot = sum(r["completion_tokens"] or 0 for r in ok)
    toks = sorted(r["decode_tok_s"] for r in ok if r["decode_tok_s"])
    ttfts = sorted(r["ttft_s"] for r in ok if r["ttft_s"] is not None)
    firsts = [r["t_first_s"] for r in ok if r.get("t_first_s") is not None]
    lasts = [r["t_last_s"] for r in ok if r.get("t_last_s") is not None]
    dw = None
    if firsts and lasts and max(lasts) > min(firsts):
        dw = sum((r["completion_tokens"] or 1) - 1 for r in ok) / (max(lasts) - min(firsts))
    def pct(v, p):
        if not v:
            return None
        k = max(0, min(len(v) - 1, int(round(p * (len(v) - 1)))))
        return v[k]
    return {
        "n": n, "mode": mode, "label": label, "wall_s": wall,
        "rows_ok": len(ok), "rows_err": len(rows) - len(ok),
        "total_completion_tokens": tot,
        "aggregate_tok_s": tot / wall if wall else None,
        "decode_window_tok_s": dw,
        "per_session_tok_s_p50": pct(toks, 0.50), "per_session_tok_s_p95": pct(toks, 0.95),
        "per_session_tok_s_min": toks[0] if toks else None,
        "ttft_p50_s": pct(ttfts, 0.50), "ttft_max_s": ttfts[-1] if ttfts else None,
        "rows": rows,
    }


def _run_burst(picks, mode, max_tokens, outdir, n_label, tag_prefix=""):
    os.makedirs(outdir, exist_ok=True)
    rows = [None] * len(picks)
    epoch = time.monotonic()

    def work(i, item):
        tag, kind, text = item
        r = streamed([{"role": "user", "content": text}], mode, max_tokens, timeout=900, epoch=epoch)
        ct = (r["usage"] or {}).get("completion_tokens")
        rows[i] = {"tag": tag, "kind": kind, "ttft_s": r["ttft_s"], "gen_wall_s": r["gen_wall_s"],
                   "completion_tokens": ct,
                   "decode_tok_s": (ct - 1) / r["gen_wall_s"] if (ct and r["gen_wall_s"] and ct > 1) else None,
                   "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
                   "finish": r["finish"], "err": r["err"], "http_code": r["http_code"],
                   "t_req_s": r.get("t_req_s"), "t_first_s": r.get("t_first_s"), "t_last_s": r.get("t_last_s")}
        open(os.path.join(outdir, f"{tag_prefix}{tag}.txt"), "wb").write(
            ((r["reasoning"] or "") + "\x00" + (r["text"] or "")).encode())

    ths = [threading.Thread(target=work, args=(i, it)) for i, it in enumerate(picks)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.monotonic() - epoch
    return _burst_stats(rows, wall, n_label, mode, os.path.basename(outdir))


def cmd_conc(a):
    pool = load_pool("both")
    if a.picks:
        idx = [int(x) for x in a.picks.split(",")]
    else:
        idx = LADDER_PICKS[a.n]
    picks = [pool[i] for i in idx]
    assert len(picks) == a.n, f"picks {len(picks)} != n {a.n}"
    out = _run_burst(picks, a.mode, a.max_tokens, a.out, a.n)
    json.dump(out, open(os.path.join(a.out, f"conc-{a.n}-{a.mode}.json"), "w"), indent=1)
    for r in out["rows"]:
        print(f"[conc] {r['tag']} ttft={r['ttft_s'] and round(r['ttft_s'],3)} "
              f"tok/s={r['decode_tok_s'] and round(r['decode_tok_s'],1)} ct={r['completion_tokens']} err={r['err']}")
    print(f"[conc] n={a.n} mode={a.mode} wall={out['wall_s']:.1f}s total={out['total_completion_tokens']} "
          f"agg={out['aggregate_tok_s'] and round(out['aggregate_tok_s'],2)} "
          f"dw={out['decode_window_tok_s'] and round(out['decode_window_tok_s'],2)} "
          f"p50={out['per_session_tok_s_p50'] and round(out['per_session_tok_s_p50'],1)} "
          f"p95={out['per_session_tok_s_p95'] and round(out['per_session_tok_s_p95'],1)} "
          f"ttft_p50={out['ttft_p50_s'] and round(out['ttft_p50_s'],2)} "
          f"ttft_max={out['ttft_max_s'] and round(out['ttft_max_s'],2)} errs={out['rows_err']}")
    return 1 if out["rows_err"] else 0


def cmd_solo(a):
    """Same picks run sequentially — the solo tapes for the concurrent-vs-solo identity bar."""
    os.makedirs(a.out, exist_ok=True)
    pool = load_pool("both")
    idx = [int(x) for x in a.picks.split(",")]
    meta = {"rows": []}
    for i in idx:
        tag, kind, text = pool[i]
        r = streamed([{"role": "user", "content": text}], a.mode, a.max_tokens, timeout=900)
        ct = (r["usage"] or {}).get("completion_tokens")
        meta["rows"].append({"tag": tag, "ttft_s": r["ttft_s"], "completion_tokens": ct,
                             "decode_tok_s": (ct - 1) / r["gen_wall_s"] if (ct and r["gen_wall_s"] and ct > 1) else None,
                             "finish": r["finish"], "err": r["err"]})
        open(os.path.join(a.out, tag + ".txt"), "wb").write(
            ((r["reasoning"] or "") + "\x00" + (r["text"] or "")).encode())
        print(f"[solo] {tag} tok/s={meta['rows'][-1]['decode_tok_s'] and round(meta['rows'][-1]['decode_tok_s'],1)} err={r['err']}")
    json.dump(meta, open(os.path.join(a.out, f"solo-{a.mode}.json"), "w"), indent=1)
    return 1 if any(r["err"] for r in meta["rows"]) else 0


def cmd_twin(a):
    os.makedirs(a.out, exist_ok=True)
    l3 = json.load(open(L3_POOL))
    dec = load_pool("decode")
    turns = [l3["A4630"]] + [t for _, _, t in dec[:7]]
    msgs, rows = [], []
    for i, content in enumerate(turns, 1):
        msgs.append({"role": "user", "content": content})
        r = streamed(msgs, "vendor", a.max_tokens, timeout=900)
        rows.append({"turn": i, "ttft_s": r["ttft_s"], "gen_wall_s": r["gen_wall_s"],
                     "usage": r["usage"], "finish": r["finish"], "err": r["err"]})
        print(f"[twin] turn={i} ttft={r['ttft_s'] and round(r['ttft_s'],3)} usage={r['usage']} err={r['err']}")
        if r["err"]:
            break
        msgs.append({"role": "assistant", "content": (r["reasoning"] or "") + (r["text"] or "")})
    json.dump({"rows": rows}, open(os.path.join(a.out, "twin.json"), "w"), indent=1)
    return 0


def cmd_compare(a):
    ta = {f: open(os.path.join(a.a, f), "rb").read()
          for f in os.listdir(a.a) if f.endswith(".txt")}
    tb = {f: open(os.path.join(a.b, f), "rb").read()
          for f in os.listdir(a.b) if f.endswith(".txt")}
    shared = sorted(set(ta) & set(tb))
    diverged = []
    for f in shared:
        if ta[f] != tb[f]:
            x, y = ta[f], tb[f]
            off = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
            diverged.append((f, off, len(x), len(y)))
    print(f"[compare] {a.a} vs {a.b}: shared={len(shared)} identical={len(shared)-len(diverged)} diverged={len(diverged)}")
    for f, off, la, lb in diverged:
        print(f"  DIVERGED {f} first_diff_at={off} len_a={la} len_b={lb}")
    return 1 if diverged else 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--out", required=True)
    s = sub.add_parser("timed"); s.add_argument("--out", required=True)
    s.add_argument("--max-tokens", type=int, default=256)
    s = sub.add_parser("conc"); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, required=True); s.add_argument("--mode", default="greedy")
    s.add_argument("--max-tokens", type=int, default=256); s.add_argument("--picks", default=None)
    s = sub.add_parser("solo"); s.add_argument("--out", required=True)
    s.add_argument("--picks", required=True); s.add_argument("--mode", default="greedy")
    s.add_argument("--max-tokens", type=int, default=256)
    s = sub.add_parser("twin"); s.add_argument("--out", required=True)
    s.add_argument("--max-tokens", type=int, default=128)
    s = sub.add_parser("compare"); s.add_argument("--a", required=True); s.add_argument("--b", required=True)
    a = p.parse_args()
    sys.exit({"sample": cmd_sample, "timed": cmd_timed, "conc": cmd_conc, "solo": cmd_solo,
              "twin": cmd_twin, "compare": cmd_compare}[a.cmd](a))


if __name__ == "__main__":
    main()
