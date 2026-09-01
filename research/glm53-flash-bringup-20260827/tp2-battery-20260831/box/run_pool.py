#!/usr/bin/env python3
"""3way-decision pool driver (glm5 plain vs native-MTP vs DFlash2, box stage). stdlib only.

Derived from spec-battery-20260830/box/run_pool.py (same pools, same tape definition, same
usage.spec aggregation) so the numbers are directly comparable to the banked native-arm rows.
Added here: `conc` (c=4 mixed-pool concurrency), `--pool` on timed, `--vendor-only`.

Subcommands:
  sample  --out DIR                       fresh-boot output sample (greedy 64, decode d00)
  cell    --out DIR --pool decode|l3|both --mode greedy|vendor --max-tokens N [--k K]
  timed   --out DIR                       per-boot timed cell: streamed greedy decode pool
          (TTFT + decode tok/s) + deep TTFT rows (l3 WARM ~0.4k + A4630 ~3.7k cold)
          + ONE vendor-default sampled row
  twin    --out DIR                       8-turn larger-prompt conversation, streamed,
          per-turn TTFT + usage (cache + spec)
  conc    --out DIR --n 4                  c=N concurrent mixed-pool streamed rows
  compare --a DIR --b DIR                 byte-identity over shared *.txt tapes
  agg     --dirs DIR [DIR..]              acceptance table from cell meta files
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
DECODE_POOL = "/root/memra-tp2/research/glm53-flash-bringup-20260827/decode-attribution-receipts/prompts.json"
L3_POOL = "/root/l3-ab-prompts.json"


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


def one_shot(prompt_or_messages, mode, max_tokens, timeout=600):
    body = {"model": MODEL, "max_tokens": max_tokens}
    if isinstance(prompt_or_messages, str):
        body["messages"] = [{"role": "user", "content": prompt_or_messages}]
    else:
        body["messages"] = prompt_or_messages
    if mode == "greedy":
        body["temperature"] = 0
    # vendor mode: NO sampling params on the wire (serving law)
    t0 = time.monotonic()
    try:
        with post(body, timeout) as r:
            payload = json.loads(r.read())
        return payload, time.monotonic() - t0, None
    except urllib.error.HTTPError as e:
        return None, time.monotonic() - t0, f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001
        return None, time.monotonic() - t0, repr(e)


def streamed(messages, mode, max_tokens, timeout=600):
    """Returns (ttft_s, gen_wall_s, usage, text, reasoning, finish, err)."""
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
    text, reasoning = [], []
    try:
        with post(body, timeout) as r:
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
    except Exception as e:  # noqa: BLE001
        return None, None, usage, "".join(text), "".join(reasoning), finish, repr(e)
    if t_first is None:
        return None, None, usage, "", "", finish, "no content chunks"
    return t_first - t0, (t_last - t_first), usage, "".join(text), "".join(reasoning), finish, None


def tape_bytes(payload):
    msg = payload["choices"][0]["message"]
    return ((msg.get("reasoning") or msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")).encode()


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


def cmd_cell(a):
    os.makedirs(a.out, exist_ok=True)
    # NOTE: K is a BOOT PIN (MEMRA_SPEC_K env), never a request field — there is no
    # request-level spec_k on this server, so --k here is a receipt LABEL of the boot's pin
    # (the caller passes the same value it set on the boot); it never travels on the wire.
    meta = {"pool": a.pool, "mode": a.mode, "max_tokens": a.max_tokens,
            "k_boot_pin_label": a.k, "rows": []}
    items = load_pool(a.pool)
    if a.tags:
        want = [t for t in a.tags.split(",") if t]
        bytag = {t: (t, k, x) for t, k, x in items}
        missing = [t for t in want if t not in bytag]
        if missing:
            print(f"[cell] FATAL: unknown tags {missing} (pool={a.pool})")
            return 2
        items = [bytag[t] for t in want]
    elif a.limit:
        items = items[: a.limit]
    for tag, kind, text in items:
        payload, wall, err = one_shot(text, a.mode, a.max_tokens)
        row = {"tag": tag, "kind": kind, "wall_s": round(wall, 2), "err": err}
        if payload:
            open(os.path.join(a.out, tag + ".txt"), "wb").write(tape_bytes(payload))
            json.dump(payload, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
            u = payload.get("usage") or {}
            row["finish"] = payload["choices"][0].get("finish_reason")
            row["completion_tokens"] = u.get("completion_tokens")
            row["spec"] = u.get("spec")
            row["tape_sha"] = hashlib.sha256(tape_bytes(payload)).hexdigest()[:16]
        meta["rows"].append(row)
        print(f"[cell {a.mode} k={a.k}] {tag} wall={wall:.1f}s spec={row.get('spec')} err={err}")
    json.dump(meta, open(os.path.join(a.out, f"meta-{a.mode}.json"), "w"), indent=1)
    errs = [r for r in meta["rows"] if r["err"]]
    print(f"[cell] done rows={len(meta['rows'])} errors={len(errs)}")
    return 0


def cmd_timed(a):
    os.makedirs(a.out, exist_ok=True)
    out = {"pool_rows": [], "deep_ttft": [], "vendor_row": None}
    for tag, kind, text in load_pool("decode"):
        ttft, gen, usage, txt, rsn, fin, err = streamed(
            [{"role": "user", "content": text}], "greedy", a.max_tokens)
        ct = (usage or {}).get("completion_tokens")
        toks = (ct - 1) / gen if (ct and gen and ct > 1) else None
        row = {"tag": tag, "kind": kind, "ttft_s": ttft, "gen_wall_s": gen,
               "completion_tokens": ct, "decode_tok_s": toks,
               "spec": (usage or {}).get("spec"), "finish": fin, "err": err}
        out["pool_rows"].append(row)
        open(os.path.join(a.out, tag + ".txt"), "wb").write((rsn + "\x00" + txt).encode())
        print(f"[timed] {tag} ttft={ttft and round(ttft,3)} tok/s={toks and round(toks,1)} spec={row['spec']} err={err}")
    # l3 deep pool: greedy tok/s on the deep shapes too (owner: decode tok/s on BOTH pools)
    for tag, kind, text in load_pool("l3"):
        ttft, gen, usage, txt, rsn, fin, err = streamed(
            [{"role": "user", "content": text}], "greedy", a.max_tokens)
        ct = (usage or {}).get("completion_tokens")
        toks = (ct - 1) / gen if (ct and gen and ct > 1) else None
        row = {"tag": tag, "kind": kind, "ttft_s": ttft, "gen_wall_s": gen,
               "completion_tokens": ct, "decode_tok_s": toks,
               "prompt_tokens": (usage or {}).get("prompt_tokens"),
               "spec": (usage or {}).get("spec"), "finish": fin, "err": err}
        out["pool_rows"].append(row)
        open(os.path.join(a.out, tag + ".txt"), "wb").write((rsn + "\x00" + txt).encode())
        print(f"[timed l3] {tag} ttft={ttft and round(ttft,3)} tok/s={toks and round(toks,1)} ptok={row['prompt_tokens']} err={err}")
    # deep TTFT rows (cold, short max_tokens) at ~0.4k (WARM) and ~3.7k (A4630)
    for tag, kind, text in load_pool("l3"):
        if tag not in ("l3-WARM", "l3-A4630"):
            continue
        ttft, gen, usage, txt, rsn, fin, err = streamed(
            [{"role": "user", "content": text}], "greedy", 64)
        out["deep_ttft"].append({"tag": tag, "ttft_s": ttft,
                                 "prompt_tokens": (usage or {}).get("prompt_tokens"), "err": err})
        print(f"[timed deep] {tag} ttft={ttft and round(ttft,3)} err={err}")
    # ONE vendor-default sampled row (serving law: the real traffic shape, no params)
    tag, kind, text = load_pool("decode")[4]
    ttft, gen, usage, txt, rsn, fin, err = streamed(
        [{"role": "user", "content": text}], "vendor", a.max_tokens)
    ct = (usage or {}).get("completion_tokens")
    toks = (ct - 1) / gen if (ct and gen and ct > 1) else None
    out["vendor_row"] = {"tag": tag, "ttft_s": ttft, "decode_tok_s": toks,
                         "completion_tokens": ct, "spec": (usage or {}).get("spec"),
                         "finish": fin, "err": err}
    print(f"[timed vendor] {tag} tok/s={toks and round(toks,1)} spec={out['vendor_row']['spec']} err={err}")
    json.dump(out, open(os.path.join(a.out, "timed.json"), "w"), indent=1)
    return 0


def cmd_twin(a):
    os.makedirs(a.out, exist_ok=True)
    l3 = json.load(open(L3_POOL))
    dec = load_pool("decode")
    turns = [l3["A4630"]] + [t for _, _, t in dec[:7]]
    msgs, rows = [], []
    for i, content in enumerate(turns, 1):
        msgs.append({"role": "user", "content": content})
        ttft, gen, usage, txt, rsn, fin, err = streamed(msgs, "vendor", a.max_tokens, timeout=900)
        rows.append({"turn": i, "ttft_s": ttft, "gen_wall_s": gen, "usage": usage,
                     "finish": fin, "err": err})
        print(f"[twin] turn={i} ttft={ttft and round(ttft,3)} usage={usage} err={err}")
        if err:
            break
        msgs.append({"role": "assistant", "content": (rsn or "") + (txt or "")})
    json.dump({"rows": rows}, open(os.path.join(a.out, "twin.json"), "w"), indent=1)
    return 0


def cmd_conc(a):
    """c=N concurrent streamed rows over a mixed pool (agentic + prose + deep)."""
    os.makedirs(a.out, exist_ok=True)
    pool = load_pool("both")
    picks = [pool[0], pool[6], pool[3], pool[11]][: a.n]  # code, prose, code, l3-A4630
    rows = [None] * len(picks)
    t_start = time.monotonic()

    def work(i, item):
        tag, kind, text = item
        ttft, gen, usage, txt, rsn, fin, err = streamed(
            [{"role": "user", "content": text}], a.mode, a.max_tokens, timeout=900)
        ct = (usage or {}).get("completion_tokens")
        rows[i] = {"tag": tag, "kind": kind, "ttft_s": ttft, "gen_wall_s": gen,
                   "completion_tokens": ct,
                   "decode_tok_s": (ct - 1) / gen if (ct and gen and ct > 1) else None,
                   "spec": (usage or {}).get("spec"), "finish": fin, "err": err}
        open(os.path.join(a.out, tag + ".txt"), "wb").write(((rsn or "") + "\x00" + (txt or "")).encode())

    ths = [threading.Thread(target=work, args=(i, it)) for i, it in enumerate(picks)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.monotonic() - t_start
    tot = sum(r["completion_tokens"] or 0 for r in rows if r)
    out = {"n": a.n, "mode": a.mode, "wall_s": wall, "total_completion_tokens": tot,
           "aggregate_tok_s": tot / wall if wall else None, "rows": rows}
    json.dump(out, open(os.path.join(a.out, f"conc-{a.n}-{a.mode}.json"), "w"), indent=1)
    for r in rows:
        print(f"[conc] {r['tag']} ttft={r['ttft_s'] and round(r['ttft_s'],3)} "
              f"tok/s={r['decode_tok_s'] and round(r['decode_tok_s'],1)} spec={r['spec']} err={r['err']}")
    print(f"[conc] n={a.n} wall={wall:.1f}s total_tokens={tot} aggregate_tok_s={tot/wall if wall else 0:.1f}")
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


def cmd_agg(a):
    print(f"{'cell':<28} {'mode':<8} {'class':<8} {'n':>3} {'acc/cyc':>8} {'tok/cyc':>8} {'accrate':>8}")
    for d in a.dirs:
        for mf in sorted(f for f in os.listdir(d) if f.startswith("meta-")):
            meta = json.load(open(os.path.join(d, mf)))
            groups = {}
            for r in meta["rows"]:
                s = r.get("spec")
                if not s or r.get("err"):
                    continue
                cls = "prose" if r["kind"] == "prose" else ("l3deep" if r["kind"] == "l3deep" else "agentic")
                for g in (cls, "ALL"):
                    groups.setdefault(g, []).append(s)
            for g, specs in sorted(groups.items()):
                acc = sum(s["accepted"] for s in specs)
                rnd = sum(s["rounds"] for s in specs)
                drf = sum(s["drafted"] for s in specs)
                if rnd:
                    print(f"{os.path.basename(d):<28} {meta['mode']:<8} {g:<8} {len(specs):>3} "
                          f"{acc/rnd:>8.3f} {(acc+rnd)/rnd:>8.3f} {acc/drf if drf else 0:>8.3f}")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--out", required=True)
    s = sub.add_parser("cell"); s.add_argument("--out", required=True)
    s.add_argument("--pool", default="both"); s.add_argument("--mode", default="greedy")
    s.add_argument("--max-tokens", type=int, default=128); s.add_argument("--k", type=int, default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--tags", default=None, help="comma-separated pool tags (exact subset)")
    s = sub.add_parser("timed"); s.add_argument("--out", required=True)
    s.add_argument("--max-tokens", type=int, default=256)
    s = sub.add_parser("twin"); s.add_argument("--out", required=True)
    s.add_argument("--max-tokens", type=int, default=128)
    s = sub.add_parser("conc"); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=4); s.add_argument("--mode", default="greedy")
    s.add_argument("--max-tokens", type=int, default=256)
    s = sub.add_parser("compare"); s.add_argument("--a", required=True); s.add_argument("--b", required=True)
    s = sub.add_parser("agg"); s.add_argument("--dirs", nargs="+", required=True)
    a = p.parse_args()
    sys.exit({"sample": cmd_sample, "cell": cmd_cell, "timed": cmd_timed, "twin": cmd_twin,
              "conc": cmd_conc, "compare": cmd_compare, "agg": cmd_agg}[a.cmd](a))


if __name__ == "__main__":
    main()
