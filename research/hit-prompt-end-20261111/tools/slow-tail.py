#!/usr/bin/env python3
"""memra#1233: what the slow tail of a boot's hit-role requests is made of.

The all-90 hit TTFT p95 sits at rank 84.55 of 89. A hit-role request that restores nothing is a
cold prime, so with M of them the p95 is the (M - 5.45)-th slowest: where it lands inside the cold
distribution is set by M alone. Per boot this prints M, the not-restored requests' TTFTs in order,
the cold-role requests' p50 / p95 / max, the cold prime's own speed (ms per 1,000 tokens of the
ticks that primed at least 512), each cold request's TTFT as a multiple of an unshared 4,860-token
prime at that speed, and the fence's longest run: consecutive ticks that primed only short
suffixes while another row still had prompt to prime, and their summed prefill time (the time
waiting prefill was held back in one stretch).

usage: slow-tail.py RUN_DIR [RUN_DIR...]
"""
import gzip
import json
import sys
from pathlib import Path


def opened(path: Path):
    gz = Path(str(path) + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", errors="replace")
    return open(path, errors="replace")


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def ticks(log: Path) -> dict:
    cold_s = cold_tok = 0.0
    run_s = run_n = best_s = best_n = 0.0
    with opened(log) as fh:
        for line in fh:
            if not line.startswith("[tick]"):
                continue
            d = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
            toks = int(d["prefill_single_tokens"]) + int(d["prefill_batch_tokens"])
            pf = float(d["prefill_ms"])
            if toks >= 512:
                cold_s += pf
                cold_tok += toks
            # A tick that primed only suffixes while a row was still priming held that row back.
            if 0 < toks < 512 and int(d["priming"]) >= 1:
                run_s += pf
                run_n += 1
                if run_s > best_s:
                    best_s, best_n = run_s, run_n
            elif toks >= 512 or (toks == 0 and int(d["priming"]) == 0):
                run_s = run_n = 0.0
    return {"ms_per_1k": 1000 * cold_s / cold_tok if cold_tok else None, "fence_run_ms": best_s, "fence_run_ticks": int(best_n)}


for d in [Path(a) for a in sys.argv[1:]]:
    with opened(d / "cell.jsonl") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    labels = []
    for r in rows:
        if r.get("kind") == "summary" and r["label"] not in labels:
            labels.append(r["label"])
    print(f"## {d.name}\n")
    print("| boot | not restored (M) | all-90 p95 ms | not-restored TTFTs, ms, ascending | cold-role TTFT p50 / p95 / max | ms per 1,000 cold tokens | cold-role p50 / p95 as multiples of an unshared prime | longest fenced run, ticks / ms |")
    print("|---|---|---|---|---|---|---|---|")
    for label in labels:
        req = [r for r in rows if r.get("kind") == "request" and r.get("target") == label]
        hits = [r for r in req if r.get("cache_role") == "hit"]
        missed = sorted(float(r["ttft_ms"]) for r in hits if not (r.get("cached_tokens") or 0))
        cold = [float(r["ttft_ms"]) for r in req if r.get("cache_role") == "miss"]
        t = ticks(d / f"server-{label}.log")
        solo = 4.86 * t["ms_per_1k"]
        print(f"| {label} | {len(missed)} | {pct([float(r['ttft_ms']) for r in hits], 0.95):,.0f} | {' '.join(f'{x:,.0f}' for x in missed)} | "
              f"{pct(cold, 0.5):,.0f} / {pct(cold, 0.95):,.0f} / {max(cold):,.0f} | {t['ms_per_1k']:.0f} | "
              f"{pct(cold, 0.5) / solo:.2f} / {pct(cold, 0.95) / solo:.2f} | {t['fence_run_ticks']} / {t['fence_run_ms']:,.0f} |")
    print()
