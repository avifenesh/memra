#!/usr/bin/env python3
"""memra#1233: split one boot's frozen-cell requests by what the hit restored and where its first
token waited. Joins the cell rows (request_id, cache_role, cached_tokens, ttft_ms) with the
server's `[ttft]` trace lines (MEMRA_TTFT_TRACE=1) by request id.

Classes: hit-whole (cached == prompt), hit-suffix (0 < cached < prompt), hit-missed (a hit row
that restored nothing), cold (miss rows). Per class: N, client TTFT p50/p95, and the trace's
prime_wait / prime / decode_wait p50/p95 (decode_wait = prime end to first decode: a hit's
suffix that shared its tick with a cold chunk waits here).

usage: ttft-split.py CELL_JSONL SERVER_LOG LABEL [--json]"""
import json
import re
import sys

RE = re.compile(r"\[ttft\] id=(\S+) .*?prime_wait_ms=(\S+) prime_ms=(\S+) decode_wait_ms=(\S+)")


def opened(path):
    """Open a banked text file, gzipped or not (`path` or `path.gz`)."""
    import gzip
    from pathlib import Path
    gz = Path(str(path) + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", errors="replace")
    return open(path, errors="replace")



def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def num(v):
    try:
        return float(v)
    except ValueError:
        return None


cell, log, label = sys.argv[1:4]
trace = {}
for line in opened(log):
    m = RE.search(line)
    if m:
        trace[m.group(1)] = tuple(num(m.group(i)) for i in (2, 3, 4))
rows = [json.loads(ln) for ln in opened(cell) if ln.strip()]
rows = [r for r in rows if r.get("kind") == "request" and r.get("target") == label]
classes = {}
for r in rows:
    cached, prompt = r.get("cached_tokens") or 0, r.get("prompt_tokens")
    if r.get("cache_role") == "miss":
        c = "cold"
    elif cached == prompt:
        c = "hit-whole"
    elif cached > 0:
        c = "hit-suffix"
    else:
        c = "hit-missed"
    classes.setdefault(c, []).append((r.get("ttft_ms"), *trace.get(r.get("request_id"), (None, None, None))))
out = {"label": label, "requests": len(rows), "traced": sum(1 for r in rows if r.get("request_id") in trace)}
for c, xs in sorted(classes.items()):
    cols = list(zip(*xs))
    out[c] = {"n": len(xs)} | {
        f"{name}_{q}": pct(col, p) for name, col in zip(("ttft", "prime_wait", "prime", "decode_wait"), cols)
        for q, p in (("p50", 0.5), ("p95", 0.95))
    }
if "--json" in sys.argv:
    print(json.dumps(out, sort_keys=True))
else:
    print(f"{label}: {out['requests']} requests, {out['traced']} traced")
    for c in ("hit-whole", "hit-suffix", "hit-missed", "cold"):
        if c in out:
            o = out[c]
            print(f"  {c:10s} n={o['n']:3d} ttft p50/p95 {o['ttft_p50']}/{o['ttft_p95']}  prime_wait {o['prime_wait_p50']}/{o['prime_wait_p95']}  "
                  f"prime {o['prime_p50']}/{o['prime_p95']}  decode_wait {o['decode_wait_p50']}/{o['decode_wait_p95']}")
