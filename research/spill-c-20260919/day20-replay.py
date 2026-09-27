#!/usr/bin/env python3
"""Day 20 replay (lane/spill-c-20260919, DAY20.md rule, written before the after cell was read):
PASS = `identity; bounded`, every clause required, over the before and after `ev/` dirs of the tapladder cells.
identity: every rung EXACT before is EXACT after with the same spec_sha256, spec_len and acceptance a/b.
bounded:  every completed after rung's largest dflash.rs-traced allocation <= 420,966,400 bytes (a 4111-row
          chunk sink) and its [dflash-taps] former_full_tap_bytes == prompt x 102,400.
moved:    every rung refused before either completes EXACT after, or its last [alloc-trace] line before the
          OOM names a file:line other than the tap sink (dflash.rs:3977 on the before tree).
usage: day20-replay.py <before_ev> <after_ev> <words...>"""
import gzip, re, sys

before, after, words = sys.argv[1], sys.argv[2], sys.argv[3:]
TAP_ROW = 102_400
CHUNK_CAP = 4111 * TAP_ROW


def log_lines(ev, w):
    p = f"{ev}/w{w}.log"
    try:
        return gzip.open(p + ".gz", "rt", errors="replace").read().splitlines()
    except FileNotFoundError:
        return open(p, errors="replace").read().splitlines()


def parse(ev, w):
    lines = log_lines(ev, w)
    r = {"exact": any(l.endswith("| EXACT") for l in lines), "oom": any("OUT_OF_MEMORY" in l for l in lines)}
    for l in lines:
        m = re.match(r"\[dspark-q38-gate\] \S+: prompt=(\d+) spec_sha256=(\w+) plain_sha256=(\w+) spec_len=(\d+) plain_len=(\d+) prime_s=([\d.]+)", l)
        if m:
            r.update(prompt=int(m[1]), sha=m[2], plain_sha=m[3], spec_len=int(m[4]), prime_s=float(m[6]))
        m = re.match(r"\[dspark-q38\] acceptance (\d+)/(\d+)", l)
        if m:
            r["acc"] = f"{m[1]}/{m[2]}"
        m = re.match(r"\[dflash-taps\] base=(\d+) rows=(\d+) max_chunk_rows=(\d+) chunk_tap_bytes=(\d+) carry_bytes=(\d+) former_full_tap_bytes=(\d+)", l)
        if m:
            r.update(rows=int(m[2]), max_chunk_rows=int(m[3]), chunk_tap_bytes=int(m[4]), carry_bytes=int(m[5]), former=int(m[6]))
    traces = [(int(m[1]), m[2]) for l in lines for m in [re.match(r"\[alloc-trace\] (\d+) bytes from (\S+)", l)] if m]
    dfl = [t for t in traces if "dflash.rs" in t[1]]
    r["max_dflash_alloc"] = max(dfl)[0] if dfl else 0
    r["max_dflash_site"] = max(dfl)[1] if dfl else "-"
    r["last_trace"] = traces[-1] if traces else None
    return r


ok = True
clauses = []


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    clauses.append(f"{name}={'ok' if cond else 'FAIL'} ({detail})")


for w in words:
    b, a = parse(before, w), parse(after, w)
    tag = f"w{w}"
    if b["exact"]:
        check(f"{tag}.identity", a["exact"] and a.get("sha") == b.get("sha") and a.get("spec_len") == b.get("spec_len") and a.get("acc") == b.get("acc"),
              f"before sha={b.get('sha','')[:16]} acc={b.get('acc')} len={b.get('spec_len')}; after exact={a['exact']} sha={a.get('sha','')[:16]} acc={a.get('acc')} len={a.get('spec_len')}")
    else:
        moved = a["exact"] or (a["oom"] and a["last_trace"] is not None and "dflash.rs" not in a["last_trace"][1])
        check(f"{tag}.moved", moved, f"before last_trace={b['last_trace']}; after exact={a['exact']} oom={a['oom']} last_trace={a['last_trace']}")
    if a["exact"]:
        check(f"{tag}.bounded", a["max_dflash_alloc"] <= CHUNK_CAP and a.get("former") == a.get("prompt", -1) * TAP_ROW,
              f"after max_dflash_alloc={a['max_dflash_alloc']}@{a['max_dflash_site']} chunk_tap_bytes={a.get('chunk_tap_bytes')} carry_bytes={a.get('carry_bytes')} former={a.get('former')} before tap={b['max_dflash_alloc']}")
    print(f"{tag}: before prompt={b.get('prompt')} exact={b['exact']} tap={b['max_dflash_alloc']} prime_s={b.get('prime_s')} | after prompt={a.get('prompt')} exact={a['exact']} max_dflash_alloc={a['max_dflash_alloc']} prime_s={a.get('prime_s')}")
print("TAPLADDER20 rule " + " ".join(clauses))
print(f"DAY20 REPLAY tapladder20b: {'PASS' if ok else 'FAIL'} ({len(clauses)} checks) -> {'identity; bounded' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
