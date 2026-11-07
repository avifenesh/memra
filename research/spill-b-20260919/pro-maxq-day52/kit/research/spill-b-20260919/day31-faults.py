#!/usr/bin/env python3
"""Day 31 fault record (DAY31.md section 2), written after the first crash was seen: a lister, not a verdict. For each
boot dir it prints, from the stamped server.log and client.jsonl, every panic, `[worker]` PANIC/respawn/FATAL line and
`[engine-error]` line verbatim with its server.log line number, the request in flight or in admission at that stamp,
the last sequential request that returned before it (tag, P, G, finish, P+G against booked-64), and the boot's
non-200 and no-status rows. It never edits its inputs and computes no pass mark.

A boot whose server.log was committed gzipped (server.log.gz) is read through gzip; line numbers are the same.

usage: day31-faults.py <boot dir> ...
"""
import gzip, json, os, re, sys

KEYS = ("panicked at", "[worker] PANIC", "[worker] respawn", "[worker] FATAL", "[engine-error]")
for d in sys.argv[1:]:
    name = os.path.basename(d.rstrip("/"))
    lp = os.path.join(d, "server.log")
    fh = open(lp, encoding="utf-8", errors="replace") if os.path.exists(lp) else \
        gzip.open(lp + ".gz", "rt", encoding="utf-8", errors="replace")
    lines = fh.read().split("\n")
    rows = [json.loads(l) for l in open(os.path.join(d, "client.jsonl"))] if os.path.exists(os.path.join(d, "client.jsonl")) else []
    booked = {}
    if os.path.exists(os.path.join(d, "rows.jsonl")):
        for l in open(os.path.join(d, "rows.jsonl")):
            r = json.loads(l); booked[r["tag"]] = r.get("booked_ctx")
    seq = [r for r in rows if r["class"] != "burst"]
    hits = []
    for i, l in enumerate(lines, 1):
        m = re.match(r"(\d+) (.*)", l)
        if m and any(k in m.group(2) for k in KEYS):
            hits.append((i, int(m.group(1)), m.group(2)))
    panics = sum(1 for _, _, s in hits if "panicked at" in s)
    respawns = sum(1 for _, _, s in hits if s.startswith("[worker] respawn"))
    fatal = any(s.startswith("[worker] FATAL") for _, _, s in hits)
    print(f"== {name}: panics={panics} respawns={respawns} fatal={fatal} engine_errors={sum(1 for _, _, s in hits if s.startswith('[engine-error]'))}")
    for i, t, s in hits:
        cur = next((r for r in rows if r["submit_ms"] <= t <= (r.get("done_ms") or 1 << 62)), None)
        prev = [r for r in seq if (r.get("done_ms") or 0) <= t and r["status"] == 200]
        p = prev[-1] if prev else None
        pu = (p or {}).get("usage") or {}
        pb = booked.get(p["tag"]) if p else None
        edge = (f"prev={p['tag']} P={pu.get('prompt_tokens')} G={pu.get('completion_tokens')} finish={p.get('finish_reason')} "
                f"P+G={(pu.get('prompt_tokens') or 0) + (pu.get('completion_tokens') or 0)} booked-64={pb - 64 if pb else None}") if p else "prev=-"
        print(f"  server.log:{i} t={t} in={cur['tag'] if cur else '-'} {edge}")
        print(f"    {s[:400]}")
    bad = [r for r in rows if r["status"] != 200]
    first_dead = next((r["tag"] for r in rows if r["status"] is None), None)
    print(f"  non-200 rows={len(bad)} status_none={sum(1 for r in rows if r['status'] is None)} first_no_status={first_dead}")
    for r in bad:
        if r["status"] is not None:
            print(f"    {r['tag']} status={r['status']} err={(r.get('err') or '')[:160]}")
