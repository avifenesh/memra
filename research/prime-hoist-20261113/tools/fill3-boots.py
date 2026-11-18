#!/usr/bin/env python3
"""memra#1321 GB10 fill3 rows: per arm and boot, MemFree at the boot, the first call's need per
prime, growth, trims, OOM lines, [prime-ws] lines, and the answer texts compared across boots/arms.
usage: fill3-boots.py DIR (holding spread-<arm>-5.7k-fill3/)"""
import json
import re
import sys
from pathlib import Path

base = Path(sys.argv[1])
texts = {}
for arm in ("dmain2", "ds2i"):
    d = base / f"spread-{arm}-5.7k-fill3"
    for boot in sorted(p for p in d.glob("b*-mode0") if p.is_dir()):
        n = boot.name[1]
        guard = (d / f"guard-b{n}.log").read_text() if (d / f"guard-b{n}.log").exists() else ""
        m = re.search(rf"boot {n} starts with MemFree (\d+) MB", guard)
        rows = [json.loads(l) for l in (boot / "rows.jsonl").read_text().splitlines() if l.strip()]
        need = [r["calls"][0]["demand_mib"] for r in rows]
        grow = [r["calls"][0]["growth_mib"] for r in rows]
        grow_all = [sum(c["growth_mib"] for c in r["calls"]) for r in rows]
        trims = sum(1 for r in rows if r.get("trim_lines"))
        oom = sum(r.get("oom_lines", 0) for r in rows)
        log = (boot / "serve.log").read_text(errors="replace")
        ws = log.count("[prime-ws]")
        status = sorted({r["status"] for r in rows})
        texts[(arm, n)] = [r["text_sha256"] for r in rows]
        print(f"{arm} b{n}: MemFree {m.group(1) if m else '?'} MB, n={len(rows)} status={status}, "
              f"need p1-2 {need[:2]}, from p3 {min(need[2:])}-{max(need[2:])}, growth first call {grow}, "
              f"growth from p4 (all calls) {sorted(set(grow_all[3:]))}, trims {trims}, oom {oom}, [prime-ws] {ws}")
keys = sorted(texts)
ref = texts[keys[0]]
for k in keys:
    print(k, "texts equal to", keys[0], texts[k] == ref)
