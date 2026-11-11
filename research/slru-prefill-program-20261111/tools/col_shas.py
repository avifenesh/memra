#!/usr/bin/env python3
"""Per probe cell: sha16 of the prefill columns, of the decode columns, and of the whole table.

usage: col_shas.py <cell dir>...
"""
import hashlib
import pathlib
import re
import sys


def h(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


for d in sys.argv[1:]:
    log = pathlib.Path(d) / "probe.log"
    if not log.exists():
        print(f"{d}: no probe.log")
        continue
    rows = [l.split() for l in log.read_text().splitlines() if re.match(r"^\d{4} ", l)]
    pre = [" ".join([r[0], r[1], r[2]]) for r in rows]
    dec = [" ".join([r[0], r[3], r[4]]) for r in rows]
    full = [" ".join(r) for r in rows]
    modes = sorted(set(re.findall(r"engaged mode=(\w+)", log.read_text())))
    print(f"{d}: rows={len(rows)} prefill={h(pre)} decode={h(dec)} table={h(full)} slru_pairs_modes={modes}")
