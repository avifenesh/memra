#!/usr/bin/env python3
"""memra#1321 fold: one line per fresh boot of a first-row cell (b-first.sh) or any b-sweep row
directory. Reads rows.jsonl (status, margin seen, out-of-memory lines) and, from serve.log, the
[diag1255] snapshots of the placed prime: pool reserved/used after its first call and before its
second (the gap in which other sessions run), the [prime-ws] lines and the gap's allocation sum.
usage: first-table.py DIR...   (DIR holds rows.jsonl and serve.log)"""
import json
import re
import sys
from pathlib import Path

SNAP = re.compile(r"\[diag1255\] (dspark-prime-[a-z-]+) id=(\S+?)(?: rows=\d+)?: driver_free=(\d+)MB pool_res=(\d+)MB "
                  r"pool_used=(\d+)MB pool_unused=(\d+)MB eff_free=(\d+)MB")
PASS = re.compile(r"\[diag1255\] gate-pass id=(\S+) prompt=(\d+) path=(\w+)")
GAP = re.compile(r"\[diag1255-ring\] gap-before-chunk id=(\S+): sum (\d+) bytes")

print("boot | prompt | status | seen MiB | oom | call1-after res/used | call2-before res/used | gap pool growth | "
      "gap alloc MB | prime-ws")
for d in sys.argv[1:]:
    d = Path(d)
    rows = [json.loads(l) for l in (d / "rows.jsonl").read_text().splitlines() if l.strip()] if (d / "rows.jsonl").exists() else []
    log = (d / "serve.log").read_text(errors="replace").splitlines()
    long_ids = [m.group(1) for l in log if (m := PASS.search(l)) and int(m.group(2)) > 1000]
    for k, row in enumerate(rows):
        # The placed primes are the long prompts after the learning one, in order.
        rid = long_ids[k + 1] if len(long_ids) > k + 1 else None
        snaps = [(m.group(1), int(m.group(4)), int(m.group(5)), int(m.group(3)))
                 for l in log if (m := SNAP.search(l)) and m.group(2) == rid]
        after = [s for s in snaps if s[0] == "dspark-prime-chunk-after"]
        before = [s for s in snaps if s[0] == "dspark-prime-chunk-before"]
        a1 = after[0] if after else None
        b2 = before[1] if len(before) > 1 else None
        gaps = [int(m.group(2)) for l in log if (m := GAP.search(l)) and m.group(1) == rid]
        ws = [re.sub(r".*GDN scan workspace ", "", w).split(":")[0] for w in row.get("prime_ws", [])]
        print(f"{d.name} | {row['prompt_tokens']} | {row['status']} | {row['margin_seen_mib']} | {row['oom_lines']} | "
              f"{f'{a1[1]}/{a1[2]}' if a1 else '-'} | {f'{b2[1]}/{b2[2]}' if b2 else '-'} | "
              f"{(b2[1] - a1[1]) if a1 and b2 else '-'} | {(gaps[1] >> 20) if len(gaps) > 1 else '-'} | {','.join(ws) or '-'}")
