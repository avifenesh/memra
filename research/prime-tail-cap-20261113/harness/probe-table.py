#!/usr/bin/env python3
"""memra#1258: per served cold prime in a b-probe.py log, what it allocated (decimal MB from the
[dflash-alloc] byte readings): the KV and drafter plane (target-kv-before -> draft-kv-after pool
reserved), the prime itself (draft-kv-after -> draft-ingest-after reserved growth, with the pool's
unused bytes at its start), and the gate's charge (cost incl. plane, MiB on the diag line).
usage: probe-table.py LOG..."""
import re
import sys

RE_PH = re.compile(r"\[dflash-alloc\] phase=(\S+) driver_free_bytes=(\d+) driver_total_bytes=\d+ "
                   r"pool_reserved_bytes=(\d+) pool_used_bytes=(\d+)")
RE_PASS = re.compile(r"gate-pass id=\S+ prompt=(\d+) path=(\w+) cost=(\d+)MB draft=(\d+)MB reserve=(\d+)MB "
                     r"required=(\d+)MB")
MB = 1_000_000
for path in sys.argv[1:]:
    print(f"## {path}")
    print("| prompt | path | cost MiB | floor MiB | KV + plane MB | unused at prime MB | prime growth MB | prime + KV + plane MB | over cost + floor MB |")
    print("|---|---|---|---|---|---|---|---|---|")
    cur: dict = {}
    for line in open(path, errors="replace"):
        if line.startswith("boot:"):
            break
        if (m := RE_PASS.search(line)):
            cur = {"prompt": int(m.group(1)), "path": m.group(2), "cost": int(m.group(3)),
                   "floor": int(m.group(5))}
        elif (m := RE_PH.search(line)) and cur:
            ph, _free, res, used = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            cur[ph] = (res, used)
            if ph == "draft-ingest-after" and "target-kv-before" in cur and "draft-kv-after" in cur:
                kvp = cur["draft-kv-after"][0] - cur["target-kv-before"][0]
                unused = cur["draft-kv-after"][0] - cur["draft-kv-after"][1]
                grow = res - cur["draft-kv-after"][0]
                total = kvp + grow + unused
                need = (cur["cost"] + cur["floor"]) * (1 << 20)
                print(f"| {cur['prompt']} | {cur['path']} | {cur['cost']} | {cur['floor']} | {kvp / MB:.0f} | "
                      f"{unused / MB:.0f} | {grow / MB:.0f} | {total / MB:.0f} | {(total - need) / MB:+.0f} |")
                cur = {}
