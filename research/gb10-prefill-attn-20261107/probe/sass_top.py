#!/usr/bin/env python3
"""sass_top.py CSV: summarize an `ncu --page source --csv --print-source sass` export: the stall
mix, the instructions with L2 global excessive sectors, shared excessive wavefronts and local
(spill) traffic, and the executed instruction mix per MMA. stdlib only."""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[1]
data = rows[2:]
ix = {h: i for i, h in enumerate(hdr)}


def num(r, k):
    try:
        return float(r[ix[k]].replace(",", ""))
    except (KeyError, ValueError, IndexError):
        return 0.0


tot = {k: sum(num(r, k) for r in data) for k in hdr if k.startswith("stall_") and "Not Issued" not in k}
s = sum(tot.values()) or 1.0
print("stall samples by reason (share of all samples):")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {k:28s} {v / s * 100:5.1f}%")
for key in ("L2 Theoretical Sectors Global Excessive", "L1 Wavefronts Shared Excessive",
            "L2 Theoretical Sectors Local"):
    print(f"\ntop by {key}:  total {sum(num(r, key) for r in data):.3e}")
    for r in sorted(data, key=lambda r: -num(r, key))[:8]:
        if num(r, key) <= 0:
            break
        print(f"  {num(r, key):10.3e}  {r[ix['Source']][:80]}")
print("\ninstruction mix (executed warp instructions):")
mix = {}
for r in data:
    toks = r[ix["Source"]].split()
    if not toks:
        continue
    op = toks[1] if toks[0].startswith("@") and len(toks) > 1 else toks[0]
    op = op.split(".")[0]
    mix[op] = mix.get(op, 0) + num(r, "Instructions Executed")
t = sum(mix.values()) or 1.0
for k, v in sorted(mix.items(), key=lambda kv: -kv[1])[:18]:
    print(f"  {k:10s} {v / t * 100:5.1f}%")
mma = sum(v for k, v in mix.items() if k.endswith("MMA"))
print(f"  total {t:.3e} warp instructions, {t / max(mma, 1):.2f} per MMA")

# stall reasons per opcode (top opcodes by samples)
per = {}
for r in data:
    toks = r[ix["Source"]].split()
    if not toks:
        continue
    op = (toks[1] if toks[0].startswith("@") and len(toks) > 1 else toks[0]).split(".")[0]
    d = per.setdefault(op, {})
    for k in tot:
        d[k] = d.get(k, 0) + num(r, k)
print("\nstall samples by opcode (share of all samples), top reasons:")
for op, d in sorted(per.items(), key=lambda kv: -sum(kv[1].values()))[:8]:
    sv = sum(d.values())
    top = ", ".join(f"{k[6:]} {v / s * 100:.1f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:4])
    print(f"  {op:8s} {sv / s * 100:5.1f}%  [{top}]")
