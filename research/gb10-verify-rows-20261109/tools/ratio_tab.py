#!/usr/bin/env python3
"""ratio_tab.py KC.log...: the FA-VERIFY-ROWS timing rows of kernel-check logs as rows/seqs ratios
per GQA shape and launch shape (one session's 4 rows by depth, pooled sessions by depth)."""
import re, sys
pat = re.compile(r"FA-VERIFY-ROWS time gqa=(\d+) (len0=(\d+) T=4|pooled (\d+) sessions x 4 rows depth~(\d+)) sp=(\d+): seqs ([\d.]+) ms \(([\d.]+)-([\d.]+)\), rows ([\d.]+) ms \(([\d.]+)-([\d.]+)\)(?:, rows in pairs ([\d.]+) ms)?")
for f in sys.argv[1:]:
    print(f"== {f}")
    print("| gqa | shape | depth | split keys | seqs ms | rows ms | rows/seqs | ranges apart | pairs/seqs |")
    print("|---:|---|---:|---:|---:|---:|---:|---|---:|")
    for l in open(f, errors="replace"):
        m = pat.search(l)
        if not m:
            continue
        gqa, _, len0, sess, depth, sp, s, slo, shi, r, rlo, rhi, p = m.groups()
        s, r = float(s), float(r)
        shape = "1 session x 4 rows" if len0 else f"{sess} sessions x 4 rows"
        apart = "rows faster" if float(rhi) < float(slo) else ("seqs faster" if float(rlo) > float(shi) else "overlap")
        print(f"| {gqa} | {shape} | {len0 or depth} | {sp} | {s:.4f} | {r:.4f} | {r / s:.2f} | {apart} | {float(p) / s:.2f} |" if p else
              f"| {gqa} | {shape} | {len0 or depth} | {sp} | {s:.4f} | {r:.4f} | {r / s:.2f} | {apart} | |")
