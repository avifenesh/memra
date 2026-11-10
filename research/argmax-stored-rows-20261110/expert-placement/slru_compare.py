#!/usr/bin/env python3
"""slru_compare.py RESIDENT_LOG SLRU_LOG: the 12-row windows of the same cell with the MoE experts
resident and in the SLRU cache, side by side: per row both programs' top-1 and margin under each
placement, the margin differences, and whether any top-1 changed. Markdown on stdout."""
import sys


def table(path):
    rows = {}
    for line in open(path, errors="replace"):
        f = line.split()
        if len(f) >= 7 and f[0].isdigit() and f[1].isdigit():
            rows[int(f[0])] = (int(f[1]), float(f[2]), int(f[3]), float(f[4]), f[6])
    return rows


res, slru = table(sys.argv[1]), table(sys.argv[2])
print("| row | prefill top-1, margin (resident) | prefill (SLRU) | decode top-1, margin (resident) | decode (SLRU) | prefill margin diff | decode margin diff | top-1 changed |")
print("|---|---|---|---|---|---|---|---|")
dp_max = dd_max = 0.0
changed = 0
for pos in sorted(res):
    a, b = res[pos], slru[pos]
    dp, dd = abs(a[1] - b[1]), abs(a[3] - b[3])
    dp_max, dd_max = max(dp_max, dp), max(dd_max, dd)
    ch = [side for side, i in (("prefill", 0), ("decode", 2)) if a[i] != b[i]]
    changed += bool(ch)
    print(f"| {pos} | {a[0]}, {a[1]:.4f} | {b[0]}, {b[1]:.4f} | {a[2]}, {a[3]:.4f} | {b[2]}, {b[3]:.4f} | "
          f"{dp:.4f} | {dd:.4f} | {', '.join(ch) or 'no'} |")
print(f"\nrows {len(res)}; largest prefill margin difference {dp_max:.4f}, decode {dd_max:.4f}; "
      f"rows with a top-1 change {changed}")
