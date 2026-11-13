#!/usr/bin/env python3
"""round_cost.py SERVER_LOG[.gz]: memra#1266. Per phase-(a) tick of a two-member MTP cohort (two
[tick-spec] rows in the tick): the members' K, the burst's drafted counts from the [spec-acc]
rows before them, the tick's spec_ms, rounds = max(drafted_i / K_i), ms per round, and how many
`pooled MTP verify capped` lines the tick printed. Prints the ticks with capped lines and a
summary of ms per round for K=5+K=5 ticks with and without caps."""
import gzip
import re
import statistics
import sys

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
acc, specs, capped = [], [], 0
rows = []
with op(path, "rt", errors="replace") as f:
    for line in f:
        if "[spec-acc]" in line:
            m = re.search(r"burst=(\d+)/(\d+)", line)
            if m:
                acc.append(int(m.group(2)))
        elif "pooled MTP verify capped" in line:
            capped += 1
        elif "[tick-spec]" in line:
            m = re.search(r"generated=(\d+)->(\d+) .* k=(\d+)", line)
            if m:
                specs.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
        elif "[tick] act=" in line:
            m = re.search(r"spec_ms=([0-9.]+)", line)
            t = float(line.split(" ", 1)[0])
            if m and len(specs) == 2 and len(acc) >= 2:
                ks = [s[2] for s in specs]
                drafted = acc[-2:]
                rounds = max((d + k - 1) // k for d, k in zip(drafted, ks))
                toks = sum(b - a for a, b, _ in specs)
                if rounds > 0 and toks > 4:
                    rows.append((t, ks, drafted, float(m.group(1)), rounds, toks, capped))
            acc, specs, capped = [], [], 0
k55 = [r for r in rows if r[1] == [5, 5]]
for r in rows:
    if r[6]:
        print(f"t={r[0]:.3f} K={r[1]} drafted={r[2]} spec_ms={r[3]:.1f} rounds={r[4]} tokens={r[5]} "
              f"ms/round={r[3] / r[4]:.1f} capped_lines={r[6]}")
for label, sel in (("K5+K5 capped", [r for r in k55 if r[6]]), ("K5+K5 uncapped", [r for r in k55 if not r[6]])):
    if sel:
        per = [r[3] / r[4] for r in sel]
        print(f"{label}: ticks {len(sel)} ms/round median {statistics.median(per):.1f} "
              f"min {min(per):.1f} max {max(per):.1f}; ms/token {sum(r[3] for r in sel) / sum(r[5] for r in sel):.2f}")
