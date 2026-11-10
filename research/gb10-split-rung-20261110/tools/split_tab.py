#!/usr/bin/env python3
"""split_tab.py DIR [GQA...]: per-call time by split size from kernel-check's FA-VERIFY-ROWS timing
rows run under MEMRA_FA_SPLIT=S (DIR/NN-spS.log, one process per split; a repeated split is the
drift check and its runs are pooled by median).

For each GQA shape and depth, four launches: the verify's rows walk (T=4, one KV read), the seqs
launch at T=4 (each row walks the cache), and one query row through the seqs launch and through
fa_decode (plain decode at B=1 and the eager step). Each run gives a median of 24 calls; a cell is
the median of S's run medians in ms and its ratio to the 128-key split's, marked `<` when every S
run is under every 128 run, `>` when every S run is over every 128 run, `~` otherwise. The count
of runs per split is printed with the table.
"""
import re
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
want = [int(g) for g in sys.argv[2:]] or [6, 8, 4]
t4 = re.compile(r"FA-VERIFY-ROWS time gqa=(\d+) len0=(\d+) T=4 sp=(\d+): seqs ([\d.]+) ms \(([\d.]+)-([\d.]+)\), "
                r"rows ([\d.]+) ms \(([\d.]+)-([\d.]+)\)")
t1 = re.compile(r"FA-VERIFY-ROWS time1 gqa=(\d+) t_kv=(\d+) T=1 sp=(\d+): seqs ([\d.]+) ms \(([\d.]+)-([\d.]+)\), "
                r"fa_decode ([\d.]+) ms \(([\d.]+)-([\d.]+)\)")
# (gqa, depth keys, launch) -> split -> [(med, lo, hi), ...]
data = {}
for f in sorted(d.glob("*-sp*.log")):
    for line in f.read_text(errors="replace").splitlines():
        for rx, names in ((t4, ("rows T=4", "seqs T=4")), (t1, ("seqs T=1", "fa_decode T=1"))):
            m = rx.search(line)
            if not m:
                continue
            g = [float(x) for x in m.groups()[3:]]
            gqa, sp = int(m.group(1)), int(m.group(3))
            # keyed by the cache depth before the call: T=4 rows hold len0+1..len0+4 keys, T=1 holds len0+1
            depth = int(m.group(2)) - (0 if rx is t4 else 1)
            seqs, other = (g[0], g[1], g[2]), (g[3], g[4], g[5])
            if rx is t4:
                data.setdefault((gqa, depth, names[0]), {}).setdefault(sp, []).append(other)
                data.setdefault((gqa, depth, names[1]), {}).setdefault(sp, []).append(seqs)
            else:
                data.setdefault((gqa, depth, names[0]), {}).setdefault(sp, []).append(seqs)
                data.setdefault((gqa, depth, names[1]), {}).setdefault(sp, []).append(other)


def pool(runs):
    meds = [r[0] for r in runs]
    return (statistics.median(meds), min(meds), max(meds))


splits = sorted({s for v in data.values() for s in v})
runs = {s: max(len(v[s]) for v in data.values() if s in v) for s in splits}
print("runs per split: " + ", ".join(f"sp{s} {runs[s]}" for s in splits))
launches = ["rows T=4", "seqs T=4", "seqs T=1", "fa_decode T=1"]
for gqa in want:
    keys = sorted({(k[1]) for k in data if k[0] == gqa})
    if not keys:
        continue
    print(f"\n### GQA {gqa}\n")
    print("| depth (keys) | launch | " + " | ".join(f"sp{s}" for s in splits) + " |")
    print("|---:|---|" + "---:|" * len(splits))
    for depth in keys:
        for la in launches:
            row = data.get((gqa, depth, la))
            if not row or 128 not in row:
                continue
            base = pool(row[128])
            cells = []
            for s in splits:
                if s not in row:
                    cells.append("")
                    continue
                v = pool(row[s])
                if s == 128:
                    cells.append(f"{v[0]:.4f}")
                    continue
                mark = "<" if v[2] < base[1] else (">" if v[1] > base[2] else "~")  # run medians apart
                cells.append(f"{v[0]:.4f} ({v[0] / base[0]:.3f} {mark})")
            print(f"| {depth:,} | {la} | " + " | ".join(cells) + " |")
