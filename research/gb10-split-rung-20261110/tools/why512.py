#!/usr/bin/env python3
"""The why-512 figures from a split table (split_tab.py's split-tab.md): per GQA shape, launch
and depth, each split's time over 128's; then the comparisons the README states, as ranges over
every launch and depth they claim: 512's gain over 256's past a depth (gain = 1 - ratio), 1,024
against 512, and 1,024 against 128 at the shallow depths.
usage: why512.py split-tab.md [GQA ...]   (default GQA 6 and 8: 24/4 and 16/2)"""
import re, sys

path = sys.argv[1]
want = [int(g) for g in sys.argv[2:]] or [6, 8]
tab, gqa = {}, None
for line in open(path):
    m = re.match(r"### GQA (\d+)", line)
    if m:
        gqa = int(m.group(1)); continue
    m = re.match(r"\| ([\d,]+) \| ([^|]+?) \| ([\d.]+) \| (.+)\|\s*$", line)
    if m and gqa is not None:
        depth = int(m.group(1).replace(",", "")); launch = m.group(2).strip()
        cells = [c.strip() for c in m.group(4).split("|")]
        ratios = [float(re.search(r"\(([\d.]+)", c).group(1)) if "(" in c else None for c in cells]
        tab[(gqa, depth, launch)] = dict(zip((256, 512, 1024), ratios))
launches = ["rows T=4", "seqs T=4", "seqs T=1", "fa_decode T=1"]
def rng(xs):
    xs = [x for x in xs if x is not None]
    return f"{min(xs):.3f} to {max(xs):.3f}" if xs else "n/a"
for g in want:
    depths = sorted({d for (gg, d, _) in tab if gg == g})
    print(f"== GQA {g}")
    for d in depths:
        if d < 16_000:
            continue
        row = []
        for l in launches:
            r = tab.get((g, d, l))
            if r:
                row.append(f"{l}: 256 {r[256]:.3f} 512 {r[512]:.3f} 1024 {r[1024]:.3f}")
        print(f"  {d:>7}: " + " | ".join(row))
    for lo in (32_768, 65_536):
        more = [((1 - tab[(g, d, l)][512]) / (1 - tab[(g, d, l)][256]) - 1)
                for d in depths if d >= lo for l in launches
                if (g, d, l) in tab and tab[(g, d, l)][256] < 1]
        share = [((1 - tab[(g, d, l)][256]) / (1 - tab[(g, d, l)][512]))
                 for d in depths if d >= lo for l in launches
                 if (g, d, l) in tab and tab[(g, d, l)][512] < 1]
        print(f"  from {lo}: 512's gain over 256's, more by {rng([m * 100 for m in more])} %; 256's gain as a share of 512's {rng(share)}")
    for d in (131_072, 262_132):
        vs = {l: tab[(g, d, l)][1024] / tab[(g, d, l)][512] for l in launches if (g, d, l) in tab}
        print(f"  1024 / 512 at {d}: " + ", ".join(f"{l} {v:.3f}" for l, v in vs.items()))
    for d in (32_768, 49_152, 65_536):
        vs = {l: tab[(g, d, l)][1024] / tab[(g, d, l)][512] for l in launches if (g, d, l) in tab}
        if vs:
            print(f"  1024 / 512 at {d}: " + ", ".join(f"{l} {v:.3f}" for l, v in vs.items()))
    for d in (16_000, 17_000, 24_576, 32_768):
        vs = {l: tab[(g, d, l)][1024] for l in ("seqs T=1", "fa_decode T=1") if (g, d, l) in tab}
        if vs:
            print(f"  1024 / 128, one row, at {d}: " + ", ".join(f"{l} {v:.3f}" for l, v in vs.items()))
