#!/usr/bin/env python3
"""Aggregate [glm5-phase] / [glm5-phase-v] burst lines into per-boot medians.

Input: cell dirs containing glm5-phase-lines.txt / glm5-phase-v-lines.txt (c2_phase.sh).
Per-round ms fields are parsed from the tail of each burst line; the reported statistic
is the ROUNDS-WEIGHTED mean plus the plain median across bursts (bursts differ in round
count, so the weighted mean is the honest per-round figure; both are printed).
Shares, never walls (standing trace law).
"""
import os, re, statistics as st, sys

PHASE_RE = re.compile(
    r"\[glm5-phase\] rounds=(\d+) k=(\d+) .*per-round ms: draft=([\d.]+) verify=([\d.]+) "
    r"accept=([\d.]+) roll=([\d.]+) maint=([\d.]+) total=([\d.]+)")
V_RE = re.compile(
    r"\[glm5-phase-v\] rounds=(\d+) k=(\d+) \| per-round ms: vkda=([\d.]+) \(scan=([\d.]+)\) "
    r"vmla=([\d.]+) vrest=([\d.]+)")


def agg(path, rex, names):
    rows = []
    if not os.path.exists(path):
        return None
    for line in open(path):
        m = rex.search(line)
        if m:
            g = m.groups()
            rows.append((int(g[0]), int(g[1]), [float(x) for x in g[2:]]))
    if not rows:
        return None
    tot_rounds = sum(r for r, _, _ in rows)
    out = {"bursts": len(rows), "rounds": tot_rounds, "k": rows[0][1]}
    for i, name in enumerate(names):
        w = sum(r * v[i] for r, _, v in rows) / tot_rounds
        med = st.median(v[i] for _, _, v in rows)
        out[name] = (round(w, 3), round(med, 3))
    return out


def main():
    for d in sys.argv[1:]:
        name = os.path.basename(d)
        p = agg(os.path.join(d, "glm5-phase-lines.txt"), PHASE_RE,
                ["draft", "verify", "accept", "roll", "maint", "total"])
        v = agg(os.path.join(d, "glm5-phase-v-lines.txt"), V_RE,
                ["vkda", "scan", "vmla", "vrest"])
        print(f"== {name} ==")
        if p:
            print(f"  [glm5-phase]   bursts={p['bursts']} rounds={p['rounds']} k={p['k']} "
                  f"(rounds-weighted mean, median) per-round ms:")
            for f in ("draft", "verify", "accept", "roll", "maint", "total"):
                print(f"    {f:<7} {p[f][0]:>8.3f}  (med {p[f][1]:.3f})")
        else:
            print("  [glm5-phase]   NO LINES")
        if v:
            print(f"  [glm5-phase-v] bursts={v['bursts']} rounds={v['rounds']} k={v['k']} "
                  f"(rounds-weighted mean, median) per-round ms:")
            for f in ("vkda", "scan", "vmla", "vrest"):
                print(f"    {f:<7} {v[f][0]:>8.3f}  (med {v[f][1]:.3f})")
        else:
            print("  [glm5-phase-v] NO LINES (expected on the =0 per-row arm: sub-split "
                  "accumulators only tick inside the batched walk)")


if __name__ == "__main__":
    main()
