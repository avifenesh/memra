#!/usr/bin/env python3
"""Day 35 POST-HOC description (written after the cell ran; not pre-registered; decides nothing).

For every arm receipt of rtx5090-day35: per arm run, the tenant's ITL samples above three times the run's own p50
("stretched ticks", A day 16's descriptive term), their count, their sum, and the two largest gaps; per arm the median
count, median sum and median top-2 gaps over the 10 runs. Beside the ON demote runs, `demote_in - completion` per run.
This places where the ON demote's `in - completion` lands (the worst tick, a second tick, or neither) once the
pre-registered reader has read `on_minus_off` of the worst-tick stall. Descriptive only.

    day35-gaps-posthoc.py <ev_dir>
"""
import json
import os
import re
import statistics
import sys

RE_COMPLETION = re.compile(r"demote published off the tick: ticket seq=\d+ complete after \d+ poll\(s\), ([0-9.]+)ms from submission to completion")
ORDER = {1: ("prime", "off", "on"), 2: ("on", "off", "prime")}


def main():
    ev = sys.argv[1]
    for p in (1, 2):
        for kind in ORDER[p]:
            for mode in (("prime",) if kind == "prime" else ("demote", "promote")):
                path = os.path.join(ev, f"pass{p}", kind, mode, "receipt.json")
                rec = json.load(open(path))
                runs = [r for r in rec["runs"] if r["arm"] != "idle" and "stall_ms" in r]
                counts, sums, top1, top2, imc = [], [], [], [], []
                for r in runs:
                    big = [x for x in r["itl_ms"] if x > 3 * r["p50"]]
                    g = sorted(r["itl_ms"], reverse=True)
                    counts.append(len(big)); sums.append(sum(big)); top1.append(g[0]); top2.append(g[1])
                    comps = [float(m.group(1)) for ln in r["server_log_lines"] for m in [RE_COMPLETION.search(ln)] if m]
                    imc += [a - b for a, b in zip(r["server_demote_ms"], comps)]
                    print(f"  pass{p}/{kind}/{mode} run {r['run_id']}: p50={r['p50']:.2f} stretched={[round(x, 1) for x in big]} sum={sum(big):.1f} "
                          f"top2={[round(g[0], 1), round(g[1], 1)]} stall={r['stall_ms']:.1f} demote_in={r['server_demote_ms']} promote_in={r['server_promote_ms']}")
                print(f"DAY35 GAPS pass={p} arm={kind}/{mode} N={len(runs)} stretched_count_median={statistics.median(counts):.0f} "
                      f"stretched_sum_median={statistics.median(sums):.1f} top1_median={statistics.median(top1):.1f} top2_median={statistics.median(top2):.1f} "
                      f"top1_plus_top2_median={statistics.median([a + b for a, b in zip(top1, top2)]):.1f}"
                      + (f" in_minus_completion_median={statistics.median(imc):.1f} (N={len(imc)})" if imc else ""))


if __name__ == "__main__":
    main()
