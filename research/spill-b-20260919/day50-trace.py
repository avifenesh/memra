#!/usr/bin/env python3
"""DAY50 stage 0 reader (DAY50.md 1.2): one `callcost` run's kernel trace, split into calls.

usage: day50-trace.py <card> <L> <rows,comma> <reps> <cuda_gpu_trace.csv> <callcost stdout>
The probe leaves an idle gap (--gap-ms, 50 ms) before every restore and every timed call, so the trace's clusters
(events separated by at least 20 ms of idle GPU) read: leading work (the weight upload, the setup prime and its
snapshot), then per R and rep (one warm-up rep first) a restore cluster and a call cluster. Addendum B: the pairs are
the last 2 x len(rows) x (reps + 1) clusters, each restore cluster memops only and each call cluster with kernels. Per timed call: the GPU span (first start to last end),
the GPU-busy time (the union of kernel and memop intervals), the idle gaps inside the span, and the busy time in
full-attention, GDN/conv, GEMM and other kernels. The host wall comes from the probe's own `callcost` line.
The rule of 1.3 reads `busy_share = busy / wall` of the 32-row call.
"""
import csv, re, statistics, sys

card, L, rows, reps, trace, stdout = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], \
    int(sys.argv[4]), sys.argv[5], sys.argv[6]
GAP_NS = 20_000_000

ev = []
with open(trace, newline="") as f:
    for r in csv.DictReader(f):
        try:
            st = int(float(r.get("Start (ns)") or r.get("Start") or 0))
            du = int(float(r.get("Duration (ns)") or r.get("Duration") or 0))
        except ValueError:
            continue
        ev.append((st, st + du, r.get("Name") or ""))
ev.sort()
clusters = []
for s, e, n in ev:
    if clusters and s - clusters[-1][-1][1] < GAP_NS:
        clusters[-1].append((s, e, n))
    else:
        clusters.append([(s, e, n)])


def union(evs):
    tot, cur_s, cur_e = 0, None, None
    for s, e, _ in sorted(evs):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


CAT = [("attn", re.compile(r"flash|attn|fa3|sdpa|softmax", re.I)),
       ("gdn", re.compile(r"gdn|ssm|conv|delta|chunk_state|wy", re.I)),
       ("gemm", re.compile(r"gemm|cutlass|mma|mmq|mmvq|nvfp4|cublas|sm1[02]0|matmul", re.I))]


def cat_of(name):
    for c, rx in CAT:
        if rx.search(name):
            return c
    return "other"


walls = {}
for line in open(stdout, errors="replace"):
    m = re.match(r"callcost L=(\d+) R=(\d+) N=\d+ wall_ms p50=([\d.]+) .*all=\[([\d.,]*)\]", line)
    if m:
        walls[int(m.group(2))] = [float(x) for x in m.group(4).split(",") if x]
# DAY50 addendum B: the calls are the LAST pairs; anything before them (the weight upload, the setup prime and its
# snapshot) is leading work. Every pair's first cluster must be memops only (the restore), its second must hold kernels.
pairs = len(rows) * (reps + 1)
lead = len(clusters) - 2 * pairs
memop = lambda n: n.startswith("[CUDA mem")
shape_ok = lead >= 1 and all(
    all(memop(x[2]) for x in clusters[lead + 2 * k]) and any(not memop(x[2]) for x in clusters[lead + 2 * k + 1])
    for k in range(pairs)) if lead >= 1 else False
print(f"DAY50 S0-TRACE card={card} L={L} clusters={len(clusters)} leading={lead} pairs={pairs} "
      f"restore_call_shape={'ok' if shape_ok else 'off'}"
      + ("" if shape_ok else " (the per-call split is not read)"))
if not shape_ok:
    sys.exit(0)
i = lead
for r in rows:
    spans, busys, cats = [], [], {"attn": [], "gdn": [], "gemm": [], "other": []}
    for rep in range(reps + 1):
        call = clusters[i + 1]
        i += 2
        if rep == 0:
            continue
        span = call[-1][1] - call[0][0] if call else 0
        spans.append(span / 1e6)
        busys.append(union(call) / 1e6)
        for c in cats:
            cats[c].append(union([x for x in call if cat_of(x[2]) == c]) / 1e6)
    w = walls.get(r, [])
    wp = statistics.median(w) if w else float("nan")
    bp = statistics.median(busys)
    sp = statistics.median(spans)
    print(f"DAY50 S0 card={card} L={L} R={r} N={len(busys)} wall_ms p50={wp:.2f} gpu_span_ms p50={sp:.2f} "
          f"gpu_busy_ms p50={bp:.2f} busy_share={bp / wp if wp == wp and wp else float('nan'):.3f} "
          f"in_span_gaps_ms p50={sp - bp:.2f} host_outside_span_ms p50={wp - sp:.2f} "
          + " ".join(f"{c}_ms={statistics.median(v):.2f}" for c, v in cats.items()))
