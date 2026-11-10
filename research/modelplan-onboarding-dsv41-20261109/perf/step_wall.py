#!/usr/bin/env python3
"""A decode step's whole wall by layer, from an nsys trace of a dsv41-tpbench decode run (sqlite export).

A layer starts at its first mHC mix launch (two a layer on a rank: every second launch of the marker
kernel on the card is a layer start). The window of layer i runs to the start of layer i + 1, so the windows
of one step add up to the step's wall and nothing is left outside: the last layer's window holds the head,
the read-back and the host's work before the next step. The step's phase is found from the data: the
layer index whose window is longest in the median is the step's last.

Prints the median window of every layer index, the typical layer (the median of those), each index that is
more than 10% above it with its excess, and for the heaviest indexes what the excess is made of: kernel
time by name on the card, host driver calls by name (all threads), both against a typical layer.
usage: step_wall.py TRACE.sqlite --layers L [--device D] [--marker NAME] [--top N]
"""
import bisect
import sqlite3
import statistics
import sys
from collections import defaultdict


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


con = sqlite3.connect(sys.argv[1])
con.text_factory = lambda b: b.decode("utf-8", "replace")
cur = con.cursor()
layers, dev, marker, top = arg("--layers", 40), arg("--device", 0), arg("--marker", "hc_mixes_part1"), arg("--top", 4)
names = dict(cur.execute("select id, value from StringIds"))
short = lambda i: names.get(i, str(i)).split("<")[0].replace("v41_", "").replace("_kernel", "")
kern = cur.execute(
    "select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ? order by start", (dev,)
).fetchall()
ks = [k[0] for k in kern]
marks = [k[0] for k in kern if marker in names.get(k[2], "")]
starts = marks[0::2]
if len(starts) < 3 * layers:
    sys.exit(f"{len(starts)} layer starts: fewer than three steps of {layers} layers")
dur = [b - a for a, b in zip(starts, starts[1:])]
# the phase: the residue whose windows are longest is a step's last layer
by_res = defaultdict(list)
for i, d in enumerate(dur):
    by_res[i % layers].append(d)
last = max(by_res, key=lambda r: statistics.median(by_res[r]))
index = lambda i: (i - last - 1) % layers
host = cur.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
hs = [h[0] for h in host]
# whole steps only: from a layer 0 to the layer 0 after it
first0 = next(i for i in range(len(dur)) if index(i) == 0)
nsteps = (len(dur) - first0) // layers
win = defaultdict(list)  # index -> [(t0, t1)]
for i in range(first0, first0 + nsteps * layers):
    win[index(i)].append((starts[i], starts[i + 1]))
med = {x: statistics.median(b - a for a, b in w) for x, w in win.items()}
typ = statistics.median(med.values())
wall = sum(med.values())
print(f"device {dev}: {nsteps} steps of {layers} layers; step wall {wall / 1e3:.0f} us = the sum of the layers' median windows; "
      f"a typical layer {typ / 1e3:.1f} us, {layers} of them {typ * layers / 1e3:.0f} us, the rest {(wall - typ * layers) / 1e3:.0f} us")
print("layer windows, median us, by index: " + " ".join(f"{med[x] / 1e3:.0f}" for x in range(layers)))
heavy = sorted((x for x in med if med[x] > 1.1 * typ), key=lambda x: -med[x])
print("| layer index | window us | above a typical layer |")
print("|---|---|---|")
for x in heavy:
    print(f"| {x} | {med[x] / 1e3:.0f} | {(med[x] - typ) / 1e3:.0f} |")


def profile(ws):
    """Per window: kernel ns by name on the card, host driver ns by name (all threads); the means."""
    kn, hn = defaultdict(float), defaultdict(float)
    for t0, t1 in ws:
        for a, e, n in kern[bisect.bisect_left(ks, t0):bisect.bisect_left(ks, t1)]:
            kn[short(n)] += e - a
        for a, e, n in host[bisect.bisect_left(hs, t0):bisect.bisect_left(hs, t1)]:
            hn[names.get(n, str(n)).split("_v")[0]] += e - a
    return {k: v / len(ws) for k, v in kn.items()}, {k: v / len(ws) for k, v in hn.items()}


plain = [x for x in med if x not in heavy]
base_w = [w for x in plain[: max(4, len(plain) // 4)] for w in win[x]]
bk, bh = profile(base_w)
for x in heavy[:top]:
    k, h = profile(win[x])
    print(f"\nlayer index {x}: {med[x] / 1e3:.0f} us, {(med[x] - typ) / 1e3:.0f} above a typical layer; against a typical layer, us:")
    dk = sorted(((k.get(n, 0) - bk.get(n, 0), n) for n in set(k) | set(bk)), reverse=True)
    print("  kernels on the card (all its streams): " + ", ".join(f"{n} +{d / 1e3:.0f}" for d, n in dk[:8] if d > 2000))
    dh = sorted(((h.get(n, 0) - bh.get(n, 0), n) for n in set(h) | set(bh)), reverse=True)
    print("  host driver calls (all threads, summed): " + ", ".join(f"{n} +{d / 1e3:.0f}" for d, n in dh[:6] if d > 2000))
