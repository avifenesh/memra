#!/usr/bin/env python3
"""Where a decode step's wall time goes in an nsys trace of an N-rank decode run (sqlite export).

The decode window is the span of the one-token mHC kernel's launches (it runs only in one-token
forwards, twice a layer on every rank). Inside it, per decode step (a step = RANKS x LAYERS x 2 of
those launches): the wall time, each stream's GPU-busy time (kernels and copies), the kernel and
copy counts, and the host's CUDA calls by name with their count and time per step and the thread
that made them. A step whose streams are busy for a small part of its wall is host-bound.

usage: decode_window.py PROFILE.sqlite --ranks N --layers L [--marker KERNEL_SUBSTRING]
"""
import sqlite3
import sys
from collections import defaultdict


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    db = sys.argv[1]
    ranks, layers = int(arg("--ranks")), int(arg("--layers"))
    marker = arg("--marker", "hc_mixes_part1")
    con = sqlite3.connect(db)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    cur = con.cursor()
    strings = dict(cur.execute("select id, value from StringIds"))
    kern = cur.execute(
        "select start, end, shortName, streamId, deviceId, correlationId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    marks = sorted(k[0] for k in kern if marker in strings.get(k[2], ""))
    if not marks:
        sys.exit(f"no kernel named *{marker}*")
    per_step = ranks * layers * 2
    steps = len(marks) / per_step
    # skip the first step (allocation, first-touch) when there are several
    t0 = marks[per_step] if len(marks) > 2 * per_step else marks[0]
    t1 = marks[-1]
    nsteps = (len(marks) - per_step) / per_step if len(marks) > 2 * per_step else steps
    wall = (t1 - t0) / 1e6
    print(f"decode window: {len(marks)} marker launches = {steps:.2f} steps of {ranks} ranks x {layers} layers; "
          f"measured over {nsteps:.2f} steps, {wall:.3f} ms -> {wall / nsteps:.3f} ms/step, "
          f"{wall / nsteps / layers * 1000:.1f} us/layer")
    busy = defaultdict(float)
    count = defaultdict(int)
    kname = defaultdict(lambda: [0, 0.0])
    for st, en, nm, stream, dev, _ in kern:
        if st >= t0 and en <= t1:
            busy[(dev, stream)] += en - st
            count[(dev, stream)] += 1
            k = kname[strings.get(nm, "?")]
            k[0] += 1
            k[1] += en - st
    ncopy = defaultdict(int)
    for tbl in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            for st, en, stream, dev in cur.execute(f"select start, end, streamId, deviceId from {tbl}"):
                if st >= t0 and en <= t1:
                    busy[(dev, stream)] += en - st
                    ncopy[(dev, stream)] += 1
        except sqlite3.OperationalError:
            pass
    print("\n| device/stream | kernels/step | copies+sets/step | GPU busy ms/step | busy us/layer | busy share of wall |")
    print("|---|---|---|---|---|---|")
    for key in sorted(busy):
        b = busy[key] / 1e6 / nsteps
        print(f"| {key[0]}/{key[1]} | {count[key] / nsteps:.0f} | {ncopy[key] / nsteps:.0f} | {b:.3f} | "
              f"{b / layers * 1000:.1f} | {100 * b / (wall / nsteps):.0f}% |")
    # where a stream sits idle: the gap before each kernel or copy, by what ran before and after it
    ev = defaultdict(list)
    for st, en, nm, stream, dev, _ in kern:
        if st >= t0 and en <= t1:
            ev[(dev, stream)].append((st, en, strings.get(nm, "?").replace("v41_", "").replace("_kernel", "")))
    for tbl, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        try:
            for st, en, stream, dev in cur.execute(f"select start, end, streamId, deviceId from {tbl}"):
                if st >= t0 and en <= t1:
                    ev[(dev, stream)].append((st, en, label))
        except sqlite3.OperationalError:
            pass
    allint = sorted((st, en) for v in ev.values() for st, en, _ in v)
    union, cur_en = 0, t0
    for st, en in allint:
        if en > cur_en:
            union += en - max(st, cur_en)
            cur_en = en
    print(f"\nsome stream busy {union / 1e6 / nsteps:.3f} ms/step ({100 * union / (t1 - t0):.0f}% of wall); "
          f"every stream idle {(t1 - t0 - union) / 1e6 / nsteps:.3f} ms/step")
    key0 = sorted(ev)[0]
    seq = sorted(ev[key0])
    gaps = defaultdict(lambda: [0, 0.0])
    for a, b in zip(seq, seq[1:]):
        g = gaps[(a[2], b[2])]
        g[0] += 1
        g[1] += max(0, b[0] - a[1])
    tot = sum(v[1] for v in gaps.values())
    print(f"\nstream {key0[0]}/{key0[1]}: idle {tot / 1e6 / nsteps:.3f} ms/step = {tot / 1e3 / nsteps / layers:.1f} us/layer, by the events around the gap")
    print("\n| after | before | gaps/layer | idle us/layer | us/gap |")
    print("|---|---|---|---|---|")
    for (a, b), (c, ns) in sorted(gaps.items(), key=lambda x: -x[1][1])[:28]:
        print(f"| {a} | {b} | {c / nsteps / layers:.2f} | {ns / 1e3 / nsteps / layers:.1f} | {ns / c / 1e3:.1f} |")
    # host-bound or device-bound: a kernel whose launch call returned after the stream's previous
    # event ended waited for the host; the rest of its gap is the device (queueing, other streams)
    try:
        issued = dict(cur.execute("select correlationId, end from CUPTI_ACTIVITY_KIND_RUNTIME"))
    except sqlite3.OperationalError:
        issued = {}
    print("\n| device/stream | idle us/layer | waiting for the host's launch | launched, not started | kernels launched after the stream drained |")
    print("|---|---|---|---|---|")
    for key in sorted(ev):
        ks = sorted((st, en, corr) for st, en, _, stream, dev, corr in kern if (dev, stream) == key and st >= t0 and en <= t1)
        ends = sorted(en for _, en, _ in ev[key])
        import bisect
        hostw = devw = 0
        late = 0
        for st, en, corr in ks:
            i = bisect.bisect_right(ends, st) - 1
            if i < 0:
                continue
            prev = ends[i]
            gap = st - prev
            if gap <= 0:
                continue
            h = min(gap, max(0, issued.get(corr, prev) - prev))
            hostw += h
            devw += gap - h
            late += h > 0
        print(f"| {key[0]}/{key[1]} | {(hostw + devw) / 1e3 / nsteps / layers:.1f} | {hostw / 1e3 / nsteps / layers:.1f} | "
              f"{devw / 1e3 / nsteps / layers:.1f} | {100 * late / max(1, len(ks)):.0f}% |")
    # Per layer inside a step: a stream's layer runs from its first one-token mHC launch of the layer
    # to the next layer's; the last layer of a step (the head, the readback, the sampler and the
    # next step's embedding follow it) is left out. The gap before an event is the host's when the
    # event's launch call returned after the stream's previous event ended; what is left of it is a
    # wait for a peer when the event is a copy (a gather waits for its producers), and otherwise
    # an event wait or the card busy with another rank.
    import statistics
    cops = {}
    for tbl, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        try:
            for st, en, stream, dev, corr in cur.execute(f"select start, end, streamId, deviceId, correlationId from {tbl}"):
                cops.setdefault((dev, stream), []).append((st, en, label, corr))
        except sqlite3.OperationalError:
            pass
    print("\n| device/stream | layer us, median | mean | kernels busy | copies busy | waiting for the host | waiting for a peer before a copy | "
          "launched kernel not started | routed-expert tile kernels |")
    print("|---|---|---|---|---|---|---|---|---|")
    tiles = {}
    for key in sorted(ev):
        evs = [(st, en, strings.get(nm, "?"), corr) for st, en, nm, stream, dev, corr in kern if (dev, stream) == key]
        evs += cops.get(key, [])
        evs.sort()
        lstart = [e[0] for e in evs if marker in e[2] and t0 <= e[0] <= t1][0::2]
        spans, acc = [], [0.0] * 6
        nl = 0
        j = 0
        for i in range(len(lstart) - 1):
            if i % layers == layers - 1:
                continue
            a, b = lstart[i], lstart[i + 1]
            spans.append(b - a)
            nl += 1
            while evs[j][0] < a:
                j += 1
            k = j
            tl = 0.0
            while k < len(evs) and evs[k][0] < b:
                st, en, nm, corr = evs[k]
                prev = evs[k - 1][1] if k else st
                gap = max(0, st - prev)
                h = min(gap, max(0, issued.get(corr, prev) - prev))
                iscopy = nm in ("copy", "memset")
                acc[1 if iscopy else 0] += en - st
                acc[2] += h
                acc[3 if iscopy else 4] += gap - h
                if "f8_tiles" in nm:
                    tl += en - st
                k += 1
            acc[5] += tl
            tiles.setdefault(i, []).append(tl)
        if not nl:
            continue
        print(f"| {key[0]}/{key[1]} | {statistics.median(spans) / 1e3:.1f} | {sum(spans) / nl / 1e3:.1f} | "
              + " | ".join(f"{x / nl / 1e3:.1f}" for x in acc) + " |")
    full = [v for v in tiles.values() if len(v) == len(ev)]
    if full:
        print(f"\nrouted-expert tile kernels a layer, us: mean over the ranks {sum(sum(v) / len(v) for v in full) / len(full) / 1e3:.1f}, "
              f"the busiest rank {sum(max(v) for v in full) / len(full) / 1e3:.1f}, the idlest {sum(min(v) for v in full) / len(full) / 1e3:.1f}, "
              f"all ranks together {sum(sum(v) for v in full) / len(full) / 1e3:.1f}")
    api = defaultdict(lambda: [0, 0.0])
    thr = defaultdict(float)
    for tbl in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        try:
            rows = cur.execute(f"select start, end, nameId, globalTid from {tbl}").fetchall()
        except sqlite3.OperationalError:
            continue
        for st, en, nid, tid in rows:
            if st >= t0 and en <= t1:
                a = api[strings.get(nid, "?")]
                a[0] += 1
                a[1] += en - st
                thr[tid] += en - st
    print("\n| host CUDA call | calls/step | calls/layer | ms/step | us/call |")
    print("|---|---|---|---|---|")
    for nm, (c, ns) in sorted(api.items(), key=lambda x: -x[1][1])[:16]:
        print(f"| {nm} | {c / nsteps:.0f} | {c / nsteps / layers:.1f} | {ns / 1e6 / nsteps:.3f} | {ns / c / 1e3:.1f} |")
    print("\nhost threads, time inside CUDA calls per step (ms): " +
          ", ".join(f"{v / 1e6 / nsteps:.3f}" for _, v in sorted(thr.items(), key=lambda x: -x[1])[:8]))
    print("\n| kernel | launches/step | per layer per rank | GPU us/layer (all ranks) |")
    print("|---|---|---|---|")
    for nm, (c, ns) in sorted(kname.items(), key=lambda x: -x[1][1])[:30]:
        print(f"| {nm} | {c / nsteps:.0f} | {c / nsteps / layers / ranks:.2f} | {ns / 1e3 / nsteps / layers:.1f} |")


if __name__ == "__main__":
    main()
