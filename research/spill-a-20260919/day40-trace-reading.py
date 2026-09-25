#!/usr/bin/env python3
"""WP-A day 40 section 6's trace reader, written before the traces exist. Input: an Nsight Systems sqlite export of one
traced demote boot (read-only). The copy stream is the stream that runs `d2h_receipt_sha256` (design G4: every piece of
side work on it). Each launch of that kernel opens one demote window, closed by the next one (or the trace's end). Per
window, on the copy stream: the D2H memcpys (count, summed ms, first start, last end), the `d2d_receipt_digest` kernels
split into those that start before the window's first >64 KiB D2H memcpy's start (the span receipt's source group),
those between the last one's end and the span receipt's lanes D2H (64 bytes per span; the landed group), and those among
the span copies ('between'); the window's later digests (the D2D capture's) are in no group, each group's count, summed ms and wall (first start to
last end), their mean launch-to-start gap (kernel start minus its launch call's end, by correlationId), and the window's
side-work span (the receipt kernel's start to the last copy-stream activity). Printed one line per window, then the
medians over the second and later windows. A reading: nothing here is a clause. Usage: reader SQLITE"""
import sqlite3, statistics as st, sys
from bisect import bisect_left

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))
ker = cur.execute("SELECT start, end, streamId, shortName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
sha = [k for k in ker if names.get(k[3]) == "d2h_receipt_sha256"]
if not sha:
    print("TRACE no d2h_receipt_sha256 kernel"); sys.exit(2)
copy = sha[0][2]
marks = [k[0] for k in sha]
mem = cur.execute(f"SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE streamId = {copy} ORDER BY start").fetchall()
kinds = {}
try:
    kinds = dict(cur.execute("SELECT id, name FROM ENUM_CUDA_MEMCPY_OPER"))
except sqlite3.OperationalError:
    pass
launch_end = {}
for cid, e in cur.execute("SELECT correlationId, end FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
    launch_end[cid] = e
dig = [k for k in ker if k[2] == copy and names.get(k[3]) == "d2d_receipt_digest"]
print(f"TRACE copy_stream={copy} demotes={len(marks)} d2d_receipt_digest_on_copy={len(dig)} copy_memcpys={len(mem)}")
rows = []
for w, a in enumerate(marks):
    b = marks[w + 1] if w + 1 < len(marks) else float("inf")
    m = [x for x in mem if a <= x[0] < b and "DTOH" in str(kinds.get(x[3], x[3]))]
    big = [x for x in m if x[2] > 65536]
    if not big:
        continue
    s0, s1 = min(x[0] for x in big), max(x[1] for x in big)
    # The span receipt's lanes D2H (64 bytes per span) closes the landed group; the D2D capture's digests that come
    # later in the window (its own copies) are not the span receipt's.
    seal = next((x[0] for x in m if x[0] >= s1 and x[2] == 64 * len(big)), None)
    close = seal if seal is not None else s1
    d = [k for k in dig if a <= k[0] < b]
    groups = {"source": [], "landed": [], "between": []}
    for k in d:
        if k[0] < s0:
            groups["source"].append(k)
        elif s1 <= k[0] < close:
            groups["landed"].append(k)
        elif s0 <= k[0] < s1:
            groups["between"].append(k)
    def g_stat(ks):
        if not ks:
            return (0, 0.0, 0.0, float("nan"))
        gaps = [(k[0] - launch_end[k[4]]) / 1e3 for k in ks if k[4] in launch_end]
        return (len(ks), sum(k[1] - k[0] for k in ks) / 1e6, (max(k[1] for k in ks) - min(k[0] for k in ks)) / 1e6,
                st.mean(gaps) if gaps else float("nan"))
    last = max([x[1] for x in m if x[0] <= close] + [k[1] for k in groups["source"] + groups["landed"]])
    r = {
        "w": w + 1,
        "dtoh": (len(m), sum(x[1] - x[0] for x in m) / 1e6, (s1 - s0) / 1e6),
        "src": g_stat(groups["source"]), "land": g_stat(groups["landed"]), "mid": g_stat(groups["between"]),
        "span": (last - a) / 1e6,
    }
    rows.append(r)
    f = lambda t: f"n={t[0]} sum_ms={t[1]:.2f} wall_ms={t[2]:.2f} launch_gap_us={t[3]:.1f}"
    print(f"TRACE demote={r['w']} dtoh n={r['dtoh'][0]} sum_ms={r['dtoh'][1]:.2f} span_copies_wall_ms={r['dtoh'][2]:.2f} | "
          f"source {f(r['src'])} | landed {f(r['land'])} | between {f(r['mid'])} | side_work_span_ms={r['span']:.2f}")
steady = rows[1:]
if steady:
    med = lambda k, i: st.median([r[k][i] for r in steady])
    print(f"TRACE steady N={len(steady)} span_copies_wall_ms={med('dtoh', 2):.2f} source_wall_ms={med('src', 2):.2f} "
          f"landed_wall_ms={med('land', 2):.2f} landed_sum_ms={med('land', 1):.2f} landed_launch_gap_us={med('land', 3):.1f} "
          f"side_work_span_ms={st.median([r['span'] for r in steady]):.2f}")
