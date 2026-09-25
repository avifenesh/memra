#!/usr/bin/env python3
"""WP-A day 42 (DAY42.md section 1, the reading without a clause) trace reader, written before the traces exist: DAY40
section 6's reader with its grouping fixed to design S2's order. Input: an Nsight Systems sqlite export of one traced
demote boot (read-only). The copy stream is the stream that runs `d2h_receipt_sha256` (design G4: every piece of side
work on it). Each launch of that kernel opens one demote window, closed by the next one (or the trace's end). Per
window, on the copy stream: the D2H memcpys over 64 KiB (count, summed ms, first start, last end: the batch's copies);
the span receipt's kernels, by name (`span_receipt_digests` under S2; under G4 there are none, and under S the reader of
day 40 applies): the SOURCE launches (starting before the first big D2H) and the LANDED launches (starting after the
last big D2H's end: the seal), each group's count, summed ms and wall, and the seal's delay (the landed group's first
start minus the last big D2H's end); the side-work span (the receipt kernel's start to the last copy-stream activity
of the window's demote: its copies, its span kernels and the landed lanes' D2H, the first small D2H after the landed
group). Printed one line per window, then the medians over the second and later windows. A reading: nothing here is a
clause. Usage: reader SQLITE"""
import sqlite3
import statistics as st
import sys

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
cur = db.cursor()
names = dict(cur.execute("SELECT id, value FROM StringIds"))
ker = cur.execute(
    "SELECT start, end, streamId, shortName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
).fetchall()
sha = [k for k in ker if names.get(k[3]) == "d2h_receipt_sha256"]
if not sha:
    print("TRACE no d2h_receipt_sha256 kernel")
    sys.exit(2)
copy = sha[0][2]
marks = [k[0] for k in sha]
mem = cur.execute(
    f"SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE streamId = {copy} ORDER BY start"
).fetchall()
kinds = {}
try:
    kinds = dict(cur.execute("SELECT id, name FROM ENUM_CUDA_MEMCPY_OPER"))
except sqlite3.OperationalError:
    pass
spans = [k for k in ker if k[2] == copy and names.get(k[3]) == "span_receipt_digests"]
print(f"TRACE copy_stream={copy} demotes={len(marks)} span_receipt_digests_on_copy={len(spans)} copy_memcpys={len(mem)}")
rows = []
for w, a in enumerate(marks):
    b = marks[w + 1] if w + 1 < len(marks) else float("inf")
    m = [x for x in mem if a <= x[0] < b and "DTOH" in str(kinds.get(x[3], x[3]))]
    big = [x for x in m if x[2] > 65536]
    if not big:
        continue
    s0, s1 = min(x[0] for x in big), max(x[1] for x in big)
    ks = [k for k in spans if a <= k[0] < b]
    src = [k for k in ks if k[0] < s0]
    land = [k for k in ks if k[0] >= s1]

    def g(group):
        if not group:
            return (0, 0.0, 0.0)
        return (len(group), sum(k[1] - k[0] for k in group) / 1e6, (max(k[1] for k in group) - min(k[0] for k in group)) / 1e6)

    seal_delay = (min(k[0] for k in land) - s1) / 1e6 if land else float("nan")
    lanes = next((x for x in m if land and x[0] >= max(k[1] for k in land) and x[2] <= 65536), None)
    last = max([s1] + [k[1] for k in src + land] + ([lanes[1]] if lanes else []))
    r = {"w": w + 1, "dtoh": (len(big), sum(x[1] - x[0] for x in big) / 1e6, (s1 - s0) / 1e6), "src": g(src),
         "land": g(land), "seal": seal_delay, "span": (last - a) / 1e6, "land_end": (last - s1) / 1e6}
    rows.append(r)
    f = lambda t: f"n={t[0]} sum_ms={t[1]:.2f} wall_ms={t[2]:.2f}"
    print(f"TRACE demote={r['w']} copies n={r['dtoh'][0]} sum_ms={r['dtoh'][1]:.2f} wall_ms={r['dtoh'][2]:.2f} | "
          f"source {f(r['src'])} | landed {f(r['land'])} seal_delay_ms={r['seal']:.2f} | "
          f"receipt_after_copies_ms={r['land_end']:.2f} | side_work_span_ms={r['span']:.2f}")
steady = rows[1:]
if steady:
    med = lambda k, i: st.median([r[k][i] for r in steady])
    seals = [r["seal"] for r in steady if r["seal"] == r["seal"]]
    print(f"TRACE steady N={len(steady)} copies_wall_ms={med('dtoh', 2):.2f} source_wall_ms={med('src', 2):.2f} "
          f"landed_wall_ms={med('land', 2):.2f} landed_sum_ms={med('land', 1):.2f} "
          f"seal_delay_ms={st.median(seals) if seals else float('nan'):.2f} "
          f"receipt_after_copies_ms={st.median([r['land_end'] for r in steady]):.2f} "
          f"side_work_span_ms={st.median([r['span'] for r in steady]):.2f}")
