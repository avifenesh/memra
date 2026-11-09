#!/usr/bin/env python3
"""fold_call.py prof.sqlite: the kernels and device copies of a served prime's last trunk call (the
call from the last embed_gather before the trunk's last 16 h2p launches to the first draft-fill
kernel after them), per kernel name, plus the copies in the same window."""
import collections, sqlite3, sys

con = sqlite3.connect(sys.argv[1])
rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
                   "ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
h2p = [i for i, (s, e, n) in enumerate(rows) if n.startswith("fa_prefill_qw_h2p")]
first16 = h2p[-16]
start_i = max(i for i in range(first16) if rows[i][2].startswith("embed_gather"))
end_i = next((i for i in range(h2p[-1], len(rows)) if rows[i][2].split("(")[0].startswith(("void mul_mat_q_q45k", "append_quantize", "quantize_mmq_q8_1"))), len(rows))
t0, t1 = rows[start_i][0], rows[end_i - 1][1]
c = collections.Counter(); k = collections.Counter()
for s, e, n in rows[start_i:end_i]:
    n = n.split("(")[0][:60]; c[n] += (e - s) / 1e6; k[n] += 1
print(f"last trunk call: {end_i - start_i} kernels, span {(t1 - t0) / 1e6:.1f} ms, kernel sum {sum(c.values()):.1f} ms")
for n, v in c.most_common(30):
    print(f"  {v:9.2f} ms {k[n]:6d} x  {n}")
try:
    mc = con.execute("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ?", (t0, t1)).fetchall()
    tot = collections.defaultdict(lambda: [0, 0.0, 0])
    for s, e, b, kind in mc:
        tot[kind][0] += 1; tot[kind][1] += (e - s) / 1e6; tot[kind][2] += b
    for kind, (n, ms, b) in sorted(tot.items()):
        print(f"  memcpy kind {kind}: {n} copies, {ms:.2f} ms, {b / 1e6:.1f} MB")
    ms_ = con.execute("SELECT COUNT(*), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ? AND end <= ?", (t0, t1)).fetchone()
    print(f"  memset: {ms_[0]} calls, {(ms_[1] or 0) / 1e6:.2f} ms")
except sqlite3.OperationalError as err:
    print("copies:", err)
