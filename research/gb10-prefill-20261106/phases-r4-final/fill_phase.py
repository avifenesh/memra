#!/usr/bin/env python3
"""fill_phase.py prof.sqlite TTFT_S [PRE_S]: the MTP draft fill of one served prefill: the kernels between the
trunk's last call (its last kernel before the first draft-fill kernel after the last h2p launch)
and the first decode-shaped kernel, per kernel name, with the span, the kernel sum and the idle."""
import collections, sqlite3, sys

con = sqlite3.connect(sys.argv[1])
rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
                   "ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
h2p = [i for i, (s, e, n) in enumerate(rows) if n.startswith("fa_prefill_qw_h2p")]
start = next(i for i in range(h2p[-1], len(rows)) if rows[i][2].split("(")[0].startswith(("void mul_mat_q_q45k", "append_quantize", "quantize_mmq_q8_1")))
# the request window (parts_req.py's): from the first kernel after the last idle gap over 1 s before
# the trunk's last h2p, less the host time before it, for TTFT seconds; the fill is the window's
# kernels from the first draft-fill kernel on
ttft = float(sys.argv[2]); pre = float(sys.argv[3]) if len(sys.argv) > 3 else 0.13
first, prev = None, None
for s_, e_, n_ in rows:
    if s_ > rows[h2p[-1]][1]:
        break
    if prev is not None and s_ - prev > 1e9:
        first = s_
    prev = e_ if prev is None else max(prev, e_)
t_end = first - int(pre * 1e9) + int(ttft * 1e9)
seg = [r for r in rows[start:] if r[0] < t_end]
t0, t1 = seg[0][0], max(e for s, e, n in seg)
c = collections.Counter(); k = collections.Counter()
for s, e, n in seg:
    n = n.split("(")[0][:64]; c[n] += (e - s) / 1e6; k[n] += 1
busy = sum(c.values())
print(f"draft fill: {len(seg)} kernels, span {(t1 - t0) / 1e6:.1f} ms, kernel sum {busy:.1f} ms, idle {(t1 - t0) / 1e6 - busy:.1f} ms")
for n, v in c.most_common(25):
    print(f"  {v:9.2f} ms {k[n]:6d} x  {n}")
mc = con.execute("SELECT COUNT(*), SUM(end - start), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ?", (t0, t1)).fetchone()
print(f"  memcpy: {mc[0]} copies, {(mc[1] or 0) / 1e6:.2f} ms, {(mc[2] or 0) / 1e6:.1f} MB")
