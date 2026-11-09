#!/usr/bin/env python3
"""r5_decode.py prof.sqlite TTFT_S TOTAL_S [ROUNDS]: one served request's decode window (from TTFT to the
request's end, the request located as the last burst of GPU work after an idle gap over 1 s) under
Nsight Systems: kernels by class and by name, the idle between them, device copies, and per-round
figures when the round count is given (the [tick-spec] rounds of the window)."""
import collections, re, sqlite3, sys

db, ttft, total = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 0
con = sqlite3.connect(db)
rows = con.execute("SELECT k.start, k.end, s.value, k.gridX * k.gridY * k.gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                   "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
# the request: the work after the last idle gap over 1 s
start = 0
prev = rows[0][1]
for i in range(1, len(rows)):
    if rows[i][0] - prev > 1e9:
        start = i
    prev = max(prev, rows[i][1])
t_req = rows[start][0]
t_end = max(e for s, e, n, g in rows[start:])
# decode: from TTFT (counted from the request's first kernel; the host time before it is small at 1k)
t0 = t_req + int(ttft * 1e9)
win = [r for r in rows[start:] if r[0] >= t0]
span = (t_end - t0) / 1e6
CLASSES = [
    ("attention", re.compile(r"^fa_")),
    ("gdn scan/conv", re.compile(r"gdn_|ssm_conv")),
    ("matvec/gemm NVFP4", re.compile(r"nvfp4")),
    ("matvec/gemm k-quant (LM head, draft)", re.compile(r"q[456]_K|q45k|q8_0")),
    ("norms", re.compile(r"rms_norm|l2_norm|gated_rmsnorm")),
    ("quantize", re.compile(r"quantize")),
    ("argmax/sampling", re.compile(r"argmax|topk|sample|gumbel|softmax|philox")),
    ("rope/glue/adds", re.compile(r"rope|add_|mul_|silu|sigmoid|q_split|q_gate|copy|scale|embed|inc_|append")),
]
byc = collections.Counter(); byn = collections.Counter(); cnt = collections.Counter()
for s, e, n, g in win:
    n0 = n.split("(")[0]
    c = next((name for name, p in CLASSES if p.search(n0)), "other")
    byc[c] += (e - s) / 1e6; byn[n0[:60]] += (e - s) / 1e6; cnt[n0[:60]] += 1
busy = sum(byc.values())
print(f"decode window {span:.1f} ms ({len(win)} kernels), kernels {busy:.1f} ms, idle {span - busy:.1f} ms"
      + (f", {rounds} rounds: {span / rounds:.2f} ms per round" if rounds else ""))
for c, v in byc.most_common():
    print(f"  {c:40s} {v:9.1f} ms" + (f"  {v / rounds:7.3f} ms/round" if rounds else ""))
print("top kernels:")
for n, v in byn.most_common(22):
    print(f"  {v:9.1f} ms {cnt[n]:6d} x  {n}" + (f"   {v / rounds:.3f} ms/round, {cnt[n] / rounds:.1f}/round" if rounds else ""))
mc = con.execute("SELECT COUNT(*), SUM(end - start), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ?", (t0, t_end)).fetchone()
print(f"memcpy: {mc[0]} copies, {(mc[1] or 0) / 1e6:.1f} ms, {(mc[2] or 0) / 1e6:.1f} MB")
gaps = collections.Counter()
prev = None
for s, e, n, g in win:
    if prev is not None and s > prev:
        d = (s - prev) / 1e3
        k = "<5us" if d < 5 else "<20us" if d < 20 else "<100us" if d < 100 else "<1ms" if d < 1000 else ">=1ms"
        gaps[k] += d / 1e3
    prev = e if prev is None else max(prev, e)
print("idle by gap size (ms):", {k: round(v, 1) for k, v in sorted(gaps.items())})
