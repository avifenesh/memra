#!/usr/bin/env python3
"""parts_req.py prof.sqlite TTFT_S [PRE_S] [--json out.json]: per-part GPU time of one served
prefill whose window is the request itself: it starts PRE_S (host time between the request's
arrival and its first kernel, measured from the timestamped server log; default 0.13 s) before
the first kernel after the last >1 s idle gap that precedes the last prime attention launch, and
lasts TTFT_S. Prints the parts (parts.py's table), the idle gaps by size with the kernel after
each large one, and the window's phases: the trunk (to the last fa_prefill_qw_h2p), then the tail."""
import argparse, collections, json, os, re, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parts import classify  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("ttft", type=float); ap.add_argument("pre", type=float, nargs="?", default=0.13)
ap.add_argument("--json")
a = ap.parse_args()
con = sqlite3.connect(a.db)
rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
                   "ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
t_attn = max(e for s, e, n in rows if n.startswith("fa_prefill_qw_h2p"))
first, prev = None, None
for s, e, n in rows:
    if s > t_attn:
        break
    if prev is not None and s - prev > 1e9:
        first = s
    prev = e if prev is None else max(prev, e)
t0 = first - int(a.pre * 1e9); t1 = t0 + int(a.ttft * 1e9)
win = [(s, min(e, t1), n) for s, e, n in rows if t0 <= s < t1]
busy = sum(e - s for s, e, n in win) / 1e9
part = collections.Counter(); kern = collections.Counter(); cnt = collections.Counter()
for s, e, n in win:
    part[classify(n)] += (e - s) / 1e9
    k = n.split("(")[0]; kern[(k, classify(n))] += (e - s) / 1e9; cnt[k] += 1
gaps = collections.Counter(); gsum = collections.Counter(); after = collections.Counter(); asum = collections.Counter()
last = first
for s, e, n in win:
    g = s - last
    if g > 0:
        b = "<10us" if g < 1e4 else "<100us" if g < 1e5 else "<1ms" if g < 1e6 else "<10ms" if g < 1e7 else ">=10ms"
        gaps[b] += 1; gsum[b] += g
        if g >= 1e6:
            k = re.sub(r"\(.*", "", re.sub(r"^void ", "", n))[:56]; after[k] += 1; asum[k] += g
    last = max(last, e)
print(f"request window {a.ttft:.3f} s (host before the first kernel {a.pre:.3f} s); kernels {len(win)}, "
      f"kernel-sum {busy:.3f} s, idle inside {(t1 - first) / 1e9 - busy:.3f} s")
print(f"trunk (first kernel to the last fa_prefill_qw_h2p) {(t_attn - first) / 1e9:.3f} s; after it {(t1 - t_attn) / 1e9:.3f} s")
print(f"\n{'part':44s} {'s':>8s}  share")
for p, v in part.most_common():
    print(f"{p:44s} {v:8.3f}  {100 * v / busy:5.2f}%")
print("\ngap bucket      count     total_s")
for b in ["<10us", "<100us", "<1ms", "<10ms", ">=10ms"]:
    print(f"{b:10s} {gaps[b]:10d} {gsum[b] / 1e9:11.3f}")
print("\nkernels that follow gaps >= 1 ms")
for k, v in sorted(asum.items(), key=lambda x: -x[1])[:10]:
    print(f"{v / 1e9:9.3f} s {after[k]:6d} x  {k}")
tail = collections.Counter(); tcnt = collections.Counter()
for s, e, n in win:
    if s > t_attn:
        k = n.split("(")[0][:56]; tail[k] += (e - s) / 1e9; tcnt[k] += 1
print(f"\nafter the last trunk attention: kernels {sum(tail.values()):.3f} s")
for k, v in tail.most_common(14):
    print(f"  {v:8.3f} s {tcnt[k]:6d} x  {k}")
print("\ntop kernels")
for (k, p), v in kern.most_common(24):
    print(f"  {v:8.3f} s  {cnt[k]:6d} x {1000 * v / cnt[k]:9.3f} ms  [{p}] {k[:60]}")
if a.json:
    json.dump({"ttft_s": a.ttft, "pre_s": a.pre, "kernel_sum_s": busy, "parts": dict(part),
               "trunk_s": (t_attn - first) / 1e9, "tail_kernels_s": sum(tail.values()),
               "gaps": {b: [gaps[b], gsum[b] / 1e9] for b in gaps}}, open(a.json, "w"), indent=1)
