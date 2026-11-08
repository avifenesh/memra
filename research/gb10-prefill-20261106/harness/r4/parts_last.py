#!/usr/bin/env python3
"""parts_last.py: per-part GPU kernel time of the LAST served prefill in an nsys sqlite export,
when boot-time work runs back to back with it (no idle gap to split on).

  parts_last.py prof.sqlite TTFT_S [--json out.json]

The window ends at the end of the last prime attention launch (`fa_prefill_qw_*`) and starts
TTFT_S earlier; kernels that start inside it are classified with parts.py's table."""
import argparse, collections, json, os, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parts import classify  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("ttft", type=float)
ap.add_argument("--json")
a = ap.parse_args()
con = sqlite3.connect(a.db)
rows = con.execute("""SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                      JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName)
                      ORDER BY k.start""").fetchall()
ends = [e for s, e, n in rows if n.startswith("fa_prefill_qw")]
t1 = max(ends)
t0 = t1 - int(a.ttft * 1e9)
win = [(s, e, n) for s, e, n in rows if t0 <= s <= t1]
busy = sum(e - s for s, e, n in win) / 1e9
part = collections.Counter(); kern = collections.Counter(); cnt = collections.Counter()
for s, e, n in win:
    part[classify(n)] += (e - s) / 1e9
    short = n.split("(")[0]
    kern[(short, classify(n))] += (e - s) / 1e9; cnt[short] += 1
print(f"window {(t1 - t0) / 1e9:.3f} s ending at the last prime attention launch; kernels {len(win)}, kernel-sum {busy:.3f} s")
print(f"{'part':44s} {'s':>8s}  share")
for p, v in part.most_common():
    print(f"{p:44s} {v:8.3f}  {100 * v / busy:5.2f}%")
print("\ntop kernels")
for (k, p), v in kern.most_common(40):
    print(f"  {v:8.3f} s  {cnt[k]:6d} x {1000 * v / cnt[k]:9.3f} ms  [{p}] {k[:60]}")
if a.json:
    json.dump({"window_s": (t1 - t0) / 1e9, "kernel_sum_s": busy, "parts": dict(part),
               "kernels": {f"{k}|{p}": [v, cnt[k]] for (k, p), v in kern.items()}}, open(a.json, "w"), indent=1)
