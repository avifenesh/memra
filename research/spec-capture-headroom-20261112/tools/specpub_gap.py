#!/usr/bin/env python3
"""specpub_gap.py prof.sqlite TTFT_S: the prefix-cache publication of the last request in an
Nsight Systems profile of memra-server (the decode window is found as r5_decode.py finds it:
after the last idle second, from the first token on).

The publication is the one long pause between two decode kernels: the longest gap between
consecutive kernels of the window, with the batched plane copy (`copy_batch_items_u8`) not
counted as a decode kernel. Printed: the gap, the kernels on either side, and what runs inside
it: device copies and memsets (count, ms, MB), `copy_batch_items_u8` launches (count, ms), and
the CUDA runtime calls that overlap it, by name (count, ms inside the gap). Then, over the whole
window, every device copy and memset of 64 MB or more and every `copy_batch_items_u8` launch of
1 ms or more, so a publication elsewhere would show."""
import collections
import sqlite3
import sys

db, ttft = sys.argv[1], float(sys.argv[2])
con = sqlite3.connect(db)
rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                   "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
start = 0
prev = rows[0][1]
for i in range(1, len(rows)):
    if rows[i][0] - prev > 1e9:
        start = i
    prev = max(prev, rows[i][1])
t0 = rows[start][0] + int(ttft * 1e9)
t_end = max(r[1] for r in rows[start:])
win = [(s, e, n.split("(")[0]) for s, e, n in rows[start:] if s >= t0]
batch = [(s, e) for s, e, n in win if n.startswith("copy_batch_items_u8")]
dec = [(s, e, n) for s, e, n in win if not n.startswith("copy_batch_items_u8")]
best = (0, None, None, None, None)
pe, pn = dec[0][1], dec[0][2]
for s, e, n in dec[1:]:
    if s - pe > best[0]:
        best = (s - pe, pe, s, pn, n)
    if e > pe:
        pe, pn = e, n
gap, g0, g1, before, after = best
print(f"decode window {(t_end - t0) / 1e6:.1f} ms, {len(dec)} kernels")
print(f"longest pause between decode kernels: {gap / 1e6:.2f} ms, {(g0 - t0) / 1e6:.1f} ms into the window, "
      f"{before[:48]} -> {after[:48]}")


def inside(table, extra=""):
    q = f"SELECT start, end, bytes{extra} FROM {table} WHERE end > ? AND start < ?"
    return con.execute(q, (g0, g1)).fetchall()


KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
mc = collections.defaultdict(lambda: [0, 0.0, 0.0])
for s, e, b, k in inside("CUPTI_ACTIVITY_KIND_MEMCPY", ", copyKind"):
    x = mc[KIND.get(k, str(k))]
    x[0] += 1; x[1] += (e - s) / 1e6; x[2] += b / 1e6
ms = [0, 0.0, 0.0]
for s, e, b in inside("CUPTI_ACTIVITY_KIND_MEMSET"):
    ms[0] += 1; ms[1] += (e - s) / 1e6; ms[2] += b / 1e6
print("inside the pause:")
for k, (c, t, mb) in sorted(mc.items()):
    print(f"  device copies {k}: {c}, {t:.2f} ms, {mb:.1f} MB")
print(f"  memsets: {ms[0]}, {ms[1]:.2f} ms, {ms[2]:.1f} MB")
bi = [(s, e) for s, e in batch if e > g0 and s < g1]
print(f"  copy_batch_items_u8 launches: {len(bi)}, {sum(e - s for s, e in bi) / 1e6:.2f} ms")
if bi:
    print(f"  pause before the first batched copy starts: {(bi[0][0] - g0) / 1e6:.2f} ms; "
          f"after the last ends: {(g1 - bi[-1][1]) / 1e6:.2f} ms")
api = collections.defaultdict(lambda: [0, 0.0])
try:
    for s, e, name in con.execute(
            "SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId "
            "WHERE r.end > ? AND r.start < ?", (g0, g1)):
        x = api[name.split("_v")[0]]
        x[0] += 1; x[1] += (min(e, g1) - max(s, g0)) / 1e6
    print("  CUDA runtime calls overlapping the pause (count, ms inside it):")
    for name, (c, t) in sorted(api.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"    {name}: {c}, {t:.2f} ms")
except sqlite3.Error as err:
    print(f"  runtime calls: not in this profile ({err})")
print("whole decode window:")
for kind, table in (("device copies", "CUPTI_ACTIVITY_KIND_MEMCPY"), ("memsets", "CUPTI_ACTIVITY_KIND_MEMSET")):
    c, t, mb = con.execute(f"SELECT COUNT(*), SUM(end - start), SUM(bytes) FROM {table} "
                           "WHERE start >= ? AND end <= ? AND bytes >= 64000000", (t0, t_end)).fetchone()
    print(f"  {kind} of 64 MB or more: {c}, {(t or 0) / 1e6:.2f} ms, {(mb or 0) / 1e6:.1f} MB")
big = [(s, e) for s, e in batch if e - s >= 1_000_000]
print(f"  copy_batch_items_u8 launches of 1 ms or more: {len(big)}, {sum(e - s for s, e in big) / 1e6:.2f} ms "
      f"(all launches: {len(batch)})")
