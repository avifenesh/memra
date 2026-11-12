#!/usr/bin/env python3
"""specpub_split.py prof.sqlite TTFT_S: what the remaining publication pause is made of, from the
same Nsight Systems profile and the same pause specpub_gap.py reads (the longest gap between two
decode kernels of the last request's decode window).

Printed:
  host    the allocation calls' (cuMemAllocAsync) time on the calling thread inside the pause, the
          copy launches' time, every other runtime call's time, and the host's last call's end
  device  the device copies (plane = 64 MB or more, the 34 trunk and draft K/V planes; small:
          count, ms, MB, GB/s counting a read and a write), the memsets, and the device's busy
          time inside the pause (the union of copies, memsets and kernels), so idle = pause -
          busy; then the runtime calls the host was in while the device idled, and the
          allocation calls' spread (the longest five, how many took 1 ms or more)
  path    pause start -> first plane copy starts (the host before the device has work), the first
          plane copy's start -> the last's end, the last plane copy's end -> the pause's end."""
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
dec = [(s, e) for s, e, n in rows[start:] if s >= t0 and not n.startswith("copy_batch_items_u8")]
best = (0, None, None)
pe = dec[0][1]
for s, e in dec[1:]:
    if s - pe > best[0]:
        best = (s - pe, pe, s)
    pe = max(pe, e)
gap, g0, g1 = best
ms = lambda ns: ns / 1e6
print(f"pause {ms(gap):.2f} ms")

copies = {c: (s, e, b) for s, e, b, c in con.execute(
    "SELECT start, end, bytes, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE end > ? AND start < ?", (g0, g1))}
calls = con.execute(
    "SELECT r.start, r.end, s.value, r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId "
    "WHERE r.end > ? AND r.start < ? ORDER BY r.start", (g0, g1)).fetchall()
inside = lambda s, e: max(0, min(e, g1) - max(s, g0))
allocs, other, launch = [], 0, 0
pending = None
for s, e, name, corr in calls:
    n = name.split("_v")[0]
    if n == "cuMemAllocAsync":
        if pending is not None:
            allocs.append((pending, None))
        pending = inside(s, e)
    elif n.startswith("cuMemcpyDtoDAsync"):
        launch += inside(s, e)
        if pending is not None:
            allocs.append((pending, copies.get(corr)))
            pending = None
    else:
        other += inside(s, e)
if pending is not None:
    allocs.append((pending, None))
plane = [a for a, c in allocs if c and c[2] >= 64_000_000]
small = [a for a, c in allocs if not (c and c[2] >= 64_000_000)]
last_host = max(min(e, g1) for s, e, n, c in calls) if calls else g0
print("host (the calling thread, inside the pause):")
print(f"  allocations: {len(allocs)}, {ms(sum(plane) + sum(small)):.2f} ms")
print(f"  copy launches: {ms(launch):.2f} ms; every other runtime call: {ms(other):.2f} ms")
print(f"  the last runtime call ends {ms(last_host - g0):.2f} ms into the pause")

dc = [(s, e, b) for s, e, b in copies.values()]
pc = [x for x in dc if x[2] >= 64_000_000]
sc = [x for x in dc if x[2] < 64_000_000]
print("device (inside the pause):")
for label, xs in (("plane copies", pc), ("small copies", sc)):
    t = sum(inside(s, e) for s, e, b in xs)
    mb = sum(b for s, e, b in xs) / 1e6
    bw = 2 * mb / 1e3 / (t / 1e9) if t else 0
    print(f"  {label}: {len(xs)}, {ms(t):.2f} ms, {mb:.1f} MB, {bw:.0f} GB/s read + write")
mset = con.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE end > ? AND start < ?", (g0, g1)).fetchall()
print(f"  memsets: {len(mset)}, {ms(sum(inside(s, e) for s, e, b in mset)):.2f} ms, {sum(b for s, e, b in mset) / 1e6:.1f} MB")
iv = sorted([(max(s, g0), min(e, g1)) for s, e, b in dc + mset] +
            [(max(s, g0), min(e, g1)) for s, e, n in rows[start:] if e > g0 and s < g1])
busy, cur_s, cur_e = 0, None, None
for s, e in iv:
    if e <= s:
        continue
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
if cur_e is not None:
    busy += cur_e - cur_s
print(f"  busy {ms(busy):.2f} ms, idle {ms(gap - busy):.2f} ms")
# The device's idle intervals inside the pause, and the runtime calls the host was in meanwhile.
merged, cs, ce = [], None, None
for s, e in iv:
    if e <= s:
        continue
    if ce is None or s > ce:
        if ce is not None:
            merged.append((cs, ce))
        cs, ce = s, e
    else:
        ce = max(ce, e)
if ce is not None:
    merged.append((cs, ce))
idle, t = [], g0
for s, e in merged:
    if s > t:
        idle.append((t, s))
    t = max(t, e)
if t < g1:
    idle.append((t, g1))
by = {}
for a, b in idle:
    for s, e, name, corr in calls:
        o = max(0, min(e, b) - max(s, a))
        if o:
            n = name.split("_v")[0]
            by[n] = by.get(n, 0) + o
print("  the host's calls while the device idles (ms): " +
      ", ".join(f"{n} {ms(v):.2f}" for n, v in sorted(by.items(), key=lambda kv: -kv[1])[:5]))
al = sorted((min(e, g1) - max(s, g0) for s, e, name, c in calls if name.startswith("cuMemAllocAsync")), reverse=True)
print(f"  allocation calls: {len(al)}, {ms(sum(al)):.2f} ms; the longest five {', '.join(f'{ms(x):.2f}' for x in al[:5])}; "
      f"{sum(1 for x in al if x >= 1e6)} of 1 ms or more ({ms(sum(x for x in al if x >= 1e6)):.2f} ms), "
      f"{sum(1 for x in al if x < 1e6)} under 1 ms ({ms(sum(x for x in al if x < 1e6)):.2f} ms)")
if pc:
    f0 = min(s for s, e, b in pc)
    f1 = max(e for s, e, b in pc)
    print("path:")
    print(f"  pause start -> first plane copy starts: {ms(f0 - g0):.2f} ms")
    print(f"  first plane copy starts -> last plane copy ends: {ms(f1 - f0):.2f} ms")
    print(f"  last plane copy ends -> pause ends: {ms(g1 - f1):.2f} ms")
