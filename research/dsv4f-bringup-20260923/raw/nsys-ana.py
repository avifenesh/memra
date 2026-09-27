#!/usr/bin/env python3
"""Per-device GPU busy/idle over the capture, kernel counts, top kernels, API sync/memcpy totals."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 32
q = lambda s: db.execute(s).fetchall()
names = dict(q("select id, value from StringIds"))
k = q("select deviceId, start, end, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")
m = []
try:
    m = q("select deviceId, start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start")
except Exception: pass
ms = []
try:
    ms = q("select deviceId, start, end from CUPTI_ACTIVITY_KIND_MEMSET")
except Exception: pass
t0 = min(r[1] for r in k); t1 = max(r[2] for r in k)
print(f"capture span {(t1-t0)/1e6:.1f} ms over {steps} steps = {(t1-t0)/1e6/steps:.2f} ms/step")
for dev in sorted(set(r[0] for r in k)):
    iv = sorted([(r[1], r[2]) for r in k if r[0] == dev] + [(r[1], r[2]) for r in m if r[0] == dev] + [(r[1], r[2]) for r in ms if r[0] == dev])
    busy, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None: cs, ce = s, e
        elif s > ce: busy += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    busy += ce - cs
    kd = [r for r in k if r[0] == dev]
    ksum = sum(r[2]-r[1] for r in kd)
    print(f"dev{dev}: kernels {len(kd)} ({len(kd)/steps:.0f}/step) kernel-sum {ksum/1e6/steps:.2f} ms/step busy-union {busy/1e6/steps:.2f} ms/step memcpys {sum(1 for r in m if r[0]==dev)/steps:.0f}/step")
agg = collections.defaultdict(lambda: [0, 0])
for r in k:
    a = agg[(r[0], names.get(r[3], r[3]))]; a[0] += 1; a[1] += r[2]-r[1]
tot = collections.defaultdict(int)
for (d, n), v in agg.items(): tot[d] += v[1]
for dev in sorted(tot):
    print(f"--- dev{dev} top kernels (ms/step, count/step, share)")
    for (d, n), v in sorted(agg.items(), key=lambda x: -x[1][1]):
        if d != dev: continue
        if v[1] < tot[dev] * 0.012: continue
        print(f"  {v[1]/1e6/steps:7.3f} {v[0]/steps:6.0f} {100*v[1]/tot[dev]:5.1f}%  {n[:90]}")
kinds = collections.Counter(); kb = collections.Counter(); kt = collections.Counter()
for r in m: kinds[r[3]] += 1; kb[r[3]] += r[4]; kt[r[3]] += r[2]-r[1]
print("memcpy kind: count/step bytes/step ms/step", {c: (kinds[c]/steps, kb[c]//steps, round(kt[c]/1e6/steps, 3)) for c in kinds})
rt = q("select nameId, count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_RUNTIME group by nameId order by 3 desc limit 14")
print("--- runtime API (count/step, ms/step)")
for n, c, s in rt:
    print(f"  {c/steps:8.0f} {s/1e6/steps:8.3f}  {names.get(n, n)}")
