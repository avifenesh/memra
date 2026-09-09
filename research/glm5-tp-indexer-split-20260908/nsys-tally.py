#!/usr/bin/env python3
"""Export per-device kernel receipts from one nsys SQLite capture (prime or decode)."""
import csv
import json
import pathlib
import sqlite3
import sys

path = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
cols = {r[1] for r in con.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
assert {'start', 'end', 'deviceId'} <= cols, cols
name = 'demangledName' if 'demangledName' in cols else 'shortName'
assert name in cols
rows = con.execute(f'''select k.deviceId, s.value, count(*), sum(k.end-k.start)
from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.{name}=s.id
group by k.deviceId,s.value order by k.deviceId,sum(k.end-k.start) desc''').fetchall()
assert rows, 'no captured kernels'
def category(name):
    if 'kpool_score' in name: return 'score'
    if 'kpool_select' in name: return 'select'
    if 'kpool_candidates' in name: return 'pack'
    if 'tp_ar_gather_i32' in name: return 'exchange'
    if 'kpool_merge' in name: return 'merge'
    return 'other'
summary = {}
with (out / 'kernels.csv').open('w') as f:
    w = csv.writer(f)
    w.writerow(['device', 'category', 'kernel', 'launches', 'total_ns'])
    for dev, kernel, count, ns in rows:
        cat = category(kernel)
        w.writerow([dev, cat, kernel, count, ns])
        rank = summary.setdefault(str(dev), {'kernel_ns':0, 'indexer_ns':0, 'categories':{}})
        rank['kernel_ns'] += ns
        if cat != 'other': rank['indexer_ns'] += ns
        entry = rank['categories'].setdefault(cat, {'launches':0, 'ns':0})
        entry['launches'] += count
        entry['ns'] += ns
for rank in summary.values():
    rank['indexer_share_of_kernel_time'] = rank['indexer_ns'] / rank['kernel_ns']
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
