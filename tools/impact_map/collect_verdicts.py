#!/usr/bin/env python3
"""Merge skeptic journals into wave2/verdicts-narrow.json and wave2/verdicts-defects.json."""
import json, sys, glob
narrow, defects = {}, {}
for J in sys.argv[1:]:
    for l in open(J):
        r = json.loads(l)
        if r.get('type') != 'result': continue
        res = r['result']; res = json.loads(res) if isinstance(res, str) else res
        for v in (res or {}).get('verdicts', []):
            if 'claim' in v: narrow[v['claim']] = v
            elif 'id' in v: defects[v['id']] = v
json.dump(list(narrow.values()), open('wave2/verdicts-narrow.json', 'w'), indent=0)
json.dump(list(defects.values()), open('wave2/verdicts-defects.json', 'w'), indent=0)
import collections
print('narrow', len(narrow), collections.Counter(v['refuted'] for v in narrow.values()), 'defects', len(defects), collections.Counter(v['verdict'] for v in defects.values()))
