#!/usr/bin/env python3
"""Collect a workflow journal into wave1/results.json (scouts + annotated slices), partial runs included."""
import json, sys
J = sys.argv[1]; out = sys.argv[2]
labels = {}
rows = [json.loads(l) for l in open(J)]
for r in rows:
    if r.get('type') == 'started': labels[r['key']] = r['label']
scouts, annotated = {}, []
for r in rows:
    if r.get('type') != 'result': continue
    res = r.get('result'); res = json.loads(res) if isinstance(res, str) else res
    lab = labels.get(r['key'], '')
    if lab.startswith('scout:'): scouts[lab[6:]] = res
    elif lab.startswith('annotate:'):
        if res: res['slice'] = lab.split(':')[1]
        annotated.append(res or {'slice': lab.split(':')[1], 'failed': True})
json.dump({'scouts': scouts, 'annotated': annotated}, open(out, 'w'))
print('scouts', sorted(scouts), 'annotated', sorted(a['slice'] for a in annotated))
