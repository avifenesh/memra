#!/usr/bin/env python3
"""Second chance for refuted claims that narrowed families: re-check the family restriction
alone (any route, any device, any env setting). Writes verify/fam-NN.json."""
import json, sys
H = os.environ.get("MEMRA_IMPACT_WORK", os.path.expanduser("~/.cache/memra-impact-map-build"))
res = json.load(open(f'{H}/wave1/results.json'))
claims = {}
for a in res['annotated']:
    for i, rg in enumerate(a.get('regions', [])):
        claims[f"{a['slice']}:R{i}"] = rg
    for j, fd in enumerate(a.get('file_defaults', [])):
        claims[f"{a['slice']}:F{j}"] = dict(fd, line_start=None, line_end=None, whole_file=True)
v = json.load(open(f'{H}/wave2/verdicts-narrow.json'))
out = []
for x in v:
    rg = claims.get(x['claim'])
    if not x['refuted'] or not rg or '*' in (rg.get('families') or ['*']):
        continue
    out.append({"claim": x['claim'] + "~fam", "file": rg['file'],
                "lines": "whole file (file default)" if rg.get('whole_file') else [rg['line_start'], rg['line_end']],
                "label": (rg.get('label') or rg.get('note') or '')[:160], "predicate": (rg.get('predicate') or '')[:200],
                "families": rg['families'], "routes": ["*"], "devices": ["*"], "door": None,
                "basis": (rg.get('narrowing_basis') or '')[:300],
                "earlier_refutation": x['reason'][:400],
                "check": "family-only claim: only the listed families can EVER execute this region, on any route, any device and any env setting. Routes and devices are no longer claimed. Refute if any other family can execute any line of it (strictly after the first line)."})
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
for k in range(0, len(out), n):
    json.dump(out[k:k+n], open(f'{H}/verify/fam-{k//n:02d}.json', 'w'), indent=1)
print(len(out), 'family-only claims in', (len(out) + n - 1) // n, 'batches')
