import json, glob, statistics, re, gzip, csv, sys
R = sys.argv[1]
res = {}
for f in sorted(glob.glob(R + '/ab-rb-final-*.jsonl')):
    m = re.search(r'-(\d+)-(rb|final)-(long|chat)\.jsonl', f)
    rows = [json.loads(l) for l in open(f) if l.strip()]
    timed = [r for r in rows if not r.get('warmup')]
    res.setdefault((m.group(2), m.group(3)), []).append(
        (statistics.median(r['decode_step_p50_ms'] for r in timed), statistics.median(r['prefill_tok_s'] for r in timed),
         {r['ids_fnv1a'] for r in timed}))
for req in ('long', 'chat'):
    for who in ('rb', 'final'):
        v = res[(who, req)]
        st = [x[0] for x in v]; pf = [x[1] for x in v]
        print(f"{req:5s} {who:6s} runs={len(v)} step p50 ms median {statistics.median(st):.2f} [{min(st):.2f}-{max(st):.2f}]  "
              f"prefill tok/s median {statistics.median(pf):.1f} [{min(pf):.1f}-{max(pf):.1f}]  ids {sorted(set().union(*[x[2] for x in v]))}")
t = list(csv.reader(gzip.open(R + '/telemetry-rb-final.csv.gz', 'rt')))
h = [c.strip() for c in t[0]]
def col(name):
    i = [j for j, c in enumerate(h) if c.startswith(name)][0]
    out = []
    for r in t[1:]:
        try: out.append(float(r[i].split()[0]))
        except Exception: pass
    return out
for n in ('temperature.gpu', 'power.draw', 'clocks.current.sm'):
    c = col(n)
    if c: print(f"telemetry {n}: n={len(c)} p50 {statistics.median(c):.1f} min {min(c):.1f} max {max(c):.1f}")
