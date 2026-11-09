#!/usr/bin/env python3
"""Per-arm serving table for an agentic phase: pooled over the arm's runs, per pass, TTFT, E2E and
TPOT p50/p95/p99 (TPOT = (wall - TTFT) / (tokens - 1) per request), failed requests (non-200),
request and token throughput (median run), the device free floor (lowest run) and the counters
read from each run's serve.log. ITL is not recorded by this harness; TPOT stands in.

  python3 serving_table.py PHASE_DIR [--md]
"""
import json
import re
import statistics
import sys
from pathlib import Path

phase = Path(sys.argv[1])


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def free_floor(run):
    vals = []
    for line in open(run / 'gpu-tel.csv', errors='replace'):
        p = [x.strip() for x in line.split(',')]
        if len(p) >= 3 and p[2].endswith('MiB'):
            vals.append(int(p[2].split()[0]))
    return min(vals) if vals else None


def counts(run):
    log = (run / 'serve.log').read_text(errors='replace')
    return {
        'oom': len(re.findall(r'out of memory|CUDA_ERROR_OUT_OF_MEMORY', log, re.I)),
        'defer': log.count('VRAM defer'),
        'yield': len(re.findall(r'\[prefix-cache\] reserve: .*transient reserve after an insert', log)),
        'prime_yield': len(re.findall(r'\[prefix-cache\] reserve: .*prime call needs', log)),
        'edf': log.count('earliest deadline first'),
        'batch_retry': log.count('retrying the same batched step'),
    }


arms = {}
for run in sorted(p for p in phase.iterdir() if p.is_dir() and re.fullmatch(r'\d\d-[A-Z]', p.name)):
    if not (run / 'rows.json').exists():
        continue
    rows = json.loads((run / 'rows.json').read_text())['rows']
    res = json.loads((run / 'result.json').read_text())
    a = arms.setdefault(run.name[-1], {'runs': [], 'rows': {'prime': [], 'measure': []}, 'tok_s': [],
                                       'req_s': [], 'free': [], 'counts': []})
    a['runs'].append(run.name)
    for r in rows:
        a['rows'].setdefault(r['phase'], []).append(r)
    m = res['measure']
    a['tok_s'].append(m['agg_tok_s'])
    a['req_s'].append(m['ok'] / m['wall_s'])
    a['free'].append(free_floor(run))
    c = counts(run)
    c['hits'] = res['prime'].get('hits', 0) + m.get('hits', 0)
    c['cached_tokens'] = res['prime'].get('cached_tokens', 0) + m.get('cached_tokens', 0)
    a['counts'].append(c)

out = []
for arm, a in sorted(arms.items()):
    row = {'arm': arm, 'runs': a['runs'], 'N': len(a['runs'])}
    for ph in ('prime', 'measure'):
        rs = a['rows'].get(ph, [])
        ok = [r for r in rs if r.get('status') == 200 and r.get('ttft_s') is not None]
        ttft = [r['ttft_s'] for r in ok]
        e2e = [r['wall_s'] for r in ok]
        tpot = [(r['wall_s'] - r['ttft_s']) / (r['completion_tokens'] - 1)
                for r in ok if (r.get('completion_tokens') or 0) > 1]
        row[ph] = {
            'requests': len(rs), 'failed': sum(1 for r in rs if r.get('status') != 200),
            'ttft': [pct(ttft, q) for q in (0.5, 0.95, 0.99)],
            'e2e': [pct(e2e, q) for q in (0.5, 0.95, 0.99)],
            'tpot_ms': [None if x is None else 1000 * x for x in (pct(tpot, q) for q in (0.5, 0.95, 0.99))],
        }
    row['tok_s_median'] = statistics.median(a['tok_s'])
    row['req_s_median'] = statistics.median(a['req_s'])
    row['free_floor_lowest_mib'] = min(f for f in a['free'] if f is not None)
    row['counts'] = {k: sum(c[k] for c in a['counts']) for k in a['counts'][0]}
    out.append(row)

if '--md' in sys.argv:
    f = lambda v: '-' if v is None else f'{v:.1f}'
    print('| arm | N | pass | failed | TTFT p50 / p95 / p99 s | E2E p50 / p95 / p99 s | TPOT p50 / p95 / p99 ms |')
    print('|---|---|---|---|---|---|---|')
    for r in out:
        for ph in ('prime', 'measure'):
            p = r[ph]
            print(f"| {r['arm']} | {r['N']} | {ph} | {p['failed']} of {p['requests']} | "
                  f"{' / '.join(map(f, p['ttft']))} | {' / '.join(map(f, p['e2e']))} | {' / '.join(map(f, p['tpot_ms']))} |")
    print()
    print('| arm | measured tok/s (median run) | requests/s (median run) | free floor, lowest run MiB | OOM lines | VRAM defers | reserve yields / prime-call yields | prefix hits (cached tokens), both passes | deadline-order ticks | batch OOM retries |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in out:
        c = r['counts']
        print(f"| {r['arm']} | {r['tok_s_median']:.2f} | {r['req_s_median']:.3f} | {r['free_floor_lowest_mib']} | "
              f"{c['oom']} | {c['defer']} | {c['yield']} / {c['prime_yield']} | {c['hits']} ({c['cached_tokens']:,}) | {c['edf']} | {c['batch_retry']} |")
else:
    print(json.dumps(out, indent=1))
