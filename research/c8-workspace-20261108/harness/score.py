"""Scored-phase summary: one row per run and the per-arm median (N stated).

Agentic runs (agentic2.py rows): prime and measured passes separately. Mixed runs (mixed.py): the lane's
summary fields. Run directories are named NN-A / NN-B (arm letter last).

  python3 score.py PHASE_DIR [--md]
"""
import argparse
import datetime
import json
import os
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('phase')
ap.add_argument('--md', action='store_true')
a = ap.parse_args()


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def tel(d, t0, t1):
    free, util = [], []
    for line in open(d / 'gpu-tel.csv'):
        p = [x.strip() for x in line.split(',')]
        if len(p) < 7 or not p[6].endswith('%'):
            continue
        try:
            ts = datetime.datetime.strptime(p[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
        except ValueError:
            continue
        if t0 <= ts <= t1:
            free.append(int(p[2].split()[0]))
            util.append(int(p[6].split()[0]))
    return (min(free) if free else None), (round(statistics.mean(util), 1) if util else None)


def agentic(d, rows, phase):
    rs = [r for r in rows if r['phase'] == phase]
    if not rs:
        return None
    t0 = min(r['t_start'] for r in rs)
    t1 = max(r['t_start'] + r['wall_s'] for r in rs)
    ok = [r for r in rs if r['status'] == 200 and not r['error'] and r['ttft_s'] is not None]
    tt = [r['ttft_s'] * 1000 for r in ok]
    toks = sum(r['completion_tokens'] or 0 for r in ok)
    gen = [r['gen_tps'] for r in ok if r.get('gen_tps')]
    free, util = tel(d, t0, t1)
    wait = [(r['ttft_s'] if r in ok else r['wall_s']) * 1000 for r in rs]
    return {
        'ok': len(ok), 'n': len(rs), 'e408': sum(1 for r in rs if r['status'] == 408),
        'e_stream': sum(1 for r in rs if r['status'] == 200 and r['error']),
        'e_other': sum(1 for r in rs if r['status'] not in (200, 408)),
        'wall_s': round(t1 - t0, 1), 'ttft_p50': round(pct(tt, .5)) if tt else None,
        'ttft_p99': round(pct(tt, .99)) if tt else None, 'ttft_max': round(max(tt)) if tt else None,
        'over90': sum(1 for x in tt if x > 90000), 'tok_s': round(toks / (t1 - t0), 2),
        'per_user_tps': round(statistics.median(gen), 2) if gen else None,
        'hits': sum(1 for r in ok if (r['cached'] or 0) > 0), 'cached': sum(r['cached'] or 0 for r in ok),
        'free_min': free, 'util': util, 'tokens': toks,
        # what every user waited for its first byte, an error counted at the time it arrived
        'wait_p50_all': round(pct(wait, .5)), 'wait_p99_all': round(pct(wait, .99)),
    }


out = []
for d in sorted(Path(a.phase).iterdir()):
    if not (d / 'rows.json').exists():
        continue
    arm = d.name.split('-')[-1] if d.name[:2].isdigit() else d.name  # off-A etc. are their own arm
    j = json.load(open(d / 'rows.json'))
    log = (d / 'serve.log').read_text(errors='replace')
    counters = {'oom': log.count('CUDA_ERROR_OUT_OF_MEMORY'), 'hol': log.count('[prefill-order] earliest'),
                'headroom': log.count('[prefix-cache] headroom:') + log.count('[prefix-cache] reserve:'), 'defer': log.count('VRAM defer')}
    if isinstance(j, dict) and j.get('rows') and 'phase' in j['rows'][0]:
        rec = {'run': d.name, 'arm': arm, 'prime': agentic(d, j['rows'], 'prime'),
               'measure': agentic(d, j['rows'], 'measure'), **counters}
    else:
        s = json.load(open(d / 'result.json'))
        s = s.get('summary', s)
        rows = j['rows'] if isinstance(j, dict) else j
        t0 = min(r['t_start'] for r in rows)
        t1 = max(r['t_start'] + r['wall_s'] for r in rows)
        free, util = tel(d, t0, t1)
        rec = {'run': d.name, 'arm': arm, 'mixed': {
            'tok_s': s['batch_agg_tok_s'], 'ttft_p50': round(s['batch_ttft_ms_p50']),
            'rows': s['batch_rows'], 'repeat_hits': s['repeat_hits'], 'repeat_cached': s['repeat_cached'],
            'repeat_ttft_med': round(statistics.median([x for x in s['repeat_ttft_ms'] if x])),
            'free_min': free, 'util': util}, **counters}
    out.append(rec)

for r in out:
    print(json.dumps(r))
# per-arm medians
arms = sorted({r['arm'] for r in out})
for arm in arms:
    rs = [r for r in out if r['arm'] == arm]
    med = {'arm': arm, 'N': len(rs)}
    for part in ('prime', 'measure', 'mixed'):
        vals = [r[part] for r in rs if r.get(part)]
        if not vals:
            continue
        m = {}
        for k in vals[0]:
            xs = [v[k] for v in vals if isinstance(v.get(k), (int, float))]
            if xs:
                m[k] = statistics.median(xs)
        med[part] = m
    for k in ('oom', 'hol', 'headroom', 'defer'):
        med[k] = sum(r[k] for r in rs)
    print('MEDIAN (counters summed)', json.dumps(med))
