"""Summarize one arm directory: client rows, [ttft] phase lines, GPU telemetry, admission counters.

  python3 analyze.py ARM_DIR [ARM_DIR ...]
"""
import json
import os
import re
import statistics
import sys


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def arm(d):
    out = {'arm': os.path.basename(d)}
    rows = json.load(open(os.path.join(d, 'rows.json')))
    rows = rows['rows'] if isinstance(rows, dict) else rows
    t0 = min(r['t_start'] for r in rows)
    t1 = max(r['t_start'] + (r.get('wall_s') or 0) for r in rows)
    out['window_s'] = round(t1 - t0, 1)
    st = {}
    for r in rows:
        k = str(r.get('status', 200)) + ('+err' if r.get('error') else '')
        st[k] = st.get(k, 0) + 1
    out['status'] = st
    ok = [r for r in rows if r.get('ttft_s') is not None and not r.get('error')]
    tt = [r['ttft_s'] for r in ok]
    out['ttft_s_p50'] = round(pct(tt, .5), 2) if tt else None
    out['ttft_s_p99'] = round(pct(tt, .99), 2) if tt else None
    out['ttft_s_max'] = round(max(tt), 2) if tt else None
    out['over_90s'] = sum(1 for x in tt if x > 90)
    ptok = sum(r.get('prompt_tokens') or 0 for r in ok)
    ctok = sum(r.get('cached') or 0 for r in ok)
    out['prompt_tokens'] = ptok
    out['cached_tokens'] = ctok
    out['computed_tok_per_s'] = round((ptok - ctok) / (t1 - t0), 1)
    # server phases
    ph = {k: [] for k in ('queue_wait_ms', 'prime_wait_ms', 'prime_ms', 'total_ms')}
    lines = open(os.path.join(d, 'serve.log'), errors='replace').read().splitlines()
    for l in lines:
        if '[ttft]' not in l:
            continue
        for k in ph:
            m = re.search(r'\b' + k + r'=([0-9.]+)', l)
            if m:
                ph[k].append(float(m.group(1)))
    for k, v in ph.items():
        if v:
            out[k + '_p50'] = round(pct(v, .5))
            out[k + '_max'] = round(max(v))
    out['ttft_lines'] = len(ph['total_ms'])
    for tag in ('VRAM defer', 'reclaim-on-defer:', 'admit-trim', 'prefix-cache] insert', 'prefix-cache] evict',
                'prefix-cache] hit', 'prime-batch]', 'deadline', '[shed]', 'abort]'):
        out['n_' + tag.strip('[]: ').replace(' ', '_').replace(']', '')] = sum(1 for l in lines if tag in l)
    # GPU telemetry inside the client window
    util, free = [], []
    for l in open(os.path.join(d, 'gpu-tel.csv')):
        p = [x.strip() for x in l.split(',')]
        if len(p) < 7 or not p[6].endswith('%'):
            continue
        try:
            import datetime
            ts = datetime.datetime.strptime(p[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
        except ValueError:
            continue
        if t0 <= ts <= t1:
            util.append(int(p[6].split()[0]))
            free.append(int(p[2].split()[0]))
    if util:
        out['gpu_util_mean'] = round(statistics.mean(util), 1)
        out['gpu_util_frac_lt50'] = round(sum(1 for u in util if u < 50) / len(util), 3)
        out['device_free_min_mib'] = min(free)
    return out


for d in sys.argv[1:]:
    try:
        print(json.dumps(arm(d)))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({'arm': d, 'error': repr(e)}))
