"""Phase T split: per arm and pass, first-token wait by prompt size (an error counted when it
arrived), and the `[ttft]` trace's split of each served first token into worker queue wait,
prime_wait and prime, medians over the arm's runs pooled.

  python3 split_t.py PHASE_DIR
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
BUCKETS = ((0, 4000, 'under 4k'), (4000, 8000, '4k-8k'), (8000, 12000, '8k-12k'), (12000, 10**9, '12k+'))


def bucket(n):
    for lo, hi, name in BUCKETS:
        if lo <= n < hi:
            return name


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def trace(run, phase):
    """Served first tokens of one pass: the measured pass starts at its first row's send."""
    rows = json.load(open(run / 'rows.json'))['rows']
    starts = [r['t_start'] for r in rows if r['phase'] == 'measure']
    t_meas = min(starts) if starts else float('inf')
    out = []
    for line in open(run / 'serve.log', errors='replace'):
        if '[ttft]' not in line or 'outcome=first_sse_byte' not in line:
            continue
        t = float(line.split(' ', 1)[0])
        if (t >= t_meas) != (phase == 'measure'):
            continue
        g = {k: float(v) for k, v in re.findall(r'(\w+_ms)=([0-9.]+)', line)}
        out.append((int(re.search(r'prompt_tokens=(\d+)', line).group(1)), g))
    return out


runs = sorted(p for p in d.iterdir() if (p / 'rows.json').exists())
size = {}
for run in runs:
    for r in json.load(open(run / 'rows.json'))['rows']:
        if r['status'] == 200 and r.get('prompt_tokens'):
            size[(r['phase'], r['k'])] = r['prompt_tokens']
arms = defaultdict(list)
for run in runs:
    arms[run.name.split('-', 1)[1]].append(run)
for arm, rs in arms.items():
    for phase in ('measure', 'prime'):
        waits, fails, by = [], 0, defaultdict(list)
        for run in rs:
            for r in json.load(open(run / 'rows.json'))['rows']:
                if r['phase'] != phase:
                    continue
                ok = r['status'] == 200 and not r['error'] and r['ttft_s'] is not None
                w = r['ttft_s'] if ok else r['wall_s']
                fails += not ok
                waits.append(w)
                by[bucket(size.get((phase, r['k']), 0))].append(w)
        if not waits:
            continue
        line = f"{arm:5} {phase:7} N={len(rs)} failed {fails:2}  wait p50 {pct(waits, .5):5.1f} p99 {pct(waits, .99):5.1f} |"
        for _, _, name in BUCKETS:
            if by[name]:
                line += f" {name} {statistics.median(by[name]):5.1f}"
        print(line)
        tr = defaultdict(lambda: defaultdict(list))
        for run in rs:
            for n, g in trace(run, phase):
                for k in ('queue_wait_ms', 'prime_wait_ms', 'prime_ms', 'decode_wait_ms'):
                    tr[bucket(n)][k].append(g.get(k, 0) / 1000)
        for _, _, name in BUCKETS:
            if tr[name]:
                m = {k: statistics.median(v) for k, v in tr[name].items()}
                print(f"      trace {name:8} n={len(tr[name]['prime_ms']):3} queue {m['queue_wait_ms']:5.1f} "
                      f"prime_wait {m['prime_wait_ms']:5.1f} prime {m['prime_ms']:5.1f} decode_wait {m['decode_wait_ms']:4.1f}")
