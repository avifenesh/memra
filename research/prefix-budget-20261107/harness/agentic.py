"""Agentic cell (ab_agent.py run shape) with per-request TTFT and cached-token rows.

Same bodies as ~/.cache/model-scout/harness/ab_agent.py: every turn of the workload is primed
once with max_tokens=1 at the cell's concurrency, then the measured pass sends the same bodies
again. The measured pass is where the prefix cache pays: a resident entry turns its prefill
into a restore. Rows record ttft, cached tokens, prompt tokens and generation rate per request,
and the prefix-cache counters are diffed around each pass.

  python3 agentic.py WORKLOAD.json OUT.json --users 1 [--temperature 1.0] [--max-tokens 768]
"""
import argparse, json, re, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, '/home/avifenesh/.cache/spark-cast-qualification-20261024-2250')
sys.path.insert(0, '/home/avifenesh/.cache/model-scout/harness')
import client  # noqa: E402

BASE = client.BASE


def metrics():
    raw = urllib.request.urlopen(BASE + '/metrics', timeout=15).read().decode()
    out = {}
    for m in re.finditer(r"^([A-Za-z_:]*(?:prefix_cache|evict|reclaim)[A-Za-z_:]*)(?:\{[^}]*\})? ([0-9.e+-]+)$", raw, re.M):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


ap = argparse.ArgumentParser()
ap.add_argument('workload')
ap.add_argument('out')
ap.add_argument('--users', type=int, default=1)
ap.add_argument('--think', type=int, default=1)
ap.add_argument('--temperature', type=float, default=1.0)
ap.add_argument('--max-tokens', type=int, default=768)
ap.add_argument('--seed', type=int, default=1234)
a = ap.parse_args()

work = json.loads(Path(a.workload).read_text())
kw = {'enable_thinking': True} if a.think else {'enable_thinking': False}
bodies = [{'model': 'gate', 'messages': w['messages'], 'tools': w['tools'], 'max_tokens': a.max_tokens,
           'temperature': a.temperature, 'chat_template_kwargs': kw, 'seed': a.seed + k}
          for k, w in enumerate(work)]


def prime(b):
    t0 = time.time()
    try:
        r = client.stream(dict(b, max_tokens=1))
        return {'ttft_s': r['ttft_s'], 'cached': (r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens'),
                'prompt_tokens': r['usage'].get('prompt_tokens'), 'wall_s': r['wall_s']}
    except AssertionError:
        return {'ttft_s': None, 'cached': None, 'prompt_tokens': None, 'wall_s': time.time() - t0}


m0 = metrics()
t0 = time.time()
with ThreadPoolExecutor(a.users) as pool:
    primes = list(pool.map(prime, bodies))
prime_wall = time.time() - t0
m1 = metrics()
t1 = time.time()
with ThreadPoolExecutor(a.users) as pool:
    rs = list(pool.map(client.stream, bodies))
wall = time.time() - t1
m2 = metrics()

rows = []
for w, p, r in zip(work, primes, rs):
    u = r['usage']
    rows.append({'session': w['session'], 'turn': w['turn'], 'prime': p,
                 'ttft_s': r['ttft_s'], 'wall_s': r['wall_s'], 'prompt_tokens': u.get('prompt_tokens'),
                 'cached': (u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                 'completion_tokens': u.get('completion_tokens'), 'generation_tps': r['generation_tps'],
                 'finish': r.get('finish'), 'content': r['message'].get('content'),
                 'reasoning': r.get('reasoning', ''), 'tool_calls': r['message'].get('tool_calls')})
toks = sum(x['completion_tokens'] for x in rows)
ttfts = sorted(x['ttft_s'] for x in rows)
pct = lambda v, q: v[min(len(v) - 1, int(q * len(v)))]
prompt_total = sum(x['prompt_tokens'] or 0 for x in rows)
cached_total = sum(x['cached'] or 0 for x in rows)
cell = {
    'users': a.users, 'temperature': a.temperature, 'max_tokens': a.max_tokens, 'turns': len(rows),
    'prime_wall_s': round(prime_wall, 1), 'wall_s': round(wall, 1), 'tokens': toks,
    'aggregate_gen_tps': round(toks / (max(r['finished_unix'] for r in rs) - min(r['started_unix'] + r['ttft_s'] for r in rs)), 2),
    'per_user_gen_tps_median': round(statistics.median(x['generation_tps'] for x in rows if x['generation_tps']), 2),
    'measured_hits': sum(1 for x in rows if (x['cached'] or 0) > 0),
    'measured_cached_tokens': cached_total, 'measured_prompt_tokens': prompt_total,
    'measured_cached_frac': round(cached_total / prompt_total, 4) if prompt_total else None,
    'ttft_ms_p50': round(1000 * pct(ttfts, 0.5), 1), 'ttft_ms_p95': round(1000 * pct(ttfts, 0.95), 1),
    'ttft_ms_p99': round(1000 * pct(ttfts, 0.99), 1), 'ttft_ms_mean': round(1000 * sum(ttfts) / len(ttfts), 1),
    'prime_metrics_delta': {k: m1.get(k, 0) - m0.get(k, 0) for k in m1},
    'measured_metrics_delta': {k: m2.get(k, 0) - m1.get(k, 0) for k in m2},
}
json.dump({'cell': cell, 'rows': rows}, open(a.out, 'w'), indent=1)
print(json.dumps(cell), flush=True)
