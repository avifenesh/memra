#!/usr/bin/env python3
"""The DSv4 route receipt after the #501 estimate fix: the same 12000-word prime, then the same c16 burst.

#500: /health phase and progress across a long prime (250 ms polls), then idle.
#501: 16 concurrent requests on a 4-lane route: which are served, which shed, with the shed
      answer's status, Retry-After and X-RateLimit headers; /metrics before and after.
#503: a session larger than the route's context gets a 400 naming the largest fitting session;
      four concurrent 900k-capacity sessions exercise the memory door (admit, defer, refuse), with
      the server's [admit-mem] lines as the receipt.
"""
import argparse, json, threading, time, urllib.request, urllib.error
ap = argparse.ArgumentParser(); ap.add_argument('--port', required=True); ap.add_argument('--key', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); base = f'http://127.0.0.1:{a.port}'
out = open(a.out, 'w')
def log(**kw): out.write(json.dumps(kw) + '\n'); out.flush(); print(json.dumps(kw)[:300])
def get(path):
    try:
        with urllib.request.urlopen(urllib.request.Request(base + path, headers={'authorization': f'Bearer {a.key}'}), timeout=10) as r:
            return r.status, r.read().decode(errors='replace'), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors='replace'), dict(e.headers)
    except Exception as e:
        return -1, str(e), {}
def chat(content, max_tokens, stream=False, timeout=900):
    body = json.dumps({'model': 'dsv4f', 'max_tokens': max_tokens, 'stream': stream, 'temperature': 0.0,
                       'messages': [{'role': 'user', 'content': content}]}).encode()
    req = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'content-type': 'application/json', 'authorization': f'Bearer {a.key}'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode(errors='replace')
            return r.status, txt[:400], dict(r.headers), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors='replace')[:400], dict(e.headers), time.time() - t0
    except Exception as e:
        return -1, str(e)[:400], {}, time.time() - t0
# ---- #500
st, body, _ = get('/health'); log(cell='500', what='health idle', status=st, body=body[:300])
long_prompt = ' '.join(f'word{i % 97}' for i in range(12000)) + '\nSummarize.'
res = {}
th = threading.Thread(target=lambda: res.setdefault('r', chat(long_prompt, 16)))
th.start(); polls = []
while th.is_alive():
    st, body, _ = get('/health'); polls.append((round(time.time(), 2), st, body)); time.sleep(0.25)
th.join()
phases = []
for t, st, b in polls:
    try:
        j = json.loads(b); r = [x for x in j.get('routes', []) if 'dsv4' in json.dumps(x)] or j.get('routes') or [j]
        phases.append(json.dumps(r)[:200])
    except Exception:
        phases.append(b[:120])
log(cell='500', what='health during a 12000-word prime', polls=len(polls), request_status=res['r'][0], request_s=round(res['r'][3], 1))
for k in (1, len(polls) // 4, len(polls) // 2, 3 * len(polls) // 4, len(polls) - 1):
    log(cell='500', what='health poll', at=k, status=polls[k][1], body=polls[k][2])
st, body, _ = get('/health'); log(cell='500', what='health after', status=st, body=body[:300])
# ---- #501
st, m0, _ = get('/metrics'); log(cell='501', what='metrics before', body=m0[:600])
results = [None] * 16
def one(i): results[i] = chat(f'Count from one to {i + 5}.', 128)
ths = [threading.Thread(target=one, args=(i,)) for i in range(16)]
[t.start() for t in ths]; [t.join() for t in ths]
for i, r in enumerate(results):
    st, body, hdr, dt = r
    keep = {k: v for k, v in hdr.items() if k.lower().startswith(('retry-after', 'x-ratelimit'))}
    log(cell='501', what='c16 request', i=i, status=st, seconds=round(dt, 2), headers=keep, body=body[:160] if st != 200 else '')
st, m1, _ = get('/metrics'); log(cell='501', what='metrics after', body=m1[:600])
log(cell='done')
