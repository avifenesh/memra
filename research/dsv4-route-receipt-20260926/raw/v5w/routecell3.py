#!/usr/bin/env python3
"""The DSv4 route's two-card receipt, third pass (memra #500) against a running server.

#500: /health across a prime longer than the stall bound (120 s), the request carrying a 600 s
      timeout_ms so its own deadline does not cut the prime (v5u's 90 s default did): every 1 s poll logged compactly
      (HTTP status, process phase, scheduler phase, the route's phase, prime rows, progress age).
#503: the memory door under contention. One 900k-capacity session holds a card's worth of state
      while others ask for the same: first held by a long generation, then by a long prime. The
      server's [admit-mem] lines are the verdicts; this file records what each client saw.
"""
import argparse, json, threading, time, urllib.request, urllib.error
ap = argparse.ArgumentParser(); ap.add_argument('--port', required=True); ap.add_argument('--key', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); base = f'http://127.0.0.1:{a.port}'
out = open(a.out, 'w'); T0 = time.time()
def log(**kw): kw['t'] = round(time.time() - T0, 2); out.write(json.dumps(kw) + '\n'); out.flush(); print(json.dumps(kw)[:300], flush=True)
def get(path):
    try:
        with urllib.request.urlopen(urllib.request.Request(base + path, headers={'authorization': f'Bearer {a.key}'}), timeout=10) as r:
            return r.status, r.read().decode(errors='replace')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors='replace')
    except Exception as e:
        return -1, str(e)
def compact(st, body):
    try:
        w = json.loads(body)['worker']; r = (w.get('routes') or [{}])[0]
        return dict(http=st, phase=w['phase'], sched=w['scheduler_phase'], beat_ms=w['beat_age_ms'],
                    top_progress_ms=w['forward_progress_age_ms'], route_phase=r.get('phase'),
                    prime_rows=r.get('prime_rows'), route_progress_ms=r.get('progress_age_ms'),
                    rounds=r.get('rounds'), waiting=r.get('waiting'), stall_ms=w['stall_threshold_ms'])
    except Exception:
        return dict(http=st, raw=body[:200])
def body_of(content, max_tokens, stream):
    return json.dumps({'model': 'dsv4f', 'max_tokens': max_tokens, 'stream': stream, 'temperature': 0.0,
                       'messages': [{'role': 'user', 'content': content}]}).encode()
def post(content, max_tokens, timeout=1200, deadline_ms=None):
    body = json.loads(body_of(content, max_tokens, False))
    if deadline_ms: body['timeout_ms'] = deadline_ms
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'content-type': 'application/json', 'authorization': f'Bearer {a.key}'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors='replace')[:300], time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors='replace')[:300], time.time() - t0
    except Exception as e:
        return -1, str(e)[:300], time.time() - t0
def hold(name, content, max_tokens, hold_s):
    """Streaming session: a reader thread records the first SSE line, the first content delta and
    any error; after hold_s the socket is shut down (the server sees the client leave)."""
    import socket
    req = urllib.request.Request(base + '/v1/chat/completions', data=body_of(content, max_tokens, True), headers={'content-type': 'application/json', 'authorization': f'Bearer {a.key}'})
    t0 = time.time(); log(cell='503', who=name, what='start', max_tokens=max_tokens)
    try:
        r = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        log(cell='503', who=name, what='http error', status=e.code, body=e.read().decode(errors='replace')[:300],
            headers={k: v for k, v in e.headers.items() if k.lower().startswith(('retry-after', 'x-ratelimit'))}, s=round(time.time() - t0, 2))
        return
    except Exception as e:
        log(cell='503', who=name, what='exception', err=str(e)[:200], s=round(time.time() - t0, 2)); return
    hdr = {k: v for k, v in r.headers.items() if k.lower().startswith(('retry-after', 'x-ratelimit'))}
    log(cell='503', who=name, what='headers', status=r.status, headers=hdr, s=round(time.time() - t0, 2))
    seen = {'lines': 0, 'first_line_s': None, 'first_content_s': None, 'error': None, 'done': False}
    def reader():
        try:
            for raw in r:
                ln = raw.decode(errors='replace').strip()
                if not ln: continue
                seen['lines'] += 1
                if seen['first_line_s'] is None: seen['first_line_s'] = round(time.time() - t0, 2)
                if '"error"' in ln and seen['error'] is None:
                    seen['error'] = ln[:300]; log(cell='503', who=name, what='sse error', s=round(time.time() - t0, 2), line=ln[:300])
                if seen['first_content_s'] is None and '"content":"' in ln.replace(' ', '') and '"content":""' not in ln.replace(' ', ''):
                    seen['first_content_s'] = round(time.time() - t0, 2)
                if ln == 'data: [DONE]': seen['done'] = True; break
        except Exception as e:
            seen['reader_exit'] = str(e)[:100]
    rt = threading.Thread(target=reader, daemon=True); rt.start()
    rt.join(hold_s)
    try:
        r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
    except Exception as e:
        seen['shutdown'] = str(e)[:100]
    rt.join(5)
    log(cell='503', who=name, what='client closes', held_s=round(time.time() - t0, 2), **seen)
# ---- #500: a prime past the stall bound
st, body = get('/health'); log(cell='500', what='health idle', **compact(st, body))
long_prompt = ' '.join(f'word{i % 97}' for i in range(24000)) + '\nSummarize.'
res = {}
th = threading.Thread(target=lambda: res.setdefault('r', post(long_prompt, 16, deadline_ms=600000)))
th.start()
while th.is_alive():
    st, body = get('/health'); log(cell='500', what='poll', **compact(st, body)); time.sleep(1.0)
th.join()
log(cell='500', what='request done', status=res['r'][0], request_s=round(res['r'][2], 1), body=res['r'][1][:160])
st, body = get('/health'); log(cell='500', what='health after', **compact(st, body))
log(cell='done')
