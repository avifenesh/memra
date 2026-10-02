#!/usr/bin/env python3
"""Live hybrid metrics checks after the fresh-server cache-meter arm.

A normal max-token finish is a success. The cancellation arm closes its socket
while generation is still active. Premature producer termination is separately
covered by the HTTP fixture tests, not fabricated by calling max_tokens truncation.
"""
import argparse
import concurrent.futures
import threading
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

from prometheus_metrics import histogram, parse_samples, scalar

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--model', required=True)
p.add_argument('--api-key-file', type=Path, required=True)
p.add_argument('--metrics-token-file', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--promtool', required=True)
p.add_argument('--spec', action='store_true')
p.add_argument('--queue', action='store_true', help='fresh server with MEMRA_MAX_SESSIONS=1')
p.add_argument('--identity-only', action='store_true', help='capture the same clean request against an immutable baseline')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
api = a.api_key_file.read_text().strip()
operator = a.metrics_token_file.read_text().strip()
checks = []


def check(name, ok, detail=None):
    checks.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)


def get(path, token=None, accept=None):
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    if accept:
        headers['Accept'] = accept
    req = urllib.request.Request(a.base + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def snap(label):
    status, headers, raw = get('/metrics', operator, 'text/plain')
    check(label + ' metrics HTTP 200', status == 200)
    text = raw.decode()
    (a.out / (label + '.prom')).write_text(text)
    r = subprocess.run([a.promtool, 'check', 'metrics'], input=text, text=True,
                       capture_output=True, timeout=30)
    (a.out / (label + '.promtool.log')).write_text(r.stdout + r.stderr)
    check(label + ' promtool', r.returncode == 0, r.stdout + r.stderr)
    return parse_samples(text)


def stream(label, max_tokens=16, cancel=False, model=None, first_token=None, long_output=False):
    prompt = ('Write at least two thousand words explaining how a database works, with detailed examples.'
              if cancel or long_output else 'Count from one to twenty in words, separated by commas.')
    body = {'model': model or a.model, 'prompt': prompt,
            'max_tokens': max_tokens, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}, 'cache_salt': 'metric-' + label}
    req = urllib.request.Request(a.base + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + api, 'Content-Type': 'application/json'})
    started = time.monotonic()
    row = {'request': body, 'events': [], 'done': False, 'cancelled_by_client': False}
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            row['status'] = r.status
            for line in r:
                if not line.startswith(b'data:'):
                    continue
                value = line[5:].strip()
                if value == b'[DONE]':
                    row['done'] = True
                    break
                event = json.loads(value)
                row['events'].append({'elapsed_seconds': time.monotonic() - started, 'event': event})
                if event.get('usage'):
                    row['usage'] = event['usage']
                for choice in event.get('choices') or []:
                    if choice.get('finish_reason'):
                        row['finish_reason'] = choice['finish_reason']
                    text = choice.get('text') or (choice.get('delta') or {}).get('content') or ''
                    if first_token is not None and text:
                        first_token.set()
                    if cancel and text:
                        # Force a transport close now, before a large output can finish.
                        r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                        row['cancelled_by_client'] = True
                        (a.out / (label + '.json')).write_text(json.dumps(row, indent=2) + '\n')
                        return row
    except urllib.error.HTTPError as e:
        row['status'] = e.code
        row['error'] = e.read().decode()
    row['elapsed_seconds'] = time.monotonic() - started
    (a.out / (label + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    return row


def count(samples, metric, lane=None):
    labels = {'model': a.model}
    if lane:
        labels['lane'] = lane
    try:
        return histogram(samples, 'memra_hybrid_' + metric + '_seconds', **labels)['count']
    except KeyError:
        return 0


def shared(samples, name):
    try:
        return histogram(samples, 'memra_' + name, model=a.model, route='hybrid', lane='interactive')
    except KeyError:
        return {'count': 0, 'sum': 0}


def outcome(samples, name, model=None, route='hybrid', **labels):
    try:
        return scalar(samples, name, model=model or a.model, route=route, lane='interactive', **labels)
    except KeyError:
        return 0


if a.identity_only:
    row = stream('clean')
    check('baseline clean stream has terminal success', row.get('status') == 200 and row['done'] and row.get('finish_reason') in {'length', 'stop'})
    check('baseline clean stream generated the requested tokens', row.get('usage', {}).get('completion_tokens') == 16)
    if a.spec:
        check('baseline MTP arm engaged', (row.get('usage', {}).get('spec') or {}).get('drafted', 0) > 0)
    (a.out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    raise SystemExit(0 if all(c['ok'] for c in checks) else 1)

if a.queue:
    try:
        before = snap('before-queue')
        emitted = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            lead = pool.submit(stream, 'queue-leader', max_tokens=256, first_token=emitted, long_output=True)
            if not emitted.wait(120):
                raise RuntimeError('queue leader did not emit a first token')
            follower = stream('queue-follower', max_tokens=16)
            leader = lead.result(timeout=120)
        for label, row in [('leader', leader), ('follower', follower)]:
            check('queued ' + label + ' completes', row['done'] and row.get('finish_reason') in {'length', 'stop'})
        after = snap('after-queue')
        status, _, raw = get('/metrics', operator)
        metrics = json.loads(raw)
        (a.out / 'queue-metrics.json').write_bytes(raw)
        check('real queue pressure engaged session cap', metrics.get('admission_session_defers', 0) > 0, metrics.get('admission_session_defers'))
        for metric in ('queue_wait', 'ttft', 'e2e'):
            check(metric + ' counts both queued-cell requests', count(after, metric) - count(before, metric) == 2)
        queue = histogram(after, 'memra_hybrid_queue_wait_seconds', model=a.model)
        check('queue wait histogram records positive wait', queue['sum'] > 0, queue['sum'])
        expected = sum(max(row.get('usage', {}).get('completion_tokens', 0) - 1, 0) for row in (leader, follower))
        check('queued-cell emitted gaps account for both streams', count(after, 'token_gap', 'interactive') - count(before, 'token_gap', 'interactive') == expected)
    finally:
        (a.out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        print(json.dumps({'passed': sum(c['ok'] for c in checks), 'failed': sum(not c['ok'] for c in checks)}), flush=True)
    raise SystemExit(0 if checks and all(c['ok'] for c in checks) else 1)

try:
    for token, expected, label in [(None, 401, 'missing'), ('invalid-metrics-fixture', 401, 'wrong'), (api, 403, 'completion')]:
        for accept in ['text/plain', 'application/json']:
            status, _, raw = get('/metrics', token, accept)
            check(f'{label} credential {accept}: HTTP {expected}', status == expected, raw.decode())
    status, headers, raw = get('/metrics', operator)
    before_json = json.loads(raw)
    (a.out / 'default-metrics.json').write_bytes(raw)
    check('default metrics remains JSON', status == 200 and 'json' in headers.get('Content-Type', headers.get('content-type', '')))
    negotiated_status, negotiated_headers, negotiated = get('/metrics', operator, 'application/json, text/plain, */*')
    (a.out / 'before-negotiated-metrics.json').write_bytes(negotiated)
    negotiated_json = json.loads(negotiated)
    # These are separate live observations, so startup/worker publication may change
    # values between them. Exact JSON-builder parity is checked on immutable source.
    core = {'admitted', 'completed', 'tokens_out', 'prompt_tokens_in', 'cached_tokens_in', 'computed_tokens_in'}
    check('JSON preference retains the established JSON contract',
          negotiated_status == 200 and 'json' in negotiated_headers.get('Content-Type', negotiated_headers.get('content-type', ''))
          and core <= set(before_json) and core <= set(negotiated_json)
          and not any(key.startswith('memra_') for key in negotiated_json))
    before = snap('before-clean')
    clean = stream('clean')
    check('clean stream has terminal success and DONE', clean.get('status') == 200 and clean.get('finish_reason') in {'length', 'stop'} and clean['done'])
    n = clean.get('usage', {}).get('completion_tokens', 0)
    check('clean stream generated tokens', n > 1, n)
    if a.spec:
        spec = clean.get('usage', {}).get('spec') or {}
        check('MTP arm engaged', spec.get('drafted', 0) > 0, spec)
    after = snap('after-clean')
    for metric in ('queue_wait', 'ttft', 'e2e'):
        check(metric + ' records clean session exactly once', count(after, metric) - count(before, metric) == 1)
    check('clean emitted-gap count is output tokens minus one', count(after, 'token_gap', 'interactive') - count(before, 'token_gap', 'interactive') == n - 1)
    for metric in ('queue_wait_seconds', 'ttft_seconds', 'e2e_seconds', 'prefill_tokens', 'completion_tokens'):
        check('shared ' + metric + ' records clean request once', shared(after, metric)['count'] - shared(before, metric)['count'] == 1)
    check('shared token gaps count actual emitted events', shared(after, 'tpot_seconds')['count'] - shared(before, 'tpot_seconds')['count'] == n - 1)
    check('completion histogram sums exact tokens', shared(after, 'completion_tokens')['sum'] - shared(before, 'completion_tokens')['sum'] == n)
    check('prefill histogram sums worker prompt usage', shared(after, 'prefill_tokens')['sum'] - shared(before, 'prefill_tokens')['sum'] == clean['usage']['prompt_tokens'])
    check('clean response does not count an error', outcome(after, 'memra_response_errors_total') - outcome(before, 'memra_response_errors_total') == 0)
    check('HTTP clean outcome counted once', outcome(after, 'memra_requests_total', code='200') - outcome(before, 'memra_requests_total', code='200') == 1)
    check('max_tokens finish is a complete stream', outcome(after, 'memra_streams_incomplete_total') == outcome(before, 'memra_streams_incomplete_total'))
    before_bad = after
    for i in range(3):
        bad = stream('bad-model-' + str(i), model='unloaded-label-' + str(i))
        check('unknown model rejected ' + str(i), bad.get('status') in {400, 404}, bad.get('status'))
    after_bad = snap('after-rejected')
    added = set(after_bad) - set(before_bad)
    check('rejected model names collapse to one fixed unknown row',
          all(dict(labels).get('model') == 'unknown' and dict(labels).get('route') == 'unresolved' for _, labels in added))
    check('all unknown-model refusals counted',
          outcome(after_bad, 'memra_requests_refused_total', model='unknown', route='unresolved') - outcome(before_bad, 'memra_requests_refused_total', model='unknown', route='unresolved') == 3)
    for metric in ('queue_wait', 'ttft', 'e2e'):
        check(metric + ' unchanged after pre-admission rejection', count(after_bad, metric) == count(before_bad, metric))
    cancelled = stream('cancelled', max_tokens=1024, cancel=True)
    check('client cancelled before terminal frame', cancelled['cancelled_by_client'] and not cancelled.get('finish_reason') and not cancelled['done'])
    # A later completed request is a worker-progress barrier, not a metrics polling loop.
    recovery = stream('recovery', max_tokens=8)
    check('request after cancellation completes', recovery['done'] and recovery.get('finish_reason') in {'length', 'stop'})
    final = snap('after-cancel-and-recovery')
    check('cancelled session excluded from success E2E', count(final, 'e2e') - count(after_bad, 'e2e') == 1)
    check('cancel and recovery each admitted', count(final, 'queue_wait') - count(after_bad, 'queue_wait') == 2)
    check('model labels remain bounded', all(dict(labels).get('model', a.model) in {a.model, 'unknown'} for _, labels in final))
    check('backend labels remain bounded', all(dict(labels).get('route', 'hybrid') in {'hybrid', 'dsv4', 'unresolved'} for _, labels in final))
    check('lane labels remain bounded', all(dict(labels).get('lane', 'interactive') in {'interactive', 'judge', 'harvest'} for _, labels in final))
    check('HTTP client cancellation counted as 499', outcome(final, 'memra_requests_total', code='499') - outcome(after_bad, 'memra_requests_total', code='499') == 1)
    check('cancelled stream counted incomplete', outcome(final, 'memra_streams_incomplete_total') - outcome(after_bad, 'memra_streams_incomplete_total') == 1)
    check('client cancellation is distinct from producer truncation', outcome(final, 'memra_streams_truncated_total') == outcome(after_bad, 'memra_streams_truncated_total'))
    check('shared success E2E excludes client cancellation', shared(final, 'e2e_seconds')['count'] - shared(after_bad, 'e2e_seconds')['count'] == 1)
    _, _, default_raw = get('/metrics', operator)
    current_default = json.loads(default_raw)
    status, _, raw = get('/metrics', operator, 'application/json, text/plain, */*')
    after_json = json.loads(raw)
    check('JSON negotiation/schema remains compatible', status == 200 and set(after_json) == set(current_default) and set(before_json) <= set(after_json))
    check('JSON has no Prometheus histogram fields', not any(key.startswith('memra_hybrid_') for key in after_json))
    (a.out / 'after-default-metrics.json').write_bytes(default_raw)
    (a.out / 'after-metrics.json').write_bytes(raw)
finally:
    (a.out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps({'passed': sum(c['ok'] for c in checks), 'failed': sum(not c['ok'] for c in checks)}), flush=True)
raise SystemExit(0 if all(c['ok'] for c in checks) else 1)
