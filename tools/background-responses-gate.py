#!/usr/bin/env python3
"""Bounded native background endpoint acceptance under an inherited GPU lock.

Build the background_accounting_gate example first. Run OFF and ON as paired arms.
This observes endpoint behavior and accounting callbacks, not model qualification.
"""
import argparse, hashlib, json, os, pathlib, signal, socket, subprocess, sys, threading, time, urllib.request, urllib.error
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=pathlib.Path, required=True)
parser.add_argument('--binary', type=pathlib.Path, required=True)
parser.add_argument('--model', type=pathlib.Path, required=True)
parser.add_argument('--port', type=int, default=18130)
parser.add_argument('--external-lock', type=int, required=True)
parser.add_argument('--door-off', action='store_true')
args = parser.parse_args()
out = args.out
out.mkdir(parents=True, exist_ok=True)
door = '0' if args.door_off else '1'
lock = pathlib.Path(os.environ['MEMRA_GPU_LOCK'])
assert os.path.samefile(f'/proc/self/fd/{args.external_lock}', lock), 'inherited lock must match MEMRA_GPU_LOCK'
with socket.socket() as probe:
    probe.bind(('127.0.0.1', args.port))
binary = args.binary.resolve()
model = args.model.resolve()

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
manifest = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(), 'binary_sha256': digest(binary), 'model': str(model), 'model_sha256': digest(model), 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True), 'context': 32768, 'temperature': 0, 'cache': 'MEMRA_PREFIX_CACHE_MB=0', 'port': args.port, 'background_door': door}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
env = {k: v for k, v in os.environ.items() if not k.startswith('MEMRA_') or k in {'MEMRA_GPU_LOCK', 'MEMRA_CI_LOCK', 'MEMRA_CI_LOCK_HELD', 'MEMRA_RIG_LOCK_FD'}}
env.update(MEMRA_MODELS='q9=' + str(model), MEMRA_CTX='32768', MEMRA_ADDR=f'127.0.0.1:{args.port}', MEMRA_BACKGROUND_RESPONSES=door, MEMRA_PREFIX_CACHE_MB='0', MEMRA_API_KEYS='acme:' + hashlib.sha256(b'gate-owner').hexdigest() + ',blue:' + hashlib.sha256(b'gate-foreign').hexdigest())
ready = threading.Event()
listening = threading.Event()
settled = {}
receipt_lock = threading.Lock()
unsettled = []
server = subprocess.Popen([str(binary)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, pass_fds=(args.external_lock,))

def reader():
    with (out / 'server.log').open('w') as log, (out / 'receipts.jsonl').open('w') as rows:
        for line in server.stdout:
            at = time.monotonic()
            log.write(line)
            log.flush()
            if '[server] listening on ' in line:
                listening.set()
                ready.set()
            if line.startswith('GATE_RECEIPT '):
                row = json.loads(line.split(' ', 1)[1])
                row['observed_monotonic'] = at
                with receipt_lock:
                    settled.setdefault(row['id'], []).append(row)
                rows.write(json.dumps(row) + '\n')
                rows.flush()
            if line.startswith('GATE_UNSETTLED '):
                unsettled.append(line.strip())
        ready.set()
thread = threading.Thread(target=reader)
thread.start()
base = f'http://127.0.0.1:{args.port}'
http_log = (out / 'http.jsonl').open('w')

def request(method, path, body=None, key='gate-owner'):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            status = r.status
            raw = r.read()
    except urllib.error.HTTPError as r:
        status = r.code
        raw = r.read()
    value = json.loads(raw)
    http_log.write(json.dumps({'at': start, 'method': method, 'path': path, 'key_role': 'owner' if key == 'gate-owner' else 'foreign', 'status': status, 'body': value}) + '\n')
    http_log.flush()
    return (status, value)
try:
    assert ready.wait(150), 'server readiness deadline'
    assert listening.is_set() and server.poll() is None, 'server exited before readiness; inspect server.log'
    short = {'model': 'q9', 'input': 'What is two plus two? Answer with one number.', 'max_output_tokens': 64, 'temperature': 0, 'reasoning': {'effort': 'none'}}
    status, sync = request('POST', '/v1/responses', short)
    assert status == 200 and sync['status'] == 'completed', (status, sync)
    status, short_bg = request('POST', '/v1/responses', dict(short, background=True))
    if door == '0':
        assert status == 400 and short_bg['error']['param'] == 'background', (status, short_bg)
        (out / 'summary.json').write_text(json.dumps({'pass': True, 'door_off_refuses': True, 'synchronous_control': sync}, indent=2))
        print('PASS door OFF refusal and synchronous control', flush=True)
        sys.exit(0)
    assert status == 200 and short_bg['status'] == 'queued', (status, short_bg)
    short_id = short_bg['id']
    for sample in range(100):
        status, short_result = request('GET', '/v1/responses/' + short_id)
        assert status == 200
        if short_result['status'] not in ('queued', 'in_progress'):
            break
        threading.Event().wait(0.1)

    def content(value):
        return [(item.get('type'), [(c.get('type'), c.get('text')) for c in item.get('content', [])]) for item in value['output']]
    assert short_result['status'] == 'completed', short_result
    assert content(sync) == content(short_result), (sync, short_result)
    assert sync['usage'] == short_result['usage'], (sync['usage'], short_result['usage'])
    phrase = 'The quick brown fox jumps over the lazy dog while a quiet river flows past the old stone bridge.'
    payload = {'model': 'q9', 'input': 'Return the requested JSON array exactly. No commentary.', 'background': True, 'max_output_tokens': 24000, 'temperature': 0, 'reasoning': {'effort': 'none'}, 'text': {'format': {'type': 'json_schema', 'name': 'long_result', 'strict': True, 'schema': {'type': 'array', 'items': {'type': 'string', 'const': phrase}, 'minItems': 600, 'maxItems': 600}}}}
    (out / 'request.json').write_text(json.dumps(payload, indent=2))
    start = time.monotonic()
    status, submitted = request('POST', '/v1/responses', payload)
    assert status == 200 and submitted['status'] == 'queued', (status, submitted)
    ident = submitted['id']
    assert time.monotonic() - start < 30
    for method, path in [('GET', '/v1/responses/' + ident), ('POST', '/v1/responses/' + ident + '/cancel')]:
        status, foreign = request(method, path, key='gate-foreign')
        assert status == 404, (status, foreign)
    result = None
    for sample in range(120):
        assert server.poll() is None, 'worker process exited'
        status, value = request('GET', '/v1/responses/' + ident)
        assert status == 200, (status, value)
        if value['status'] not in ('queued', 'in_progress'):
            result = value
            break
        threading.Event().wait(5)
    assert result is not None, 'background completion deadline'
    assert result['status'] == 'completed', result
    text = ''.join((c['text'] for item in result['output'] if item.get('type') == 'message' for c in item['content'] if c.get('type') == 'output_text'))
    decoded = json.loads(text)
    assert decoded == [phrase] * 600, 'stored result differs from accepted schema'
    with receipt_lock:
        rows = settled[ident].copy()
    assert len(rows) == 1 and rows[0]['outcome'] == 'complete', rows
    duration = rows[0]['observed_monotonic'] - start
    assert duration > 90, {'native_duration_s': duration, 'requirement': '>90s'}
    assert rows[0]['completion_tokens'] == result['usage']['output_tokens']
    status, repeat = request('GET', '/v1/responses/' + ident)
    assert repeat == result
    assert request('POST', '/v1/responses/' + ident + '/cancel')[0] == 409
    status, second = request('POST', '/v1/responses', payload)
    assert status == 200
    cancel_id = second['id']
    threading.Event().wait(3)
    status, cancel = request('POST', '/v1/responses/' + cancel_id + '/cancel')
    assert status == 200 and cancel['status'] == 'cancelled', cancel
    assert cancel['usage']['output_tokens'] > 0, cancel
    assert request('GET', '/v1/responses/' + cancel_id)[1] == cancel
    assert request('POST', '/v1/responses/' + cancel_id + '/cancel')[0] == 409
    assert request('GET', '/v1/responses/' + cancel_id, key='gate-foreign')[0] == 404
    with receipt_lock:
        cancel_rows = settled[cancel_id].copy()
    assert len(cancel_rows) == 1 and cancel_rows[0]['outcome'] == 'cancel_partial', cancel_rows
    assert cancel_rows[0]['completion_tokens'] == cancel['usage']['output_tokens']
    assert not unsettled, unsettled
    summary = {'pass': True, 'complete_id': ident, 'complete_native_s': duration, 'complete_tokens': result['usage']['output_tokens'], 'cancel_id': cancel_id, 'cancel_tokens': cancel['usage']['output_tokens'], 'rows_per_job': 1}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
finally:
    http_log.close()
    if server.poll() is None:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=25)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
    thread.join(timeout=5)
