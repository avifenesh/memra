"""Serving-route correctness/chunk gate. Every boot owns the GPU lock and its child PID."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import pathlib
import re
import secrets
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--binary', required=True)
p.add_argument('--sha', required=True)
p.add_argument('--out', required=True)
p.add_argument('--arm', choices=['off', 'on'], required=True)
p.add_argument('--mode', choices=['chain', 'solo', 'pair', 'chunk'], required=True)
p.add_argument('--chunk', type=int, default=1024)
p.add_argument('--long-request')
p.add_argument('--oracle', action='store_true')
a = p.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=False)
source = pathlib.Path(a.source)
lock = open('/tmp/memra-gpu.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
apps_cmd = ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader']
assert not subprocess.check_output(apps_cmd, text=True).strip(), 'GPU occupied'
binary = pathlib.Path(a.binary)
assert hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest() == a.sha
profile = json.loads((source / 'profile.json').read_text())
model = profile['MEMRA_MODELS'].split('=')[0]
key = secrets.token_hex(24)
(out / 'keys.toml').write_text('[[keys]]\ntenant="prime-fairness-gate"\nsha256="' + hashlib.sha256(key.encode()).hexdigest() + '"\n')
(out / 'keys.toml').chmod(0o600)
(out / 'models.toml').write_bytes((source / 'models.toml').read_bytes())
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
profile.update(MEMRA_ADDR=f'127.0.0.1:{port}', MEMRA_MODEL_METADATA=str(out / 'models.toml'),
               MEMRA_API_KEYS=str(out / 'keys.toml'), MEMRA_REQUEST_LEDGER=str(out / 'ledger.jsonl'),
               MEMRA_PRIME_CHUNK=str(a.chunk), MEMRA_PRIME_YIELD=str(int(a.arm == 'on')),
               MEMRA_MAX_SESSIONS='4', MEMRA_TICK_TRACE='1', MEMRA_TTFT_TRACE='1')
profile.pop('MEMRA_REQUEST_LEDGER', None)  # public engine build
if a.mode != 'chunk':
    profile.update(MEMRA_SPEC_GATE_LOW='64', MEMRA_SPEC_GATE_HIGH='65')
if a.oracle:
    profile['MEMRA_ALLOC_TRACE'] = '1'
(out / 'profile.json').write_text(json.dumps(profile, indent=2))
nonce = uuid.uuid4().hex
log = (out / 'server.log').open('w')
proc = subprocess.Popen([str(binary)], cwd=out, env={'PATH': os.environ['PATH'], **profile, 'MEMRA_METRICS_TOKEN': key},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
identity = {'binary_sha256': a.sha, 'boot_nonce': nonce, 'pid': proc.pid,
            'start_ticks': pathlib.Path(f'/proc/{proc.pid}/stat').read_text().split()[21], 'args': vars(a)}
(out / 'identity.json').write_text(json.dumps(identity, indent=2))
print('BOOT', json.dumps(identity), flush=True)
url = f'http://127.0.0.1:{port}'

def request(path, body=None, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}), timeout=timeout)

def call(label, payload, greedy=True):
    body = dict(payload)
    body.update(model=model, stream=True, stream_options={'include_usage': True}, cache_salt='gate-chain' if a.mode == 'chain' else 'gate-' + label)
    if greedy:
        body.update(temperature=0, seed=42, top_p=1, top_k=0, min_p=0,
                    presence_penalty=0, frequency_penalty=0, repetition_penalty=1)
    started = time.monotonic()
    events, content, reasoning = [], [], []
    row = {'label': label, 'ttft_s': None, 'done': False, 'usage': {}}
    with request('/v1/chat/completions', body) as response:
        row['status'] = response.status
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                row['done'] = True
                break
            event = json.loads(data)
            events.append(event)
            if event.get('error'):
                raise RuntimeError(event['error'])
            if event.get('usage'):
                row['usage'] = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
                if (text or thought) and row['ttft_s'] is None:
                    row['ttft_s'] = time.monotonic() - started
                content.append(text)
                reasoning.append(thought)
                if choice.get('finish_reason'):
                    row['finish'] = choice['finish_reason']
    row.update(total_s=time.monotonic() - started, content=''.join(content), reasoning=''.join(reasoning))
    row['output_sha256'] = hashlib.sha256(json.dumps([row['content'], row['reasoning']], ensure_ascii=False).encode()).hexdigest()
    (out / (label + '-request.json')).write_text(json.dumps(body))
    (out / (label + '-events.json')).write_text(json.dumps(events))
    (out / (label + '-result.json')).write_text(json.dumps(row, indent=2))
    assert row['done'] and row['finish'] in ('stop', 'length') and row['ttft_s'] is not None, row
    print('REQUEST', json.dumps({k: v for k, v in row.items() if k not in ('content', 'reasoning')}), flush=True)
    return row

rows = []
try:
    deadline = time.monotonic() + 180
    while True:
        if proc.poll() is not None:
            raise RuntimeError('server exited at boot')
        try:
            with request('/readyz', timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('boot')
        time.sleep(.5)
    with request('/metrics') as response:
        before = json.load(response)
    small = {'messages': [{'role': 'user', 'content': 'Garden records. ' +
             'Morning sun reaches two beds. Volunteers water seedlings and record soil moisture. '*60 +
             '\nGive one practical next step in one short sentence.'}], 'max_tokens': 128}
    if a.mode == 'chain':
        messages = small['messages'].copy()
        for turn in range(1, 5):
            row = call(f'turn{turn}', {'messages': messages, 'max_tokens': 256})
            rows.append(row)
            answer = ('<think>' + row['reasoning'] + '</think>\n' if row['reasoning'] else '') + row['content']
            messages = messages + [{'role': 'assistant', 'content': answer},
                                  {'role': 'user', 'content': f'Follow-up {turn}: choose one detail and give one concrete action in a short sentence.'}]
    else:
        long = json.loads(pathlib.Path(a.long_request).read_text())
        long['max_tokens'] = 64 if a.mode != 'chunk' else 2048
        if a.mode == 'pair':
            offset = (out / 'server.log').stat().st_size
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(call, 'long', long)
                deadline = time.monotonic() + 300
                while '[prime-chunk]' not in (out / 'server.log').read_text()[offset:]:
                    if first.done():
                        raise RuntimeError('long finished without a chunk receipt')
                    if time.monotonic() > deadline:
                        raise TimeoutError('first long chunk')
                    time.sleep(.01)
                time.sleep(.05)
                second = pool.submit(call, 'small', small)
                rows = [first.result(), second.result()]
        else:
            rows.append(call('long', long, greedy=a.mode != 'chunk'))
            if a.mode == 'solo':
                rows.append(call('small', small))
    with request('/metrics') as response:
        after = json.load(response)
    text = (out / 'server.log').read_text()
    yields = len(re.findall(r'^\[prime-yield\]', text, re.M))
    summary = {'rows': [{k: v for k, v in row.items() if k not in ('content', 'reasoning')} for row in rows],
               'yield_count': yields, 'metric_delta': {k: after.get(k, 0)-before.get(k, 0)
                for k in ('served_spec','served_dspark','served_plain','step_oom_parks','admission_vram_defers','admission_session_defers')},
               'boot_nonce': nonce}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    assert summary['metric_delta']['step_oom_parks'] == 0, summary
    if a.mode == 'pair':
        assert yields > 0 and summary['metric_delta']['served_plain'] == 0, summary
    print('SUMMARY', json.dumps(summary), flush=True)
finally:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    log.close()
    (out / 'cleanup.json').write_text(json.dumps({'own_pid': proc.pid, 'exit': proc.returncode,
        'gpu_compute_apps_while_lock_held': subprocess.check_output(apps_cmd, text=True).strip()}, indent=2))
    lock.close()
