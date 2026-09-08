"""Replay immutable receipt requests on an isolated, locked qualification server."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--templates', type=Path, required=True)
ap.add_argument('--binary', type=Path, required=True)
ap.add_argument('--run', type=Path, required=True)
ap.add_argument('--cells', nargs='+', required=True)
ap.add_argument('--control', action='store_true')
ap.add_argument('--harness-tools', type=Path)
a = ap.parse_args()
a.run.mkdir(parents=True, exist_ok=False)
lock = open('/tmp/memra-gpu.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
sys.path.insert(0, str(a.harness_tools or a.templates.parent / 'baseline/tools'))
import cache_qualification as cq
profile = json.loads((a.templates / 'profile.json').read_text())
key = secrets.token_hex(24)
(a.run / 'key').write_text(key)
(a.run / 'key').chmod(0o600)
(a.run / 'keys.toml').write_text('[[keys]]\ntenant="retained-admission"\nsha256="' + hashlib.sha256(key.encode()).hexdigest() + '"\n')
(a.run / 'keys.toml').chmod(0o600)
shutil.copyfile(a.templates / 'models.toml', a.run / 'models.toml')
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
profile.pop('MEMRA_REQUEST_LEDGER', None)
profile.update(MEMRA_ADDR=f'127.0.0.1:{port}', MEMRA_MODEL_METADATA=str(a.run / 'models.toml'), MEMRA_API_KEYS=str(a.run / 'keys.toml'))
(a.run / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
os.environ['CACHE_BATTERY_KEY_FILE'] = str(a.run / 'key')
log = (a.run / 'server.log').open('w')
proc = subprocess.Popen([str(a.binary)], cwd=a.run, env={'PATH': os.environ['PATH'], 'HOME': str(a.run), 'MEMRA_METRICS_TOKEN': key, **profile}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(a.run / 'server.pid').write_text(str(proc.pid))
base = f'http://127.0.0.1:{port}'
report = {'binary_sha256': hashlib.file_digest(a.binary.open('rb'), 'sha256').hexdigest(), 'rows': [], 'passed': False}
def get(path):
    with urllib.request.urlopen(urllib.request.Request(base + path, headers={'Authorization': 'Bearer ' + key}), timeout=10) as r:
        return json.load(r)
def metrics(name):
    value = get('/metrics')
    (a.run / (name + '-metrics.json')).write_text(json.dumps(value, indent=2) + '\n')
    return value
try:
    for _ in range(300):
        assert proc.poll() is None, 'server exited during boot'
        try:
            get('/readyz')
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError('readiness timeout')
    metrics('ready')
    for cell in a.cells:
        body = json.loads((a.templates / (cell + '-request.json')).read_text())
        before = metrics(cell + '-before')
        offset = (a.run / 'server.log').stat().st_size
        row = cq.completion(base, body, a.run, cell, raw_tape=not body.get('stream', False) and body.get('temperature') == 0)
        time.sleep(1)
        after = metrics(cell + '-after')
        with (a.run / 'server.log').open('rb') as stream:
            stream.seek(offset)
            delta = stream.read().decode(errors='replace')
        faults = [l for l in delta.splitlines() if re.search(r'step[- ]OOM|source=oom-replay|engine-error|CUDA_ERROR|panicked|FATAL|CRITICAL', l)]
        excerpts = [l for l in delta.splitlines() if re.search(r'retained prefix|DSPARK restore|\[dspark-acc\]|\[admit-oom\]|\[admission\] request cost|\[dflash-oracle\]|cold admission required', l)]
        rec = {'cell': cell, 'row': row, 'faults': faults, 'log': excerpts, 'before': before, 'after': after}
        report['rows'].append(rec)
        print(json.dumps({'cell': cell, 'verdict': row['verdict'], 'prompt': row.get('prompt_tokens'), 'cached': row.get('cached_tokens'), 'completion': row.get('completion_tokens'), 'error': row.get('error')}), flush=True)
        (a.run / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
        assert not faults, faults[:1]
        if not a.control:
            assert row['completed_answer'] or (row.get('finish') == 'length' and row.get('completion_tokens') == body['max_tokens']), row.get('error')
            assert row.get('spec', {}).get('drafted', 0) > 0, 'DFlash did not engage'
    report['passed'] = True
except Exception as error:
    report['error'] = str(error)
    print('FAILED', str(error), flush=True)
finally:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    log.close()
    report['server_exit'] = proc.returncode
    report['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    (a.run / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
if not report['passed']:
    sys.exit(1)
