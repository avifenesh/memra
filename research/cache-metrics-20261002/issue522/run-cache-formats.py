#!/usr/bin/env python3
"""Verify cache accounting and Prometheus against both completion JSON formats."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--promtool', required=True)
parser.add_argument('--credentials-dir', type=Path, required=True)
a = parser.parse_args()
root = Path.cwd()
receipt = root / 'research/cache-metrics-20261002/issue522'
out = receipt / 'formats'
out.mkdir(parents=True, exist_ok=True)
binary = root / 'target/release/memra-server'
model = '/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf'
prior = json.loads((receipt / 'identity.json').read_text())
assert hashlib.file_digest(open(binary, 'rb'), 'sha256').hexdigest() == prior['binary_sha256']
assert os.stat(model).st_size == prior['model_bytes']
os.fstat(9)
a.credentials_dir.mkdir(parents=True, exist_ok=True)
api_file, metric_file = (a.credentials_dir / name for name in ('api-key', 'metrics-token'))
api, operator = secrets.token_hex(24), secrets.token_hex(24)
for path, value in ((api_file, api), (metric_file, operator)):
    path.write_text(value)
    path.chmod(0o600)
identity = {key: prior[key] for key in ('binary_source_sha', 'binary_sha256', 'model_sha256', 'model_bytes', 'promtool_sha256')}
identity.update(collector_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                ports=[18117, 18118], timeout_seconds=600, minimum_free_vram_mib=12000)
(out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
results = []
try:
    for mode, port in [('native', 18117), ('openai', 18118)]:
        phase = out / mode
        phase.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith('MEMRA_')}
        env.update(MEMRA_COMPAT=mode, MEMRA_MODELS='metrics-q9=' + model,
                   MEMRA_ADDR='127.0.0.1:' + str(port), MEMRA_CTX='8192',
                   MEMRA_SERVE_SPEC='0', MEMRA_MAX_SESSIONS='8',
                   MEMRA_PREFIX_CACHE_MB='512', MEMRA_GDN_CHUNK='32',
                   MEMRA_REUSE_POOL='0', MEMRA_AFFINITY='0',
                   MEMRA_API_KEY=api, MEMRA_METRICS_TOKEN=operator)
        public = {k: ('fixture credential omitted' if k in ('MEMRA_API_KEY', 'MEMRA_METRICS_TOKEN') else v)
                  for k, v in env.items() if k.startswith('MEMRA_')}
        (phase / 'environment.json').write_text(json.dumps(public, indent=2) + '\n')
        subprocess.run(['bash', 'tools/port-guard.sh', 'check', 'cache-formats', str(port)], check=True)
        (phase / 'gpu-before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
        log = open(phase / 'server.log', 'w')
        telemetry = open(phase / 'gpu-250ms.csv', 'w')
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw', '--format=csv', '--loop-ms=250'], stdout=telemetry, stderr=subprocess.STDOUT, pass_fds=(9,))
        proc = subprocess.Popen([str(binary)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, pass_fds=(9,))
        ready = threading.Event()
        started = []
        def copy_log():
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if f'[server] listening on http://127.0.0.1:{port}' in line:
                    started.append(True)
                    ready.set()
            ready.set()
        reader = threading.Thread(target=copy_log)
        reader.start()
        try:
            if not ready.wait(180) or not started:
                raise RuntimeError('server startup failed; inspect ' + str(phase / 'server.log'))
            subprocess.run(['bash', '-c', 'source tools/port-guard.sh; memra_port_owned cache-formats "$1" "$2"', '_', str(port), str(proc.pid)], check=True)
            with open(phase / 'gate.log', 'w') as output:
                result = subprocess.run(['python3', 'tools/cache-meter-gate.py', 'http://127.0.0.1:' + str(port),
                                         'metrics-q9', '--prometheus', '--promtool', a.promtool,
                                         '--api-key-file', str(api_file), '--metrics-token-file', str(metric_file),
                                         '--raw-out', str(phase / 'raw.jsonl')],
                                        env=env, stdout=output, stderr=subprocess.STDOUT, timeout=240)
            results.append({'format': mode, 'gate_exit': result.returncode})
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            reader.join()
            log.close()
            monitor.terminate()
            monitor.wait()
            telemetry.close()
            (phase / 'gpu-after-cleanup.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
finally:
    api_file.unlink(missing_ok=True)
    metric_file.unlink(missing_ok=True)
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results))
raise SystemExit(0 if len(results) == 2 and all(r['gate_exit'] == 0 for r in results) else 1)
