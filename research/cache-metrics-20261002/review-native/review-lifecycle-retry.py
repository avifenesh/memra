#!/usr/bin/env python3
"""Paired plain/MTP hybrid metrics cell. Requires the inherited canonical GPU lease."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading
import urllib.request
import sys

p = argparse.ArgumentParser()
p.add_argument('--promtool', required=True)
p.add_argument('--credentials-dir', type=Path, required=True)
a = p.parse_args()
root = Path.cwd()
sys.path.insert(0,str(root/'tools'))
from prometheus_metrics import parse_samples,scalar,histogram
out = root / 'research/cache-metrics-20261002/review-native/lifecycle-retry'
out.mkdir(parents=True, exist_ok=True)
binary = root / 'target/release/memra-server'
model = '/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf'
alias = 'metrics-q9'
os.fstat(9)
a.credentials_dir.mkdir(parents=True, exist_ok=True)
api_file, metrics_file = (a.credentials_dir / name for name in ('api-key', 'metrics-token'))
api, operator = secrets.token_hex(24), secrets.token_hex(24)
for path, value in ((api_file, api), (metrics_file, operator)):
    path.write_text(value)
    path.chmod(0o600)
identity = {
    'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'])).hexdigest(),
    'binary_sha256': hashlib.file_digest(open(binary, 'rb'), 'sha256').hexdigest(),
    'model_sha256': (root / 'research/cache-metrics-20261002/issue522/model.sha256').read_text().split()[0],
    'model_bytes': os.stat(model).st_size,
    'promtool_sha256': hashlib.file_digest(open(a.promtool, 'rb'), 'sha256').hexdigest(),
    'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'ports': [18111, 18112, 18113, 18114, 18115, 18116],
    'binary_source_diff_sha256': hashlib.sha256((root / 'research/cache-metrics-20261002/combined-v2/source.diff').read_bytes()).hexdigest(),
    'binary_source_sha': (root / 'research/cache-metrics-20261002/review-fixes/source-head.txt').read_text().strip(),
    'timeout_seconds': 1200,
    'minimum_free_vram_mib': 12000,
}
(out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')


class Server:
    def __init__(self, phase, port, spec):
        self.out = out / phase
        self.out.mkdir(parents=True, exist_ok=True)
        self.binary = binary
        self.port = port
        self.base = 'http://127.0.0.1:' + str(port)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('MEMRA_') or k in ('MEMRA_GPU_LOCK', 'MEMRA_CI_LOCK', 'MEMRA_CI_LOCK_HELD', 'MEMRA_RIG_LOCK_FD')}
        self.env.update(MEMRA_COMPAT='openai', MEMRA_MODELS=alias + '=' + model,
                        MEMRA_ADDR='127.0.0.1:' + str(port), MEMRA_CTX='8192',
                        MEMRA_PREFIX_CACHE_MB='512', MEMRA_MAX_SESSIONS=('1' if phase == 'queued' else '8'),
                        MEMRA_REUSE_POOL='0', MEMRA_AFFINITY='0', MEMRA_GDN_CHUNK='32',
                        MEMRA_API_KEY=api, MEMRA_METRICS_TOKEN=operator)
        if not spec:
            self.env['MEMRA_SERVE_SPEC'] = '0'
        if phase == 'fault': self.env['MEMRA_PANIC_AFTER'] = '1'
        if phase == 'batch':
            self.env.pop('MEMRA_API_KEY',None)
            self.env['MEMRA_API_KEYS']='bulk:'+hashlib.sha256(api.encode()).hexdigest()+':batch'
        public_env = {k: ('fixture credential omitted' if k in ('MEMRA_API_KEY', 'MEMRA_METRICS_TOKEN') else v)
                      for k, v in self.env.items() if k.startswith('MEMRA_')}
        (self.out / 'environment.json').write_text(json.dumps(public_env, indent=2) + '\n')
        self.panic_event = threading.Event()
        self.ready = threading.Event()
        self.started = False
        self.proc = self.monitor = None

    def start(self):
        subprocess.run(['bash', 'tools/port-guard.sh', 'check', 'metrics-live', str(self.port)], check=True)
        (self.out / 'gpu-before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
        self.log = open(self.out / 'server.log', 'w')
        self.telemetry = open(self.out / 'gpu-250ms.csv', 'w')
        self.monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw', '--format=csv', '--loop-ms=250'], stdout=self.telemetry, stderr=subprocess.STDOUT, pass_fds=(9,))
        self.proc = subprocess.Popen([str(self.binary)], env=self.env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, pass_fds=(9,))

        def copy_log():
            for line in self.proc.stdout:
                self.log.write(line)
                self.log.flush()
                if '[worker] PANIC in the GPU worker thread:' in line: self.panic_event.set()
                if '[server] listening on ' + self.base in line:
                    self.started = True
                    self.ready.set()
            self.ready.set()

        self.thread = threading.Thread(target=copy_log)
        self.thread.start()
        if not self.ready.wait(300) or not self.started:
            raise RuntimeError('server did not become ready; inspect ' + str(self.out / 'server.log'))
        subprocess.run(['bash', '-c', 'source tools/port-guard.sh; memra_port_owned metrics-live "$1" "$2"', '_', str(self.port), str(self.proc.pid)], check=True)

    def stop(self):
        if self.proc is not None:
            (self.out / 'gpu-at-end.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(20)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            self.thread.join()
            self.log.close()
        if self.monitor is not None:
            self.monitor.terminate()
            self.monitor.wait()
            self.telemetry.close()
        (self.out / 'gpu-after-cleanup.txt').write_bytes(subprocess.check_output(['nvidia-smi']))


results = []
try:
    for phase, port, spec in [('mtp', 18112, True)]:
        server = Server(phase, port, spec)
        try:
            server.start()
            if phase == 'batch':
                body={'model':alias,'prompt':'Count from one to twenty in words, separated by commas.','max_tokens':16,'temperature':0,'stream':False}
                req=urllib.request.Request(server.base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+api})
                with urllib.request.urlopen(req,timeout=120) as response: result=json.loads(response.read())
                req=urllib.request.Request(server.base+'/metrics',headers={'Accept':'text/plain','Authorization':'Bearer '+operator})
                with urllib.request.urlopen(req,timeout=15) as response: exposition=response.read().decode()
                (server.out/'response.json').write_text(json.dumps(result,indent=2));(server.out/'metrics.prom').write_text(exposition)
                samples=parse_samples(exposition);labels={'model':alias,'route':'hybrid','lane':'harvest'}
                checks={'http_harvest':scalar(samples,'memra_requests_total',**labels,code='200')==1,
                        'worker_harvest':histogram(samples,'memra_e2e_seconds',**labels)['count']==1,
                        'output_tokens':histogram(samples,'memra_completion_tokens',**labels)['sum']==result['usage']['completion_tokens'],
                        'no_interactive_success':samples.get(('memra_requests_total',tuple(sorted({'model':alias,'route':'hybrid','lane':'interactive','code':'200'}.items()))),0)==0}
                parsed=subprocess.run([a.promtool,'check','metrics'],input=exposition,text=True,capture_output=True,timeout=30)
                checks['promtool']=parsed.returncode==0
                (server.out/'checks.json').write_text(json.dumps(checks,indent=2));results.append({'phase':phase,'gate':'resolved-batch-lane','exit':0 if all(checks.values()) else 1})
                continue
            if phase == 'fault':
                import importlib.util
                spec = importlib.util.spec_from_file_location('native_fault', '/home/avifenesh/.local/state/memra-rig-20261002/scratch/A/native-fault.py')
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                code = module.exercise(server.base, api, operator, server.out / 'fault', server.panic_event, a.promtool)
                results.append({'phase': phase, 'gate': 'native-worker-fault', 'exit': code})
                continue
            if phase == 'cache':
                cmd = ['python3', 'tools/cache-meter-gate.py', server.base, alias,
                       '--raw-out', str(server.out / 'cache-meter.jsonl'), '--prometheus',
                       '--promtool', a.promtool, '--api-key-file', str(api_file),
                       '--metrics-token-file', str(metrics_file)]
                with open(server.out / 'cache-meter.log', 'w') as log:
                    r = subprocess.run(cmd, env=server.env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                results.append({'phase': phase, 'gate': 'cache-meter', 'exit': r.returncode})
                continue
            cmd = ['python3', 'tools/metrics-live-gate.py', '--base', server.base, '--model', alias,
                   '--api-key-file', str(api_file), '--metrics-token-file', str(metrics_file),
                   '--out', str(server.out / 'lifecycle'), '--promtool', a.promtool]
            if spec:
                cmd.append('--spec')
            if phase == 'queued':
                cmd.append('--queue')
            if phase.startswith('baseline-'):
                cmd.append('--identity-only')
            with open(server.out / 'lifecycle.log', 'w') as log:
                r = subprocess.run(cmd, env=server.env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            results.append({'phase': phase, 'gate': 'lifecycle', 'exit': r.returncode})
        finally:
            server.stop()
    def generated_text(phase):
        row = json.loads((out / phase / 'lifecycle/clean.json').read_text())
        return ''.join(choice.get('text') or '' for entry in row['events']
                       for choice in entry['event'].get('choices', []))

finally:
    (out / 'native-results.json').write_text(json.dumps(results, indent=2) + '\n')
    api_file.unlink(missing_ok=True)
    metrics_file.unlink(missing_ok=True)
print(json.dumps(results, indent=2))
raise SystemExit(0 if results and all(r['exit'] == 0 for r in results) else 1)
