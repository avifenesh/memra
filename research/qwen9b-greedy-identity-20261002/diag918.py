#!/usr/bin/env python3
import hashlib,json,os,secrets,subprocess,threading,urllib.request
from pathlib import Path
root=Path.cwd()
out=root/'research/qwen9b-greedy-identity-20261002/before'
out.mkdir(parents=True,exist_ok=True)
binary=root/'target/release/memra-server'
model='/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf'
alias='metrics-q9'
api,operator=secrets.token_hex(24),secrets.token_hex(24)
os.fstat(9)
(out/'identity.json').write_text(json.dumps({'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'binary_sha256':hashlib.file_digest(open(binary,'rb'),'sha256').hexdigest(), 'binary_source':'925b102a6e92cd1896df23a7533bb1e56f0ca05c','model_sha256':'52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de','driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
class Server:
    def __init__(self, phase, port, spec):
        self.out = out / phase
        self.out.mkdir(parents=True, exist_ok=True)
        self.binary = binary
        self.port = port
        self.base = 'http://127.0.0.1:' + str(port)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('MEMRA_') or k in ('MEMRA_GPU_LOCK', 'MEMRA_CI_LOCK', 'MEMRA_CI_LOCK_HELD', 'MEMRA_RIG_LOCK_FD')}
        self.env.update(MEMRA_COMPAT='native', MEMRA_MODELS=alias + '=' + model,
                        MEMRA_ADDR='127.0.0.1:' + str(port), MEMRA_CTX='8192',
                        MEMRA_PREFIX_CACHE_MB='512', MEMRA_MAX_SESSIONS=('1' if phase == 'queued' else '8'),
                        MEMRA_REUSE_POOL='0', MEMRA_AFFINITY='0', MEMRA_GDN_CHUNK='32',
                        MEMRA_API_KEY=api, MEMRA_METRICS_TOKEN=operator)
        if not spec:
            self.env['MEMRA_SERVE_SPEC'] = '0'
        public_env = {k: ('fixture credential omitted' if k in ('MEMRA_API_KEY', 'MEMRA_METRICS_TOKEN') else v)
                      for k, v in self.env.items() if k.startswith('MEMRA_')}
        (self.out / 'environment.json').write_text(json.dumps(public_env, indent=2) + '\n')
        self.env.update(overrides)
        (self.out / 'environment.json').write_text(json.dumps({k:v for k,v in self.env.items() if k.startswith('MEMRA_') and k not in ('MEMRA_API_KEY','MEMRA_METRICS_TOKEN')}, indent=2))
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


def post(server,path,body):
    req=urllib.request.Request(server.base+path,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+api,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:
        return r.read().decode()

prompt='Count from one to twenty in words, separated by commas.'
results={}
for phase,overrides in [('plain',{}),('mtp',{}),('mtp-k1',{'MEMRA_SPEC_K':'1'}),('mtp-rowwise',{'MEMRA_SPEC_VERIFY_ROWWISE':'1'})]:
    server=Server(phase,18111,phase!='plain')
    rows={}
    try:
        server.start()
        request={'model':alias,'prompt':prompt,'max_tokens':16,'temperature':0,'stream':True,'stream_options':{'include_usage':True},'cache_salt':'metric-clean'}
        raw=post(server,'/v1/completions',request)
        (server.out/'primary.sse').write_text(raw)
        (server.out/'primary-request.json').write_text(json.dumps(request,indent=2))
        events=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line[6:]!='[DONE]']
        assert not any('error' in e for e in events),events
        rows['primary']={'text':''.join(e.get('text','') for e in events),'events':events}
        tokenized=json.loads(post(server,'/v1/tokenize',{'model':alias,'prompt':prompt}))
        (server.out/'tokenize.json').write_text(json.dumps(tokenized,indent=2))
        ids=tokenized['tokens']
        assert len(ids)==12,tokenized
        for n in [1,8,12,15,16,17,32]:
            request={'model':alias,'prompt_ids':(ids*3)[:n],'max_tokens':16,'temperature':0,'stream':False,'cache_salt':'length-'+str(n)}
            result=json.loads(post(server,'/v1/completions',request))
            (server.out/('length-'+str(n)+'.json')).write_text(json.dumps({'request':request,'response':result},indent=2))
            assert 'error' not in result,result
            rows[str(n)]={k:result.get(k) for k in ['text','tokens','n_tokens','prompt_tokens','cached_tokens','stop_reason']}
        results[phase]=rows
    finally:
        server.stop()
        (out/'results.json').write_text(json.dumps(results,indent=2))
comparisons=[]
for phase in ['mtp','mtp-k1','mtp-rowwise']:
    for label in results['plain']:
        a,b=results['plain'][label],results[phase][label]
        comparisons.append({'phase':phase,'case':label,'text_equal':a['text']==b['text'],'tokens_equal':a.get('tokens')==b.get('tokens'),'plain_text':a['text'],'mtp_text':b['text']})
(out/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
print(json.dumps(comparisons,indent=2))
# Diagnostic captures known failing comparisons; exit reports the unchanged identity gate.
raise SystemExit(0 if all(x['text_equal'] for x in comparisons) else 1)
