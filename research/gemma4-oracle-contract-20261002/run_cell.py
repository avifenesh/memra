#!/usr/bin/env python3
"""Bounded sequential HTTP/direct-API contract capture; no health polling."""
import argparse, hashlib, json, os, pathlib, subprocess, threading, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--bin-dir', required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--inputs', required=True)
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=False)
inputs = pathlib.Path(a.inputs).read_bytes()
rows = [json.loads(x) for x in inputs.splitlines()]
for row in rows:
    assert hashlib.sha256(row['request']['prompt'].encode()).hexdigest() == row['prompt_sha256']
bin_dir = pathlib.Path(a.bin_dir)
commands = []

def request(path, payload):
    req = urllib.request.Request('http://127.0.0.1:18140' + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type':'application/json','Authorization':'Bearer local-numerical-probe'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)

for arm in ['http', 'http-unsplit']:
    cmd = [str(bin_dir/'llama-server'), '-m', a.model, '--host', '127.0.0.1', '--port', '18140',
           '-ngl','99','-c','8192','-b','4096','-ub','2048','-np','1','--jinja','--no-warmup',
           '-t','2','-tb','2','--api-key','local-numerical-probe','-lv','4']
    if arm == 'http-unsplit': cmd += ['--ctx-checkpoints', '0']
    commands.append(cmd)
    env = dict(os.environ, DIAG_LOGITS_PATH=str(out/(arm+'-logits.jsonl')))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    ready = threading.Event()
    def capture_log():
        with (out/(arm+'.log')).open('wb') as log:
            for line in iter(proc.stdout.readline, b''):
                log.write(line); log.flush()
                if b'listening on http://' in line: ready.set()
        ready.set()  # a terminated server must wake the readiness wait too
    reader = threading.Thread(target=capture_log, daemon=True); reader.start()
    try:
        if not ready.wait(180): raise TimeoutError('server readiness deadline')
        if proc.poll() is not None: raise RuntimeError(f'server exited before readiness: {proc.returncode}')
        with (out/(arm+'.jsonl')).open('w') as dest:
            for row in rows:
                tokens = request('/tokenize', {'content':row['request']['prompt'],'add_special':True,'parse_special':True})
                payload = dict(prompt=row['request']['prompt'],temperature=0,n_predict=1,n_probs=5,
                               post_sampling_probs=False,return_tokens=True,cache_prompt=False,seed=0)
                response = request('/completion', payload)
                dest.write(json.dumps(dict(i=row['i'],tokens=tokens,response=response))+'\n'); dest.flush()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        reader.join(timeout=30)
        if reader.is_alive(): raise RuntimeError('server log reader failed to exit')
        (out/(arm+'-exit.json')).write_text(json.dumps({'returncode':proc.returncode})+'\n')
for mode in ['aligned','split4','output-limit','common-init']:
    cmd = [str(bin_dir/'oracle-probe'), a.model, mode]; commands.append(cmd)
    with (out/f'{mode}.jsonl').open('wb') as stdout, (out/f'{mode}.log').open('wb') as stderr:
        subprocess.run(cmd, input=inputs, stdout=stdout, stderr=stderr, check=True, timeout=180,
                       env=dict(os.environ, ORACLE_LOGITS_DIR=str(out/(mode+'-full'))))
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
