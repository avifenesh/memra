"""Bounded native correctness pilot. Raw outputs precede parsing. No speed claim."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/workspace/adaptive-draft')
ATTEMPT = sys.argv[1] if len(sys.argv)>1 else 'r1'
assert re.fullmatch(r'[a-zA-Z0-9_-]+', ATTEMPT)
OUT = ROOT / f'raw/pilot-{ATTEMPT}'
OUT.mkdir(parents=True, exist_ok=True)
BIN = ROOT / 'memra/target/release'
MODELS = ROOT / 'models'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def run(name, argv, config, timeout=300):
    log = OUT / f'{name}.log'
    if log.exists():
        raise RuntimeError(f'Refusing to overwrite {log}')
    env = {k:v for k,v in os.environ.items() if not k.startswith('MEMRA_')}
    env.update(config)
    start = time.time()
    with log.open('wb') as output:
        proc = subprocess.Popen(argv, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = -999
    elapsed = time.time()-start
    text = log.read_text(errors='replace')
    row = dict(id=name, utc=dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(), argv=argv,
               config=config, exit_code=code, wall_seconds=elapsed, raw_sha256=digest(log))
    match = re.search(r'stream agreement (\d+)/(\d+)', text)
    if match:
        plain = re.search(r'plain tokens: (\[[^\n]+\])', text)
        spec = re.search(r'spec tokens: (\[[^\n]+\])', text)
        row['token_identity'] = bool(plain and spec and json.loads(plain[1]) == json.loads(spec[1]) and len(json.loads(plain[1])) > 0)
    if 'SELF-CONSISTENCY PASS' in text:
        row['token_identity'] = True
    row['acceptance_lines'] = [x for x in text.splitlines() if 'acceptance:' in x or '[gemma-spec]' in x]
    counts = re.findall(r'accepted=(\d+)', text) + re.findall(r'acceptance: (\d+)/', text)
    row['nonzero_acceptance'] = any(int(x)>0 for x in counts)
    row['status'] = 'pass' if code == 0 and row.get('token_identity') and row['nonzero_acceptance'] else 'failed_or_refused'
    with (ROOT/f'raw/runs-{ATTEMPT}.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)
    if code != 0:
        subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'], stdout=(OUT/f'{name}-processes.csv').open('w'))
    return row

assert (MODELS/'VERIFIED').exists(), 'Artifacts must be verified before GPU work'
source_commit = (ROOT/'SOURCE_COMMIT').read_text().strip() if (ROOT/'SOURCE_COMMIT').exists() else '3bb21381848067d922dec1320261846f99ceb29a'
manifest = {'kind':'correctness-pilot-not-serving-benchmark', 'source_commit':source_commit,
            'binaries':{name:digest(BIN/name) for name in ['gemma-gate','run-spec','draft_prompt']}}
source = 'https://raw.githubusercontent.com/lm-sys/FastChat/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl'
data = urllib.request.urlopen(source, timeout=60).read()
(OUT/'mt-bench-source.jsonl').write_bytes(data)
manifest['prompt_source'] = source
manifest['prompt_source_sha256'] = hashlib.sha256(data).hexdigest()
items = [json.loads(line) for line in data.splitlines()]
selected = [next(x for x in items if x['category']==c) for c in ['coding','reasoning','writing']]
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
prompts = []
for item in selected:
    path = OUT / f'prompt-{item["question_id"]}.txt'
    path.write_text(item['turns'][0])
    prompts.append(path)

with open('/tmp/memra-5090.lock','w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if processes:
        raise RuntimeError(f'GPU already occupied: {processes}')
    telemetry = (ROOT/'raw/telemetry.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv','-lms','250'], stdout=telemetry)
    try:
        for index,prompt in enumerate(prompts):
            run(f'qwen-p{index}-k1to8',[str(BIN/'run-spec'),str(MODELS/'qwen9.gguf')],
                {'MEMRA_PROMPT_FILE':str(prompt),'MEMRA_CHAT':'1','MEMRA_NGEN':'96','MEMRA_PRINT_TEXT':'1','MEMRA_SPEC_STATS':'1'},timeout=600)
            token_log = OUT/f'gemma-p{index}-tokenize.log'
            with token_log.open('wb') as f:
                result = subprocess.run([str(BIN/'draft_prompt'),str(MODELS/'gemma12.gguf'),str(prompt)],stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Native tokenizer failed: {token_log}')
            ids = json.loads(token_log.read_text().splitlines()[-1])
            for k in [1,2,4,6,8]:
                run(f'gemma-p{index}-k{k}',[str(BIN/'gemma-gate'),str(MODELS/'gemma12.gguf')]+[str(t) for t in ids],
                    {'MEMRA_SPEC':str(k),'MEMRA_DRAFT':str(MODELS/'gemma12-draft.gguf'),'MEMRA_NGEN':'96',
                     'MEMRA_GATE_DUMP_TOKENS':'1','MEMRA_SPEC_STATS':'1','MEMRA_SPEC_ADAPT':'0','MEMRA_SPEC_CAPMAX':'8','MEMRA_SPEC_PMIN':'0','MEMRA_SPEC_PMIN_INROUND':'0'},timeout=300)
            # Recorder engagement and ON/OFF output check. This public smoke set is
            # not training/calibration data for the eventual research comparison.
            trace_path = OUT/f'gemma-p{index}-trace.jsonl'
            run(f'gemma-p{index}-k4-traced',[str(BIN/'gemma-gate'),str(MODELS/'gemma12.gguf')]+[str(t) for t in ids],
                {'MEMRA_SPEC':'4','MEMRA_DRAFT':str(MODELS/'gemma12-draft.gguf'),'MEMRA_NGEN':'96',
                 'MEMRA_GATE_DUMP_TOKENS':'1','MEMRA_SPEC_STATS':'1','MEMRA_SPEC_ADAPT':'0','MEMRA_SPEC_CAPMAX':'8',
                 'MEMRA_SPEC_PMIN':'0','MEMRA_SPEC_PMIN_INROUND':'0','MEMRA_GEMMA_DRAFT_TRACE':str(trace_path)})
            rows = [json.loads(x) for x in trace_path.read_text().splitlines()]
            assert rows and all(len(x['confidence']) == x['drafted'] for x in rows)
            assert any(any(p>0 for p in x['confidence']) for x in rows), 'Recorder captured no computed confidence'
            off = re.search(r'spec tokens: (\[[^\n]+\])',(OUT/f'gemma-p{index}-k4.log').read_text())
            on = re.search(r'spec tokens: (\[[^\n]+\])',(OUT/f'gemma-p{index}-k4-traced.log').read_text())
            assert off and on and json.loads(off[1]) == json.loads(on[1]), 'Recorder ON/OFF token mismatch'
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        telemetry.close()
finished = [json.loads(x) for x in (ROOT/f'raw/runs-{ATTEMPT}.jsonl').read_text().splitlines()]
assert all(x['status']=='pass' for x in finished), 'At least one correctness cell failed; see raw run records'
(ROOT/'PILOT_READY.json').write_text(json.dumps({'attempt':ATTEMPT,'runs':f'raw/runs-{ATTEMPT}.jsonl','manifest':str(OUT/'manifest.json'),'utc':dt.datetime.now(dt.timezone.utc).isoformat()}))
(ROOT/'PILOT_DONE').write_text(dt.datetime.now(dt.timezone.utc).isoformat())
