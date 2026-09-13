"""Fresh offline admission evaluation with a frozen checkpoint."""
from collections import defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess

from row_oracle import ROOT, BIN, tokenize

LANE = ROOT/'memra/research/adaptive-draft-joint-20260913'
OLD = LANE/'row-oracle-receipts'
OUT = ROOT/'raw/fresh-admission'
CHECKPOINT = ROOT/'checkpoints/admission-model.json'
PROMPTS = ROOT/'checkpoints/fresh-admission-prompts'
DONE = ROOT/'FRESH_ADMISSION_DONE'
EXPECTED_PROMPTS = 24
EXTRA_CONFIG = {}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import fcntl
    OUT.mkdir(exist_ok=False)
    checkpoint = CHECKPOINT
    checkpoint_hash = sha(checkpoint)
    model = json.loads(checkpoint.read_text())
    ranks = OLD/'checkpoints/row-oracle/base.txt'
    sidecar = Path(str(ranks)+'.learned')
    frozen_hash = sha(sidecar)
    prompts = sorted(PROMPTS.glob('*.txt'))
    encoded = {p.name: tokenize(p) for p in prompts}
    (OUT/'manifest.json').write_text(json.dumps({'source_commit': (ROOT/'SOURCE_COMMIT').read_text().strip(),
        'binary_sha256': sha(BIN/'gemma-gate'), 'checkpoint_sha256':checkpoint_hash, 'selected_source':model['source'],
        'prompts': {p.name: sha(p) for p in prompts}}, indent=2))
    assert len(prompts) == EXPECTED_PROMPTS
    known = {}
    def run(prompt, mode, repeat, phase):
        name = f'{prompt.stem}-{phase}-r{repeat}-{mode}'
        trace = OUT/(name+'.jsonl')
        config = {'MEMRA_SPEC': '4', 'MEMRA_DRAFT': str(ROOT/'models/gemma12-draft.gguf'), 'MEMRA_NGEN': '128',
            'MEMRA_GATE_DUMP_TOKENS': '1', 'MEMRA_SPEC_STATS': '1', 'MEMRA_SPEC_ADAPT': '0', 'MEMRA_SPEC_CAPMAX': '4',
            'MEMRA_SPEC_PMIN': '0', 'MEMRA_SPEC_PMIN_INROUND': '0', 'MEMRA_GEMMA_DRAFT_GRAPH': '0', 'MEMRA_GEMMA_ROUND_GRAPH': '0',
            'MEMRA_GEMMA_DRAFT_RANKS': str(ranks), 'MEMRA_GEMMA_TRIM_ADAPT': '512', 'MEMRA_GEMMA_TRIM_FREEZE': '1'}
        if mode == 'full': config['MEMRA_GEMMA_ROW_PROBE'] = str(trace)
        config.update(EXTRA_CONFIG)
        if mode == 'bounded':
            config['MEMRA_GEMMA_CANDIDATE_PROBE'] = str(trace)
            config['MEMRA_GEMMA_CANDIDATE_SUFFIX'] = '1' if model['source']=='suffix' else '0'
            config['MEMRA_GEMMA_DRAFT_TRACE'] = str(OUT/(name+'-draft.jsonl'))
        argv = [str(BIN/'gemma-gate'), str(ROOT/'models/gemma12.gguf')]+list(map(str, encoded[prompt.name]))
        env = {k:v for k,v in os.environ.items() if not k.startswith('MEMRA_')}
        env.update(config)
        if not (OUT/'admission.json').exists():
            red_env = dict(env)
            red_env['MEMRA_GEMMA_ROW_PROBE'] = str(OUT/'refused-full.jsonl')
            red_env['MEMRA_GEMMA_CANDIDATE_PROBE'] = str(OUT/'refused-bounded.jsonl')
            with (OUT/'admission.log').open('xb') as f:
                red = subprocess.run(argv, env=red_env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            assert red.returncode != 0 and 'candidate and full row probes must run separately' in (OUT/'admission.log').read_text()
            (OUT/'admission.json').write_text(json.dumps({'exit_code':red.returncode, 'raw_sha256':sha(OUT/'admission.log'), 'argv':argv, 'config':{k:v for k,v in red_env.items() if k.startswith('MEMRA_')}, 'status':'expected_refusal'}))
        if EXTRA_CONFIG and not (OUT/'cadence-refusal.json').exists():
            red_env = dict(env); red_env['MEMRA_GEMMA_PROBE_EVERY'] = '0'
            with (OUT/'cadence-refusal.log').open('xb') as f:
                red = subprocess.run(argv, env=red_env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            assert red.returncode != 0 and 'probe cadence must be positive' in (OUT/'cadence-refusal.log').read_text()
            (OUT/'cadence-refusal.json').write_text(json.dumps({'exit_code':red.returncode,
                'raw_sha256':sha(OUT/'cadence-refusal.log'), 'status':'expected_refusal'}))
        log = OUT/(name+'.log')
        with log.open('xb') as f:
            try: code = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300).returncode
            except subprocess.TimeoutExpired: code = -999
        row = {'id': name, 'prompt': prompt.name, 'mode': mode, 'phase': phase, 'repeat': repeat, 'argv': argv, 'config': config,
            'exit_code': code, 'raw_sha256': sha(log), 'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        try:
            assert code == 0
            text = log.read_text()
            a = json.loads(re.search(r'plain tokens: (\[[^\n]+\])', text)[1])
            b = json.loads(re.search(r'spec tokens: (\[[^\n]+\])', text)[1])
            assert a == b and len(b) == 128 and known.setdefault(prompt.name, b) == b
            assert sha(sidecar) == frozen_hash
            assert sha(checkpoint) == checkpoint_hash
            row.update(tokens_sha256=hashlib.sha256(json.dumps(b).encode()).hexdigest(), request_seconds=float(re.search(r'request_seconds=([\d.]+)', text)[1]))
            if mode != 'off':
                observations = [json.loads(x) for x in trace.read_text().splitlines()]
                assert observations
                row.update(trace_sha256=sha(trace), states=len(observations))
                if mode == 'bounded': row['draft_trace_sha256'] = sha(OUT/(name+'-draft.jsonl'))
            row['status'] = 'pass'
        except Exception as error: row.update(status='failed', error=str(error))
        with (OUT/'runs.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ['argv','config']}), flush=True)
        assert row['status'] == 'pass', name

    with open('/tmp/memra-5090.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        with (OUT/'telemetry.csv').open('w') as f:
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu', '--format=csv', '-lms', '250'], stdout=f)
            try:
                for i, prompt in enumerate(prompts):
                    for mode in (['full','bounded'] if i%2 == 0 else ['bounded','full']): run(prompt, mode, 0, 'collection')
            finally:
                telemetry.terminate()
                telemetry.wait(timeout=10)
    DONE.write_text(f'{2*len(prompts)} runs completed')

if __name__ == '__main__': main()
