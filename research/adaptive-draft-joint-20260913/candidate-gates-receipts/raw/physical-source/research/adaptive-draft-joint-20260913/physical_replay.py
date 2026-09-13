"""Numerical qualification only; old oracle inputs are not a new policy test."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from row_oracle import ROOT, BIN, tokenize

LANE = ROOT/'memra/research/adaptive-draft-joint-20260913'
OLD = LANE/'row-oracle-receipts'
OUT = ROOT/'raw/physical-replay'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import fcntl
    OUT.mkdir(exist_ok=False)
    ranks = OLD/'checkpoints/row-oracle/base.txt'
    sidecar = Path(str(ranks)+'.learned')
    tail_hash = sha(sidecar)
    source = (ROOT/'SOURCE_COMMIT').read_text().strip()
    (OUT/'manifest.json').write_text(json.dumps({'source_commit': source, 'binary_sha256': sha(BIN/'gemma-gate'), 'kind': 'physical-replacement-qualification-not-policy-test'}, indent=2))
    known = {}
    total = {x: 0 for x in ['states', 'swaps', 'ambiguous', 'positive', 'negative', 'neutral', 'mismatches']}
    with open('/tmp/memra-5090.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        with (OUT/'telemetry.csv').open('w') as f:
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu', '--format=csv', '-lms', '250'], stdout=f)
            try:
                for i, prompt in enumerate(sorted((OLD/'raw/row-oracle/prompts').glob('*.txt'))):
                    ids = tokenize(prompt)
                    for on in ([False, True] if i % 2 == 0 else [True, False]):
                        name = prompt.stem+('-on' if on else '-off')
                        trace = OUT/(name+'.jsonl')
                        config = {'MEMRA_SPEC': '4', 'MEMRA_DRAFT': str(ROOT/'models/gemma12-draft.gguf'), 'MEMRA_NGEN': '128',
                            'MEMRA_GATE_DUMP_TOKENS': '1', 'MEMRA_SPEC_STATS': '1', 'MEMRA_SPEC_ADAPT': '0', 'MEMRA_SPEC_CAPMAX': '4',
                            'MEMRA_SPEC_PMIN': '0', 'MEMRA_SPEC_PMIN_INROUND': '0', 'MEMRA_GEMMA_DRAFT_GRAPH': '0', 'MEMRA_GEMMA_ROUND_GRAPH': '0',
                            'MEMRA_GEMMA_DRAFT_RANKS': str(ranks), 'MEMRA_GEMMA_TRIM_ADAPT': '512', 'MEMRA_GEMMA_TRIM_FREEZE': '1',
                            'MEMRA_GEMMA_ROW_PROBE': str(trace), 'MEMRA_GEMMA_ROW_REPLAY': '1' if on else '0'}
                        argv = [str(BIN/'gemma-gate'), str(ROOT/'models/gemma12.gguf')]+list(map(str, ids))
                        env = {k:v for k,v in os.environ.items() if not k.startswith('MEMRA_')}
                        env.update(config)
                        if i == 0 and not (OUT/'admission.log').exists():
                            red_env = dict(env)
                            red_env.pop('MEMRA_GEMMA_ROW_PROBE')
                            red_env['MEMRA_GEMMA_ROW_REPLAY'] = '1'
                            with (OUT/'admission.log').open('xb') as f:
                                red = subprocess.run(argv, env=red_env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
                            assert red.returncode != 0 and 'physical row replay requires MEMRA_GEMMA_ROW_PROBE' in (OUT/'admission.log').read_text()
                            (OUT/'admission.json').write_text(json.dumps({'exit_code': red.returncode, 'raw_sha256': sha(OUT/'admission.log'), 'argv': argv, 'config': {k:v for k,v in red_env.items() if k.startswith('MEMRA_')}, 'status': 'expected_refusal'}))
                        row = {'id': name, 'replay': on, 'config': config, 'argv': argv, 'prompt_sha256': sha(prompt), 'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
                        log = OUT/(name+'.log')
                        with log.open('xb') as f:
                            try: code = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300).returncode
                            except subprocess.TimeoutExpired: code = -999
                        row.update(exit_code=code, raw_sha256=sha(log))
                        try:
                            assert code == 0
                            text = log.read_text()
                            a = json.loads(re.search(r'plain tokens: (\[[^\n]+\])', text)[1])
                            b = json.loads(re.search(r'spec tokens: (\[[^\n]+\])', text)[1])
                            assert a == b and len(b) == 128
                            assert known.setdefault(prompt.name, b) == b
                            assert sha(sidecar) == tail_hash
                            records = [json.loads(x) for x in trace.read_text().splitlines()]
                            swaps = [x for x in records if x.get('kind') == 'physical_swap']
                            states = [x for x in records if x.get('schema') == 1]
                            assert states and all(s['overlap_argmax_match'] for s in states)
                            assert bool(swaps) == on
                            if on:
                                total['states'] += len(states)
                                total['swaps'] += len(swaps)
                                for x in swaps:
                                    total['ambiguous'] += x['ambiguous']
                                    total['mismatches'] += not x['matched'] and not x['ambiguous']
                                    total['positive' if x['delta'] > 0 else 'negative' if x['delta'] < 0 else 'neutral'] += 1
                                assert all(x['matched'] or x['ambiguous'] for x in swaps)
                            row.update(status='pass', states=len(states), swaps=len(swaps), trace_sha256=sha(trace), tokens_sha256=hashlib.sha256(json.dumps(b).encode()).hexdigest())
                        except Exception as error:
                            row.update(status='failed', error=str(error))
                        with (OUT/'runs.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                        print(json.dumps({k:v for k,v in row.items() if k not in ['config','argv']}), flush=True)
                        assert row['status'] == 'pass', name
            finally:
                telemetry.terminate()
                telemetry.wait(timeout=10)
    assert len(known) == 48 and total['mismatches'] == 0
    assert all(total[x] > 0 for x in ['positive', 'negative', 'neutral', 'ambiguous'])
    (OUT/'summary.json').write_text(json.dumps(total, indent=2))
    (ROOT/'PHYSICAL_DONE').write_text(json.dumps(total))

if __name__ == '__main__': main()
