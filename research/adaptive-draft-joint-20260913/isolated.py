"""Three separately sealed component experiments; deliberately no combined arm."""
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path('/workspace/adaptive-draft')
BIN = ROOT/'memra/target/release'
OUT = ROOT/'raw/isolated'
CHECK = ROOT/'checkpoints/isolated'
SEED = 20260913


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(rows):
    pairs = {}
    for r in rows:
        if r['phase'] == 'timed':
            pairs.setdefault((r['prompt'], r['repeat']), {})[r['arm']] = r
    groups = {}
    details = []
    for (prompt, repeat), arms in pairs.items():
        a, b = arms['A'], arms['B']
        assert a['tokens_sha256'] == b['tokens_sha256']
        ratio = b['request_seconds']/a['request_seconds']
        groups.setdefault(prompt, []).append(math.log(ratio))
        details.append({'prompt': prompt, 'repeat': repeat, 'request_ratio_b_over_a': ratio,
                        'decode_ratio_b_over_a': b['decode_seconds']/a['decode_seconds'],
                        'a_request_seconds': a['request_seconds'], 'b_request_seconds': b['request_seconds']})
    values = [statistics.mean(v) for v in groups.values()]
    rng = random.Random(SEED)
    boots = sorted(math.exp(statistics.mean(rng.choices(values, k=len(values)))) for _ in range(10000))
    return {'independent_prompt_groups': len(values), 'paired_repeats': len(pairs),
            'request_ratio_b_over_a': math.exp(statistics.mean(values)),
            'bootstrap_95_percentile': [boots[250], boots[9749]],
            'groups_faster_b': sum(x < 0 for x in values),
            'group_log_ratios': groups, 'pairs': details,
            'scope': 'Existing component baseline; greedy code-review mechanism study, not serving or novelty evidence.'}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    CHECK.mkdir(parents=True, exist_ok=False)
    assert (ROOT/'models/VERIFIED').exists()
    ready = json.loads((ROOT/'PILOT_READY.json').read_text())
    assert all(json.loads(x)['status'] == 'pass' for x in (ROOT/ready['runs']).read_text().splitlines())
    prompts = sorted((ROOT/'code-prompts/heldout').glob('*.txt'))[6:14]
    assert len(prompts) == 8
    ranks = ROOT/'checkpoints/gemma12-ranks4096.gguf.txt'
    manifest = {'seed': SEED, 'source_commit': (ROOT/'SOURCE_COMMIT').read_text().strip(),
                'binaries': {n: sha(BIN/n) for n in ['gemma-gate', 'draft_prompt']},
                'prompts': {p.name: sha(p) for p in prompts}, 'base_ranks_sha256': sha(ranks),
                'protocol_sha256': sha(Path(__file__).with_name('ISOLATED.md'))}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    tokens = {}
    for p in prompts:
        text = subprocess.check_output([str(BIN/'draft_prompt'), str(ROOT/'models/gemma12.gguf'), str(p)], text=True)
        tokens[p.stem] = json.loads(text.splitlines()[-1])
    known_outputs = {}

    def execute(component, prompt, repeat, arm, phase):
        name = f'{component}-{phase}-{prompt.stem}-r{repeat}-{arm}'
        rankcopy = CHECK/f'{name}.txt'
        shutil.copyfile(ranks, rankcopy)
        config = {'MEMRA_SPEC': '4', 'MEMRA_DRAFT': str(ROOT/'models/gemma12-draft.gguf'),
                  'MEMRA_NGEN': '128', 'MEMRA_GATE_DUMP_TOKENS': '1', 'MEMRA_SPEC_STATS': '1',
                  'MEMRA_SPEC_ADAPT': '0', 'MEMRA_SPEC_ADAPT_FLOOR': '1', 'MEMRA_SPEC_CAPMAX': '4',
                  'MEMRA_SPEC_PMIN': '0', 'MEMRA_SPEC_PMIN_INROUND': '0',
                  'MEMRA_GEMMA_DRAFT_GRAPH': '0', 'MEMRA_GEMMA_ROUND_GRAPH': '0',
                  'MEMRA_GEMMA_DRAFT_RANKS': str(rankcopy), 'MEMRA_GEMMA_TRIM_ADAPT': '512',
                  'MEMRA_GEMMA_TRIM_FREEZE': '1'}
        if arm == 'B':
            if component == 'head':
                config['MEMRA_GEMMA_TRIM_FREEZE'] = '0'
            elif component == 'confidence':
                config['MEMRA_SPEC_PMIN_INROUND'] = '0.7'
            elif component == 'depth':
                config['MEMRA_SPEC_ADAPT'] = '1'
        trace = OUT/f'{name}.trace.jsonl'
        if phase == 'diagnostic':
            config['MEMRA_GEMMA_DRAFT_TRACE'] = str(trace)
        env = {k: v for k, v in os.environ.items() if not k.startswith('MEMRA_')}
        env.update(config)
        argv = [str(BIN/'gemma-gate'), str(ROOT/'models/gemma12.gguf')] + list(map(str, tokens[prompt.stem]))
        start = time.time()
        log = OUT/f'{name}.log'
        with log.open('xb') as f:
            try:
                result = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = -999
        row = {'id': name, 'component': component, 'phase': phase, 'prompt': prompt.stem,
               'repeat': repeat, 'arm': arm, 'config': config, 'argv': argv,
               'utc': dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
               'exit_code': code, 'wall_seconds': time.time()-start, 'raw_sha256': sha(log)}
        try:
            text = log.read_text()
            assert code == 0, f'exit {code}'
            plain = json.loads(re.search(r'plain tokens: (\[[^\n]+\])', text)[1])
            spec = json.loads(re.search(r'spec tokens: (\[[^\n]+\])', text)[1])
            assert plain == spec and len(spec) == 128, 'Token mismatch or count'
            token_sha = hashlib.sha256(json.dumps(spec).encode()).hexdigest()
            assert known_outputs.setdefault(prompt.stem, token_sha) == token_sha, 'Cross-arm/repeat mismatch'
            row.update(tokens_sha256=token_sha, emitted=len(spec))
            timing = re.search(r'spec timing: request_seconds=([\d.]+) prime_seconds=([\d.]+) decode_seconds=([\d.]+) emitted=(\d+)', text)
            assert timing
            row.update(zip(['request_seconds', 'prime_seconds', 'decode_seconds'], map(float, timing.groups()[:3])))
            assert row['request_seconds'] > 0 and row['decode_seconds'] > 0
            counts = re.search(r'\[gemma-spec\] rounds=(\d+) drafted=(\d+) accepted=(\d+)', text)
            assert counts
            row.update(zip(['rounds', 'drafted', 'accepted'], map(int, counts.groups())))
            learned = re.search(r'\[trim-adapt\] (\d+)/(\d+) spare slots learned', text)
            row['learned_rows'] = int(learned[1])
            assert 'FR head trim: 4096 rows + 512 adaptive' in text
            if config['MEMRA_GEMMA_TRIM_FREEZE'] == '1':
                assert row['learned_rows'] == 0 and not Path(str(rankcopy)+'.learned').exists()
            elif phase == 'diagnostic':
                assert row['learned_rows'] > 0, 'Learning ablation is vacuous'
            if phase == 'diagnostic':
                traces = [json.loads(x) for x in trace.read_text().splitlines()]
                assert traces and all(x['head_rows'] == 4608 for x in traces)
                if component in ['confidence', 'depth'] and arm == 'B':
                    assert any(x['drafted'] < 4 for x in traces), 'Stopping ablation is vacuous'
                row['trace_sha256'] = sha(trace)
            row['status'] = 'pass'
        except Exception as error:
            row.update(status='failed', error=str(error))
        with (OUT/'runs.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps({k: v for k, v in row.items() if k not in ['argv', 'config']}), flush=True)
        if row['status'] != 'pass':
            raise RuntimeError(f'{name}: {row.get("error")}')
        return row

    with open('/tmp/memra-5090.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        with (OUT/'telemetry.csv').open('w') as tf:
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu', '--format=csv', '-lms', '250'], stdout=tf)
            try:
                for component in ['head', 'confidence', 'depth']:
                    rows = []
                    for arm in ['A', 'B']:
                        rows.append(execute(component, prompts[0], 0, arm, 'diagnostic'))
                    for arm in ['A', 'B']:
                        rows.append(execute(component, prompts[0], 0, arm, 'warmup'))
                    for repeat in range(6):
                        order = list(prompts)
                        random.Random(SEED+repeat).shuffle(order)
                        for i, prompt in enumerate(order):
                            # Assign by stable prompt index so every prompt reverses order.
                            arms = ['A', 'B'] if (prompts.index(prompt)+repeat) % 2 == 0 else ['B', 'A']
                            for arm in arms:
                                rows.append(execute(component, prompt, repeat, arm, 'timed'))
                    result = summarize(rows)
                    (OUT/f'{component}-summary.json').write_text(json.dumps(result, indent=2))
                    files = {p.name: sha(p) for p in OUT.glob(f'{component}-*') if p.is_file()}
                    (OUT/f'{component}-seal.json').write_text(json.dumps(files, indent=2))
                    (ROOT/f'{component.upper()}_DONE').write_text(dt.datetime.now(dt.timezone.utc).isoformat())
                    print(f'COMPONENT_COMPLETE {component} '+json.dumps(result), flush=True)
                    # Give the local lifecycle guard a stable boundary to bank this rung.
                    deadline = time.time()+240
                    while not (ROOT/f'{component.upper()}_BANKED').exists():
                        if time.time() > deadline:
                            raise RuntimeError(f'Off-host banking acknowledgement missing for {component}')
                        time.sleep(2)
            finally:
                telemetry.terminate()
                telemetry.wait(timeout=10)
    (ROOT/'ISOLATED_DONE').write_text(dt.datetime.now(dt.timezone.utc).isoformat())


if __name__ == '__main__':
    main()
