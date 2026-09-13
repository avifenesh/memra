"""Frozen saturated-head one-swap diagnostic. No learned/joint serving policy."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

ROOT = Path('/workspace/adaptive-draft')
BIN = ROOT/'memra/target/release'
OUT = ROOT/'raw/row-oracle'
SEED = 20260913

PROSE = [
    'Write a scene in which a lighthouse keeper discovers that the evening fog contains whispered directions. Use concrete sensory details and no explanation of the mystery.',
    'Explain to a museum visitor how a curator might distinguish a replica from an original ceramic bowl. Describe uncertainty and record keeping without inventing a real provenance.',
    'Turn these notes into a polite supplier email: delivery due Tuesday; arrived Friday; three cartons wet; photographs available; ask for a replacement timetable; avoid accusations.',
    'Summarize this fictional town meeting: residents support a new library, oppose losing the orchard, and ask for weekend opening. The council postpones the location vote but approves accessibility funding.',
    'Describe the work of a bookbinder repairing a damaged atlas, from examining the binding to preparing a protective case. Keep the explanation accessible.',
    'Create a short dialogue between two archaeologists who disagree about whether markings on a stone were made by people or erosion. Let each propose a test.',
    'Rewrite this notice warmly and clearly: All equipment borrowers are obligated to return borrowed articles prior to the expiration of the designated lending interval.',
    'Write an original explanation of why a community garden needs a shared watering schedule. Include a practical example of coordinating volunteers.',
    'Write a scene in which a night train stops at a station that does not appear on the conductor\'s map. Focus on a passenger observing small details.',
    'Explain to a gallery visitor how the lighting around a painting can change its appearance. Distinguish the object from the viewing conditions.',
    'Turn these notes into a polite venue email: booked room for Thursday; confirmation says Friday; 18 participants; projector required; ask to correct the date and confirm the equipment.',
    'Summarize this fictional committee discussion: a footbridge needs repairs, members disagree about closing it during work, and a temporary shuttle is proposed. A cost estimate is requested before voting.',
    'Describe the work of a conservator cleaning a tarnished brass instrument. Include documentation and caution about preserving original material.',
    'Create a dialogue between two botanists discussing why a greenhouse plant is wilting. Let each distinguish observations from possible explanations.',
    'Rewrite this notice warmly and clearly: Visitors intending to utilize the reading facilities must refrain from conducting audible telephone conversations within the premises.',
    'Explain why a neighborhood tool library needs a sign-out record. Give a simple example involving a drill and two neighbors.',
    'Write a scene in which a cartographer notices a tiny island moving between successive editions of the same map. Use observable details instead of explaining the cause.',
    'Explain to a theatre visitor how stage acoustics affect dialogue clarity. Distinguish loudness from intelligibility in familiar language.',
    'Turn these notes into a polite workshop email: reserved six places; received four tickets; order number 782; event next Saturday; request the missing tickets and written confirmation.',
    'Summarize this fictional association meeting: volunteers propose restoring a fountain, some prefer new benches, and maintenance capacity is limited. Members agree to gather accessibility feedback before deciding.',
    'Describe how a textile restorer might examine a faded embroidered banner before deciding what to repair. Emphasize documentation and reversible choices.',
    'Create a dialogue between two astronomers discussing an unexpected bright spot in a photograph. Include an instrument explanation and a way to check it.',
    'Rewrite this notice warmly and clearly: Persons wishing to obtain replacement membership credentials shall present documentary identification at the administrative reception desk.',
    'Explain why a small theatre needs a checklist for lending costumes. Include a concrete example about recording an alteration and arranging its reversal.',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(path):
    text = subprocess.check_output([str(BIN/'draft_prompt'), str(ROOT/'models/gemma12.gguf'), str(path)], text=True)
    return json.loads(text.splitlines()[-1])


def main():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=False)
    check = ROOT/'checkpoints/row-oracle'
    check.mkdir(parents=True, exist_ok=False)
    base = ROOT/'memra/research/adaptive-draft-joint-20260913/checkpoints/gemma12-ranks4096.gguf.txt'
    ranks = check/'base.txt'
    shutil.copyfile(base, ranks)
    core = list(map(int, ranks.read_text().split()))
    present = set(core)
    tail = []
    seed_sources = []
    for prompt in sorted((ROOT/'code-prompts/train').glob('*.txt')):
        ids = tokenize(prompt)
        seed_sources.append({'prompt': prompt.name, 'sha256': sha(prompt)})
        for token in ids:
            if token not in present and len(tail) < 512:
                tail.append(token)
                present.add(token)
        if len(tail) == 512:
            break
    assert len(tail) == 512, f'Only {len(tail)} training-only spare tokens; refuse unsaturated fixture'
    sidecar = Path(str(ranks)+'.learned')
    sidecar.write_text(''.join(str(i)+'\n' for i in tail))
    (check/'seed.json').write_text(json.dumps({'sources': seed_sources, 'core_sha256': sha(ranks), 'tail_sha256': sha(sidecar), 'core_rows': len(core), 'tail_rows': len(tail)}, indent=2))
    prompts = []
    pd = OUT/'prompts'
    pd.mkdir()
    for s, split in enumerate(['train', 'calibration', 'heldout']):
        code = sorted((ROOT/f'code-prompts/{split}').glob('*.txt'))
        code = code[14:22] if split == 'heldout' else code[:8]
        for p in code:
            q = pd/f'{split}-code-{p.stem}.txt'
            shutil.copyfile(p, q)
            prompts.append((split, 'code', q))
        for j, prompt in enumerate(PROSE[s*8:s*8+8]):
            q = pd/f'{split}-prose-{j}.txt'
            q.write_text(prompt+'\n')
            prompts.append((split, 'prose', q))
    encoded = {p.name: tokenize(p) for _, _, p in prompts}
    (OUT/'manifest.json').write_text(json.dumps({'seed': SEED, 'source_commit': (ROOT/'SOURCE_COMMIT').read_text().strip(),
        'binary_sha256': sha(BIN/'gemma-gate'), 'prompts': {p.name: sha(p) for _, _, p in prompts},
        'policy': 'frozen head, K4, zero confidence; complete probe every16 rounds; oracle only'}, indent=2))
    known = {}

    def run(item, probe, repeat, phase, red=False):
        split, domain, prompt = item
        name = f'{prompt.stem}-{phase}-r{repeat}-'+('on' if probe else 'off')+('-red' if red else '')
        trace = OUT/f'{name}.jsonl'
        config = {'MEMRA_SPEC': '4', 'MEMRA_DRAFT': str(ROOT/'models/gemma12-draft.gguf'), 'MEMRA_NGEN': '128',
            'MEMRA_GATE_DUMP_TOKENS': '1', 'MEMRA_SPEC_STATS': '1', 'MEMRA_SPEC_ADAPT': '0',
            'MEMRA_SPEC_CAPMAX': '4', 'MEMRA_SPEC_PMIN': '0', 'MEMRA_SPEC_PMIN_INROUND': '0',
            'MEMRA_GEMMA_DRAFT_GRAPH': '0', 'MEMRA_GEMMA_ROUND_GRAPH': '0',
            'MEMRA_GEMMA_DRAFT_RANKS': str(ranks), 'MEMRA_GEMMA_TRIM_ADAPT': '512', 'MEMRA_GEMMA_TRIM_FREEZE': '1'}
        if probe:
            config['MEMRA_GEMMA_ROW_PROBE'] = str(trace)
        if red:
            config['MEMRA_GEMMA_TRIM_FREEZE'] = '0'
        env = {k: v for k, v in os.environ.items() if not k.startswith('MEMRA_')}
        env.update(config)
        argv = [str(BIN/'gemma-gate'), str(ROOT/'models/gemma12.gguf')] + list(map(str, encoded[prompt.name]))
        start = time.time()
        log = OUT/f'{name}.log'
        with log.open('xb') as f:
            try:
                code = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300).returncode
            except subprocess.TimeoutExpired:
                code = -999
        row = {'id': name, 'prompt': prompt.name, 'split': split, 'domain': domain, 'probe': probe, 'repeat': repeat,
            'phase': phase, 'config': config, 'argv': argv, 'utc': dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
            'wall_seconds': time.time()-start, 'exit_code': code, 'raw_sha256': sha(log)}
        try:
            text = log.read_text()
            if red:
                assert code != 0 and 'row probe requires MEMRA_GEMMA_TRIM_FREEZE=1' in text
                row['status'] = 'expected_refusal'
            else:
                assert code == 0, f'exit {code}'
                a = json.loads(re.search(r'plain tokens: (\[[^\n]+\])', text)[1])
                b = json.loads(re.search(r'spec tokens: (\[[^\n]+\])', text)[1])
                assert a == b and len(b) == 128
                output_hash = hashlib.sha256(json.dumps(b).encode()).hexdigest()
                assert known.setdefault(prompt.name, output_hash) == output_hash
                row.update(tokens_sha256=output_hash, emitted=len(b), request_seconds=float(re.search(r'request_seconds=([\d.]+)', text)[1]))
                assert sha(sidecar) == json.loads((check/'seed.json').read_text())['tail_sha256']
                if probe:
                    states = [json.loads(x) for x in trace.read_text().splitlines()]
                    assert states and all(x['head_rows'] == 4608 for x in states)
                    row.update(probe_states=len(states), trace_sha256=sha(trace), numerical_mismatches=sum(not x['overlap_argmax_match'] for x in states))
                row['status'] = 'pass'
        except Exception as error:
            row.update(status='failed', error=str(error))
        with (OUT/'runs.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps({k: v for k, v in row.items() if k not in ['config', 'argv']}), flush=True)
        assert row['status'] in ['pass', 'expected_refusal'], name

    with open('/tmp/memra-5090.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        with (OUT/'telemetry.csv').open('w') as f:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu', '--format=csv', '-lms', '250'], stdout=f)
            try:
                run(prompts[0], True, 0, 'admission', red=True)
                for i, item in enumerate(prompts):
                    for on in ([False, True] if i % 2 == 0 else [True, False]):
                        run(item, on, 0, 'collection')
                cost = [x for x in prompts if x[0] == 'calibration' and x[1] == 'code']
                for repeat in range(6):
                    order = list(cost)
                    random.Random(SEED+repeat).shuffle(order)
                    for item in order:
                        for on in ([False, True] if (cost.index(item)+repeat) % 2 == 0 else [True, False]):
                            run(item, on, repeat, 'cost')
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
    (ROOT/'PROBE_DONE').write_text(dt.datetime.now(dt.timezone.utc).isoformat())


if __name__ == '__main__':
    main()
