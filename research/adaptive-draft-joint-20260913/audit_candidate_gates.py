"""Independent physical-replay and causal-discovery receipt audit."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def raw_runs(root):
    runs = [json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()]
    known = {}
    for r in runs:
        assert r['status'] == 'pass' and r['exit_code'] == 0
        path = root/(r['id']+'.log')
        assert sha(path) == r['raw_sha256']
        text = path.read_text()
        a = json.loads(re.search(r'plain tokens: (\[[^\n]+\])', text)[1])
        b = json.loads(re.search(r'spec tokens: (\[[^\n]+\])', text)[1])
        assert a == b and len(b) == 128
        assert hashlib.sha256(json.dumps(b).encode()).hexdigest() == r['tokens_sha256']
        prompt = r.get('prompt', r['id'].removesuffix('-on').removesuffix('-off'))
        assert known.setdefault(prompt, b) == b
        r['_tokens'] = b
        assert r['config']['MEMRA_GEMMA_TRIM_FREEZE'] == '1'
        assert r['config']['MEMRA_SPEC'] == '4'
        assert all(r['config'][k] == '0' for k in ['MEMRA_SPEC_ADAPT','MEMRA_SPEC_PMIN','MEMRA_SPEC_PMIN_INROUND','MEMRA_GEMMA_DRAFT_GRAPH','MEMRA_GEMMA_ROUND_GRAPH'])
        if 'trace_sha256' in r:
            trace = root/(r['id']+'.jsonl')
            assert sha(trace) == r['trace_sha256']
            r['_states'] = [json.loads(x) for x in trace.read_text().splitlines()]
    assert len(known) == 48
    return runs

def physical(bank):
    root = bank/'raw/physical-replay'
    runs = raw_runs(root)
    assert len(runs) == 96
    red = json.loads((root/'admission.json').read_text())
    assert red['exit_code'] != 0 and sha(root/'admission.log') == red['raw_sha256']
    assert 'physical row replay requires MEMRA_GEMMA_ROW_PROBE' in (root/'admission.log').read_text()
    swaps = [s for r in runs if r['replay'] for s in r['_states'] if s.get('kind') == 'physical_swap']
    ordinary = [s for r in runs if r['replay'] for s in r['_states'] if s['schema'] == 1]
    assert all(s['matched'] or s['ambiguous'] for s in swaps)
    counts = {'runs': len(runs), 'states': len(ordinary), 'swaps': len(swaps),
        'ambiguous': sum(s['ambiguous'] for s in swaps), 'max_score_error': max(s['max_abs_error'] for s in swaps),
        'positive': sum(s['delta'] > 0 for s in swaps), 'negative': sum(s['delta'] < 0 for s in swaps),
        'neutral': sum(s['delta'] == 0 for s in swaps), 'nonambiguous_mismatches': sum(not s['matched'] and not s['ambiguous'] for s in swaps)}
    assert all(counts[k] > 0 for k in ['positive','negative','neutral','ambiguous'])
    return counts

def bounded(bank, original):
    root = bank/'raw/bounded-candidates'
    runs = raw_runs(root)
    assert len(runs) == 192
    core = list(map(int, (original/'checkpoints/row-oracle/base.txt').read_text().split()))
    tail = list(map(int, (original/'checkpoints/row-oracle/base.txt.learned').read_text().split()))
    present = set(core+tail)
    by_prompt = defaultdict(dict)
    for r in runs:
        if r['phase'] == 'collection': by_prompt[r['prompt']][r['mode']] = r
    groups = defaultdict(lambda: defaultdict(int))
    max_error = 0.0
    compared = 0
    for prompt, pair in by_prompt.items():
        full = {(s['round'],s['position']):s for s in pair['full']['_states']}
        candidate = pair['bounded']
        argv_prompt = list(map(int, candidate['argv'][2:]))
        tokens = candidate['_tokens']
        g = groups[prompt.split('-')[0]+'-'+prompt.split('-')[1]]
        assert len(full) == len(candidate['_states'])
        for s in candidate['_states']:
            f = full[s['round'],s['position']]
            assert all(s[k] == f[k] for k in ['context','proposal','target'])
            recent = deque()
            n = s['context']-len(argv_prompt)
            assert 0 <= n < 128
            for token in argv_prompt + tokens[1:n+1]:
                if token in present: continue
                if token in recent: recent.remove(token)
                recent.append(token)
                if len(recent) > 256: recent.popleft()
            expected = list(reversed(recent))[:24]
            assert len(expected) == s['history_rows']
            rng = 0x9e3779b97f4a7c15 ^ (s['round']-1)
            mask = (1<<64)-1
            while len(expected) < 32:
                rng ^= (rng<<13)&mask; rng ^= rng>>7; rng ^= (rng<<17)&mask
                token = rng % s['vocab']
                if token not in present and token not in expected: expected.append(token)
            scores = dict(s['candidates'])
            assert expected == list(scores), ('candidate provenance mismatch', prompt, s['round'])
            g['states'] += 1
            g['missing_targets'] += not f['target_in_head']
            repair = f['addition_rescue'] or f['removal_rescue'] or f['swap_rescue']
            g['oracle_repairable'] += repair
            g['repairable_target_discovered'] += repair and s['target'] in scores
            g['missing_target_discovered'] += not f['target_in_head'] and s['target'] in scores
            reference = dict(f['outside_top16'])
            reference[f['target']] = f['target_full_score']
            for token in scores.keys() & reference.keys():
                error = abs(scores[token]-reference[token])
                max_error = max(max_error,error); compared += 1
                assert error <= f['tolerance'], (prompt, token, error)
    pairs = defaultdict(dict)
    for r in runs:
        if r['phase'] == 'cost': pairs[r['prompt'],r['repeat']][r['mode']] = r
    ratios = defaultdict(list)
    for (prompt, repeat), pair in pairs.items():
        assert set(pair) == {'off','bounded'}
        ratios[prompt].append(math.log(pair['bounded']['request_seconds']/pair['off']['request_seconds']))
    assert len(pairs) == 48 and len(ratios) == 8 and all(len(v)==6 for v in ratios.values())
    means = [statistics.mean(v) for v in ratios.values()]
    rng = random.Random(20260913)
    boot = sorted(math.exp(statistics.mean(rng.choices(means,k=8))) for _ in range(10000))
    return {'runs':len(runs), 'discovery':groups, 'candidate_scores_compared':compared, 'maximum_compared_score_error':max_error,
        'cost_ratio':math.exp(statistics.mean(means)), 'bootstrap95':[boot[250],boot[9749]],
        'control_seconds':statistics.mean(p['off']['request_seconds'] for p in pairs.values()),
        'bounded_seconds':statistics.mean(p['bounded']['request_seconds'] for p in pairs.values()),
        'scope':'Candidate discovery diagnostic, not a learned-policy or serving speedup result.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('bank', type=Path)
    p.add_argument('--phase', choices=['physical','bounded'], required=True)
    p.add_argument('--original', type=Path, default=Path(__file__).with_name('row-oracle-receipts'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = physical(a.bank) if a.phase == 'physical' else bounded(a.bank,a.original)
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
