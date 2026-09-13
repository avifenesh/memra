"""Audit candidate discovery against the exact earlier verifier observations."""
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import random
import statistics
from audit_candidate_gates import raw_runs, sha

def analyze(bank, original):
    root=bank/'raw/candidate-source'
    runs=raw_runs(root)
    assert len(runs)==192
    red=json.loads((root/'admission.json').read_text())
    assert red['exit_code'] != 0 and sha(root/'admission.log')==red['raw_sha256']
    assert 'candidate suffix discovery requires the bounded probe' in (root/'admission.log').read_text()
    full_runs=[r for r in raw_runs(bank/'raw/bounded-candidates') if r['phase']=='collection' and r['mode']=='full']
    full={r['prompt']:{(s['round'],s['position']):s for s in r['_states']} for r in full_runs}
    present=set(map(int,(original/'checkpoints/row-oracle/base.txt').read_text().split())) | set(map(int,(original/'checkpoints/row-oracle/base.txt.learned').read_text().split()))
    stats=defaultdict(lambda:defaultdict(int))
    compared=0; error=0.0
    for r in runs:
        assert r['config']['MEMRA_GEMMA_CANDIDATE_SUFFIX']==('1' if r['mode']=='suffix' else '0')
        if r['phase']!='collection':
            assert 'MEMRA_GEMMA_DRAFT_TRACE' not in r['config']
            continue
        path=root/(r['id']+'-draft.jsonl')
        assert sha(path)==r['draft_trace_sha256']
        blocks=[json.loads(x) for x in path.read_text().splitlines()]
        assert [b['round'] for b in blocks]==list(range(1,len(blocks)+1))
        for b in blocks:
            m=b['accepted_prefix']; k=b['drafted']
            assert k==4 and 0<=m<=k and len(b['verifier_ids'])==k+1
            assert b['proposal_ids'][:m]==b['verifier_ids'][:m]
            if m<k: assert b['proposal_ids'][m]!=b['verifier_ids'][m]
        for s in r['_states']:
            history=deque()
            ids=list(map(int,r['argv'][2:]))
            for b in blocks:
                if b['round']>=s['round']: break
                ids.extend(b['verifier_ids'] if r['mode']=='suffix' else b['verifier_ids'][:b['accepted_prefix']+1])
            for token in ids:
                if token in present: continue
                if token in history: history.remove(token)
                history.append(token)
                if len(history)>256: history.popleft()
            expected=list(reversed(history))[:24]
            assert len(expected)==s['history_rows']
            rng=0x9e3779b97f4a7c15 ^ (s['round']-1); mask=(1<<64)-1
            while len(expected)<32:
                rng^=(rng<<13)&mask; rng^=rng>>7; rng^=(rng<<17)&mask
                token=rng%s['vocab']
                if token not in present and token not in expected: expected.append(token)
            scores=dict(s['candidates'])
            assert list(scores)==expected, ('noncausal pool',r['id'],s['round'])
            f=full[r['prompt']][s['round'],s['position']]
            assert all(s[k]==f[k] for k in ['context','proposal','target'])
            key=r['mode']+'-'+r['prompt'].split('-')[0]+'-'+r['prompt'].split('-')[1]
            g=stats[key]
            g['states']+=1
            repair=f['addition_rescue'] or f['removal_rescue'] or f['swap_rescue']
            g['repairable']+=repair; g['discovered']+=repair and f['target'] in scores
            ref=dict(f['outside_top16']); ref[f['target']]=f['target_full_score']
            for token in scores.keys() & ref.keys():
                difference=abs(scores[token]-ref[token]); compared+=1; error=max(error,difference)
                assert difference<=f['tolerance']
    pairs=defaultdict(dict)
    for r in runs:
        if r['phase']=='cost': pairs[r['prompt'],r['repeat']][r['mode']]=r
    ratios=defaultdict(list)
    for (prompt,repeat),p in pairs.items():
        assert set(p)=={'committed','suffix'}
        ratios[prompt].append(math.log(p['suffix']['request_seconds']/p['committed']['request_seconds']))
    assert len(pairs)==48 and len(ratios)==8 and all(len(v)==6 for v in ratios.values())
    means=[statistics.mean(v) for v in ratios.values()]; rng=random.Random(20260913)
    boot=sorted(math.exp(statistics.mean(rng.choices(means,k=8))) for _ in range(10000))
    return {'runs':len(runs),'discovery':stats,'scores_compared':compared,'maximum_score_error':error,
        'suffix_over_committed_cost_ratio':math.exp(statistics.mean(means)),'bootstrap95':[boot[250],boot[9749]],
        'scope':'Candidate-source ablation only; verifier suffixes are hints, never reached-state labels.'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('bank',type=Path); p.add_argument('--original',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); result=analyze(a.bank,a.original); a.out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
