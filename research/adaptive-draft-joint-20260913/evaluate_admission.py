"""Evaluate frozen admission models once; no fitting or tuning on fresh labels."""
from collections import deque, defaultdict
import json
from pathlib import Path
import statistics
from admission_model import build_states, evaluate
from audit_candidate_gates import raw_runs, sha

def analyze(bank, original, study='fresh-admission'):
    assert study in ['fresh-admission','fresh-admission2','dense-admission']
    dense = study == 'dense-admission'
    second = study == 'fresh-admission2'
    root=bank/'raw'/study
    manifest=json.loads((root/'manifest.json').read_text())
    checkpoint=bank/'checkpoints'/('admission-dense-model.json' if second else 'admission-model.json')
    frozen=(bank/'raw'/('admission-dense-model-frozen.sha256' if second else 'admission-model-frozen.sha256')).read_text().split()[0]
    assert sha(checkpoint)==manifest['checkpoint_sha256']==frozen
    model=json.loads(checkpoint.read_text())
    prompts=bank/'checkpoints'/(study+'-prompts')
    if not dense:
        before = 'fresh2-prompts-before-generation.json' if second else 'fresh-prompts-before-generation.json'
        assert manifest['prompts']==json.loads((bank/'raw'/before).read_text())
    else:
        old = original/'raw/row-oracle/prompts'
        for name,digest in manifest['prompts'].items():
            assert name.startswith(('train-','calibration-')) and sha(old/name)==digest
    for name,digest in manifest['prompts'].items(): assert sha(prompts/name)==digest
    runs=raw_runs(root,expected_prompts=32 if dense else 24)
    assert len(runs)==(64 if dense else 48)
    assert all(r['config'].get('MEMRA_GEMMA_PROBE_EVERY','16')==('1' if dense else '16') for r in runs)
    candidates=[r for r in runs if r['mode']=='bounded']
    full=[r for r in runs if r['mode']=='full']
    present=set(map(int,(original/'checkpoints/row-oracle/base.txt').read_text().split())) | set(map(int,(original/'checkpoints/row-oracle/base.txt.learned').read_text().split()))
    references = {r['prompt']:{(s['round'],s['position']):s for s in r['_states']} for r in full}
    compared = 0
    max_error = 0.0
    for r in candidates:
        assert r['config']['MEMRA_GEMMA_CANDIDATE_SUFFIX']==('1' if model['source']=='suffix' else '0')
        path=root/(r['id']+'-draft.jsonl'); assert sha(path)==r['draft_trace_sha256']
        blocks=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(r['_states']) == len(references[r['prompt']])
        for s in r['_states']:
            f = references[r['prompt']][s['round'],s['position']]
            assert all(s[k]==f[k] for k in ['proposal','target','context'])
            scores = dict(f['outside_top16'])
            scores[f['target']] = f['target_full_score']
            for token,score in s['candidates']:
                if token in scores:
                    error = abs(score-scores[token]); max_error=max(max_error,error)
                    assert error <= f['tolerance']
                    compared += 1
            history=deque(); ids=list(map(int,r['argv'][2:]))
            for b in blocks:
                if b['round']>=s['round']: break
                ids.extend(b['verifier_ids'] if model['source']=='suffix' else b['verifier_ids'][:b['accepted_prefix']+1])
            for token in ids:
                if token in present: continue
                if token in history: history.remove(token)
                history.append(token)
                if len(history)>256: history.popleft()
            expected=list(reversed(history))[:24]; assert len(expected)==s['history_rows']
            rng=0x9e3779b97f4a7c15 ^ (s['round']-1); mask=(1<<64)-1
            while len(expected)<32:
                rng^=(rng<<13)&mask; rng^=rng>>7; rng^=(rng<<17)&mask
                token=rng%s['vocab']
                if token not in present and token not in expected: expected.append(token)
            assert expected==[x[0] for x in s['candidates']]
    states=build_states(candidates,full)
    metrics={kind:evaluate(states,kind,params) for kind,params in model['models'].items()}
    family={kind:{f:statistics.mean(v for p,v in m['per_prompt'].items() if p.split('-')[0]==f)
                  for f in sorted({p.split('-')[0] for p in m['per_prompt']})} for kind,m in metrics.items()}
    strongest=max(metrics[k]['prompt_mean_delta'] for k in ['noop','frequency','highest'])
    advance=[k for k in ['utility','conditional','context'] if k in metrics and metrics[k]['prompt_mean_delta']>strongest]
    return {'study':study,'runs':len(runs),'states':len(states),'source':model['source'], 'checkpoint_sha256':frozen,
        'scores_compared':compared,'maximum_score_error':max_error,
        'training_support':model['training_support'], 'metrics':metrics, 'family_means':family,
        'eligible_to_test_online':[] if dense else advance,
        'scope':'Fresh synthetic six-family one-step admission pilot. No serving speedup or broad-generalization claim.'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('bank',type=Path); p.add_argument('--original',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--study',choices=['fresh-admission','fresh-admission2','dense-admission'],default='fresh-admission')
    a=p.parse_args(); result=analyze(a.bank,a.original,a.study); a.out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
