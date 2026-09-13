"""Frozen-state admission models; fitting cannot read fresh test observations."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from audit_candidate_gates import raw_runs

GRID = [0, .01, .025, .05, .1, 1.0]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def build_states(candidate_runs, full_runs):
    full = {r['prompt']:{(s['round'],s['position']):s for s in r['_states']} for r in full_runs}
    result = []
    for r in candidate_runs:
        prompt = set(map(int, r['argv'][2:]))
        prompt_len = len(r['argv'])-2
        for s in r['_states']:
            f = full[r['prompt']][s['round'],s['position']]
            assert all(s[k] == f[k] for k in ['proposal','target','context'])
            survivors = [x for x in f['active_top'] if x[2] != 4096]
            base_token, base_score, _ = survivors[0]
            known = set(r['_tokens'][:s['context']-prompt_len+1])
            candidates = []
            ambiguous = not f['overlap_argmax_match']
            for i, (token, score) in enumerate(s['candidates']):
                margin = score-base_score
                tie = abs(margin) <= f['tolerance'] or (margin < 0 and len(survivors)>1 and abs(base_score-survivors[1][1]) <= f['tolerance'])
                ambiguous |= tie
                winner = token if margin > 0 else base_token
                delta = int(winner == f['target'])-int(f['correct'])
                features = [1.0, max(-20,min(20,margin))/8,
                    max(0,min(20,f['active_top'][0][1]-f['active_top'][1][1]))/8,
                    s['position']/3, float(token in prompt), float(token in known),
                    float(i < s['history_rows']), float(f['active_top'][0][2] == 4096)]
                candidates.append({'token':token, 'score':score, 'delta':delta, 'changes':winner != f['proposal'], 'x':features})
            result.append({'prompt':r['prompt'], 'split':r['prompt'].split('-')[0], 'target':f['target'],
                'candidates':candidates, 'ambiguous':ambiguous,
                'repairable':f['addition_rescue'] or f['removal_rescue'] or f['swap_rescue'],
                'discovered':f['target'] in dict(s['candidates'])})
    return result

def dot(a,b): return sum(x*y for x,y in zip(a,b))

def ridge(states, penalty):
    rows = []
    for s in states:
        choices = [c for c in s['candidates'] if c['changes']]
        if s['ambiguous'] or not choices: continue
        rows.extend((c['x'], c['delta'], 1/len(choices)) for c in choices)
    if len(rows) < 8: return None
    n = 8
    matrix = [[sum(w*x[i]*x[j] for x,y,w in rows)+(penalty if i==j else 0) for j in range(n)]
              +[sum(w*x[i]*y for x,y,w in rows)] for i in range(n)]
    for i in range(n):
        pivot = max(range(i,n), key=lambda j:abs(matrix[j][i]))
        matrix[i],matrix[pivot] = matrix[pivot],matrix[i]
        scale = matrix[i][i]
        assert abs(scale)>1e-12
        matrix[i] = [x/scale for x in matrix[i]]
        for j in range(n):
            if j != i:
                scale = matrix[j][i]
                matrix[j] = [a-scale*b for a,b in zip(matrix[j],matrix[i])]
    coefficients = [matrix[i][-1] for i in range(n)]
    assert all(math.isfinite(x) for x in coefficients)
    return coefficients

def evaluate(states, kind, model):
    groups = defaultdict(list)
    actions = positive = negative = excluded = 0
    for s in states:
        if s['ambiguous']: excluded += 1; continue
        choices = [c for c in s['candidates'] if c['changes']]
        def value(c):
            if kind == 'highest': return c['score']
            if kind == 'context': return dot(model['coefficients'], c['x'])
            return model['table'].get(str(c['token']),0)
        chosen = max(choices,key=lambda c:(value(c),-c['token'])) if choices and kind != 'noop' else None
        if chosen is not None and kind != 'highest' and value(chosen) <= model.get('threshold',0): chosen = None
        delta = chosen['delta'] if chosen else 0
        actions += chosen is not None; positive += delta>0; negative += delta<0
        groups[s['prompt']].append(delta)
    return {'states':sum(map(len,groups.values())), 'prompts':len(groups), 'actions':actions,
        'positive':positive, 'negative':negative, 'net':positive-negative, 'excluded':excluded,
        'prompt_mean_delta':statistics.mean(statistics.mean(v) for v in groups.values()),
        'per_prompt':{k:statistics.mean(v) for k,v in groups.items()}}

def fit(bank, dense=False):
    if dense:
        all_runs = raw_runs(bank/'raw/dense-admission', expected_prompts=32)
        full_runs = [r for r in all_runs if r['mode']=='full']
        source_runs = [{**r, 'mode':'suffix'} for r in all_runs if r['mode']=='bounded']
        modes = {'suffix':build_states(source_runs,full_runs)}
    else:
        source_runs = raw_runs(bank/'raw/candidate-source')
        full_runs = [r for r in raw_runs(bank/'raw/bounded-candidates') if r['phase']=='collection' and r['mode']=='full']
        modes = {m:build_states([r for r in source_runs if r['phase']=='collection' and r['mode']==m],full_runs) for m in ['committed','suffix']}
    recall = {m:sum(s['repairable'] and s['discovered'] for s in states if s['split']=='calibration') for m,states in modes.items()}
    source = max(modes,key=lambda m:(recall[m],m=='committed'))
    train = [s for s in modes[source] if s['split']=='train']
    cal = [s for s in modes[source] if s['split']=='calibration']
    frequency = defaultdict(int)
    for r in source_runs:
        if r['phase']=='collection' and r['mode']==source and r['prompt'].startswith('train-'):
            for token in r['_tokens']: frequency[token] += 1
    tables = {}
    for name in ['utility','conditional']:
        counts = defaultdict(lambda:[0,0])
        for s in train:
            if s['ambiguous']: continue
            for c in s['candidates']:
                if name=='conditional' and not c['changes']: continue
                counts[c['token']][0] += c['delta']; counts[c['token']][1] += 1
        tables[name] = {str(k):total/(n+4) for k,(total,n) in counts.items()}
    models = {'noop':{}, 'highest':{}, 'frequency':{'table':dict((str(k),v) for k,v in frequency.items()),'threshold':0}}
    calibration = {}
    for kind in ['utility','conditional','context']:
        choices = []
        for penalty in ([1,10,100] if kind=='context' else [0]):
            weights = ridge(train,penalty) if kind=='context' else None
            if kind=='context' and weights is None: continue
            for threshold in GRID:
                model = {'coefficients':weights,'penalty':penalty,'threshold':threshold} if kind=='context' else {'table':tables[kind],'threshold':threshold}
                score = evaluate(cal,kind,model)
                choices.append((score['prompt_mean_delta'],-score['actions'],penalty,threshold,model,score))
        if not choices:
            calibration[kind] = {'status':'insufficient_nontrivial_training_interventions'}
            continue
        chosen = max(choices,key=lambda x:x[:4])
        models[kind] = chosen[4]
        calibration[kind] = {'selected':chosen[5], 'grid':[{'mean':x[0],'actions':-x[1],'penalty':x[2],'threshold':x[3]} for x in choices]}
    support = [c for s in train if not s['ambiguous'] for c in s['candidates'] if c['changes']]
    return {'source':source, 'source_calibration_discoveries':recall, 'models':models,'calibration':calibration,
        'training_support':{'states':len(train),'nontrivial_interventions':len(support),'positive':sum(c['delta']>0 for c in support),'negative':sum(c['delta']<0 for c in support)},
        'training_and_calibration_metrics':{split:{k:evaluate(states,k,m) for k,m in models.items()} for split,states in [('train',train),('calibration',cal)]},
        'dense_training':dense,
        'inputs':{str(p.relative_to(bank)):sha(p) for name in (['dense-admission'] if dense else ['candidate-source','bounded-candidates']) for p in sorted((bank/'raw'/name).glob('*')) if p.is_file()},
        'fitter_sha256':sha(Path(__file__)), 'auditor_sha256':sha(Path(__file__).with_name('audit_candidate_gates.py')),
        'python':sys.version, 'scope':'Offline one-step admission only; no live policy or serving claim.'}

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('bank',type=Path); p.add_argument('--out',type=Path,required=True); p.add_argument('--dense',action='store_true')
    a=p.parse_args(); assert not a.out.exists(), 'Refuse checkpoint overwrite'
    checkpoint=fit(a.bank,a.dense); a.out.write_text(json.dumps(checkpoint,indent=2)); print(json.dumps({k:v for k,v in checkpoint.items() if k in ['source','source_calibration_discoveries','training_support']},indent=2))
