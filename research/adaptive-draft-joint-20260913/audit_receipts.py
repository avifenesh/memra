"""Verify the banked learning receipts and summarize the paired pilot."""
import hashlib
import json
import math
from pathlib import Path
import random
from collections import defaultdict

root=Path(__file__).resolve().parent
raw=root/'raw-learning'
rows=[json.loads(x) for x in (raw/'runs.jsonl').read_text().splitlines()]
for row in rows:
    path=raw/f'{row["id"]}.log'
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==row['raw_sha256'], row['id']
assert len(rows)==56 and all(r['exit_code']==0 for r in rows)
data_path=raw/'acceptance-data.jsonl'
data=[json.loads(x) for x in data_path.read_text().splitlines()]
fit=json.loads((root/'checkpoints/acceptance-predictors.json').read_text())
assert hashlib.sha256(data_path.read_bytes()).hexdigest()==fit['data_sha256']
prompt_manifest=json.loads((root/'prompt-manifest.json').read_text())
assert len({x['prompt_sha256'] for x in prompt_manifest['records']})==len(prompt_manifest['records'])
groups=defaultdict(list)
head_summary={}
for name in ['full','static','adaptive']:
    subset=[r for r in data if r['head']==name]
    head_summary[name]={'labeled_proposals':len(subset),'physical_head_rows':sorted({r['head_rows'] for r in subset}),
        'max_learned_rows':max(r['learned_rows'] for r in subset),
        'mean_confidence_on_labeled_proposals':sum(r['confidence'] for r in subset)/len(subset),
        'empirical_agreement_on_labeled_proposals':sum(r['label'] for r in subset)/len(subset)}
for row in data:
    if row['split']!='heldout': continue
    p=max(1e-6,min(1-1e-6,row['confidence']))
    base=[1,math.log(p/(1-p))/5,row['position']/4,math.log1p(row['context'])/10]
    support=math.log(row['head_rows'])/10
    aware=base+[support,row['learned_rows']/512,base[1]*support]
    predictions=[]
    for name,x in [('independent',base),('head_aware',aware)]:
        z=sum(a*b for a,b in zip(fit[name]['weights'],x))
        predictions.append(1/(1+math.exp(-max(-40,min(40,z)))))
    delta=(predictions[1]-row['label'])**2-(predictions[0]-row['label'])**2
    groups[row['prompt']].append(delta)
paired=[sum(values)/len(values) for values in groups.values()]
rng=random.Random(20260913)
bootstrap=sorted(sum(rng.choice(paired) for _ in paired)/len(paired) for _ in range(10000))
summary={'verified_learning_run_logs':len(rows),'labeled_proposals':len(data),'head_summary':head_summary,
    'paired_heldout_brier_delta':{'unit':'prompt group, equal group weight','groups':len(paired),'head_aware_minus_independent':sum(paired)/len(paired),
    'bootstrap_95_percentile_interval':[bootstrap[250],bootstrap[9749]],'bootstrap_replicates':10000,'seed':20260913},
    'interpretation':'Inconclusive initial diagnostic; no demonstrated head-aware prediction advantage. Unequal head capacities, greedy acceptance, six held-out prompt groups; no throughput claim.'}
(root/'receipt-audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
