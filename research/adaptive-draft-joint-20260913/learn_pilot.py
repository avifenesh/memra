"""Own-generation ranks for both targets; Gemma head/confidence pilot.

This is a diagnostic, not the equal-capacity performance comparison. Full,
static and append-only adaptive heads have different physical row counts.
"""
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path('/workspace/adaptive-draft')
BIN = ROOT/'memra/target/release'
OUT = ROOT/'raw/learning'
OUT.mkdir(parents=True,exist_ok=True)
PROMPTS = ROOT/'code-prompts'
CHECK = ROOT/'checkpoints'
CHECK.mkdir(exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def execute(name, argv, config, timeout=1800):
    log=OUT/f'{name}.log'
    assert not log.exists(), f'Refusing overwrite: {log}'
    env={k:v for k,v in os.environ.items() if not k.startswith('MEMRA_')}
    env.update(config)
    start=time.time()
    with log.open('wb') as f:
        process=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    row={'id':name,'argv':argv,'config':config,'utc':dt.datetime.fromtimestamp(start,dt.timezone.utc).isoformat(),
         'exit_code':process.returncode,'wall_seconds':time.time()-start,'raw_sha256':sha(log)}
    with (OUT/'runs.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    if process.returncode:
        raise RuntimeError(f'{name}: exit={process.returncode}; raw log retained')
    return log

ready=json.loads((ROOT/'PILOT_READY.json').read_text())
pilot=[json.loads(x) for x in (ROOT/ready['runs']).read_text().splitlines()]
assert all(x['status']=='pass' for x in pilot), 'Resolve native pilot failures before learning'
with open('/tmp/memra-5090.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU occupied'
    with (OUT/'telemetry.csv').open('w') as tf:
        telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv','-lms','250'],stdout=tf)
        try:
            for model in ['gemma12','qwen9']:
                corpus=CHECK/f'{model}-train-tokens.txt'
                ranks=CHECK/f'{model}-ranks4096.gguf'
                execute(f'{model}-own-generation',[str(BIN/'frspec-owngen'),str(ROOT/f'models/{model}.gguf'),str(ranks),'4096',
                    '--ngen','256','--temp','0','--corpus-out',str(corpus),str(PROMPTS/'train')],{})
                total=sum(len(line.split()) for line in corpus.read_text().splitlines())
                receipt={'model':model,'generated_tokens':total,'minimum_tokens':16384,'corpus_sha256':sha(corpus),
                         'ranks_sha256':sha(ranks),'prompt_manifest_sha256':sha(PROMPTS/'manifest.json'),'meets_corpus_floor':total>=16384}
                (CHECK/f'{model}-rank-receipt.json').write_text(json.dumps(receipt,indent=2))
                if total<16384:
                    raise RuntimeError(f'{model}: corpus below 4x head-size floor ({total}); extend training data before head experiments')

            records=[]
            for split in ['train','calibration','heldout']:
                # Fixed subset selected without examining model results.
                for prompt in sorted((PROMPTS/split).glob('*.txt'))[:6]:
                    tokens=subprocess.check_output([str(BIN/'draft_prompt'),str(ROOT/'models/gemma12.gguf'),str(prompt)],text=True)
                    ids=json.loads(tokens.splitlines()[-1])
                    for head in ['full','static','adaptive']:
                        name=f'{split}-{prompt.stem}-{head}'
                        trace=OUT/f'{name}.jsonl'
                        config={'MEMRA_SPEC':'4','MEMRA_DRAFT':str(ROOT/'models/gemma12-draft.gguf'),'MEMRA_NGEN':'128',
                                'MEMRA_GATE_DUMP_TOKENS':'1','MEMRA_SPEC_STATS':'1','MEMRA_SPEC_ADAPT':'0',
                                'MEMRA_SPEC_PMIN':'0','MEMRA_SPEC_PMIN_INROUND':'0','MEMRA_GEMMA_DRAFT_TRACE':str(trace)}
                        if head!='full':
                            ranks=CHECK/f'{name}-ranks.txt'
                            shutil.copyfile(CHECK/'gemma12-ranks4096.gguf.txt',ranks)
                            config.update(MEMRA_GEMMA_DRAFT_RANKS=str(ranks),MEMRA_GEMMA_TRIM_ADAPT='512' if head=='adaptive' else '0')
                        log=execute(name,[str(BIN/'gemma-gate'),str(ROOT/'models/gemma12.gguf')]+list(map(str,ids)),config,300)
                        text=log.read_text()
                        plain=re.search(r'plain tokens: (\[[^\n]+\])',text)
                        spec=re.search(r'spec tokens: (\[[^\n]+\])',text)
                        assert plain and spec and json.loads(plain[1])==json.loads(spec[1]), f'Token mismatch: {name}'
                        rows=[json.loads(line) for line in trace.read_text().splitlines()]
                        assert rows, f'No trace: {name}'
                        for row in rows:
                            # Only proposals reached along the accepted prefix have
                            # actual-continuation labels. The first rejection is 0.
                            for j,p in enumerate(row['confidence']):
                                if j>row['accepted_prefix']:
                                    continue
                                records.append({'split':split,'prompt':prompt.stem,'head':head,'position':j,
                                    'confidence':p,'context':row['context'],'head_rows':row['head_rows'],
                                    'learned_rows':row['learned_rows'],'label':int(j<row['accepted_prefix'])})
            (OUT/'acceptance-data.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        finally:
            telemetry.terminate()
            telemetry.wait(timeout=10)

# Models estimate current proposal agreement. They are not yet a deployed
# next-draw controller: a drawn token must be retained when making that decision.
def sigmoid(x):
    return 1/(1+math.exp(-max(-40,min(40,x))))

def features(row,head_aware):
    p=max(1e-6,min(1-1e-6,row['confidence']))
    x=[1,math.log(p/(1-p))/5,row['position']/4,math.log1p(row['context'])/10]
    if head_aware:
        support=math.log(row['head_rows'])/10
        x += [support,row['learned_rows']/512,x[1]*support]
    return x

def fit(rows,aware):
    xy=[(features(r,aware),r['label']) for r in rows]
    w=[0.0]*len(xy[0][0])
    for _ in range(1000):
        gradient=[0.0]*len(w)
        for x,y in xy:
            error=sigmoid(sum(a*b for a,b in zip(w,x)))-y
            for j,v in enumerate(x): gradient[j]+=error*v
        for j in range(len(w)):
            w[j]-=.15*(gradient[j]/len(xy)+(0 if j==0 else .001*w[j]))
    return w

results={}
for aware in [False,True]:
    w=fit([r for r in records if r['split']=='train'],aware)
    metrics={}
    for split in ['train','calibration','heldout']:
        rows=[r for r in records if r['split']==split]
        estimates=[max(1e-9,min(1-1e-9,sigmoid(sum(a*b for a,b in zip(w,features(r,aware)))))) for r in rows]
        metrics[split]={'n_labeled_proposals':len(rows),'n_prompt_groups':len({r['prompt'] for r in rows}),
            'brier':sum((p-r['label'])**2 for p,r in zip(estimates,rows))/len(rows),
            'log_loss':-sum(r['label']*math.log(p)+(1-r['label'])*math.log(1-p) for p,r in zip(estimates,rows))/len(rows)}
    results['head_aware' if aware else 'independent']={'weights':w,'metrics':metrics}
results['scope']='Small diagnostic on code review, greedy current-proposal acceptance; unequal head widths; no speed or joint-policy novelty claim.'
results['data_sha256']=sha(OUT/'acceptance-data.jsonl')
(CHECK/'acceptance-predictors.json').write_text(json.dumps(results,indent=2))
(ROOT/'LEARNING_DONE').write_text(dt.datetime.now(dt.timezone.utc).isoformat())
print(json.dumps(results,indent=2),flush=True)
