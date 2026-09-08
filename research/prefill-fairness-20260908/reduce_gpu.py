"""Receipt reducer. Failed/incomplete boots remain visible and never enter medians."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics

p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--out',required=True);a=p.parse_args()
root=pathlib.Path(a.root)
def percentile(values,q=.95):
    values=sorted(values)
    return values[max(0,math.ceil(q*len(values))-1)] if values else None
result={'mixed':[], 'chunks':[], 'byte_gates':[], 'failures':[]}
for run in sorted(root.iterdir()):
    if not run.is_dir() or not (run/'identity.json').exists():continue
    identity=json.loads((run/'identity.json').read_text())
    log=(run/'server.log').read_text() if (run/'server.log').exists() else ''
    if (run/'stage0-summary.json').exists():
        summary=json.loads((run/'stage0-summary.json').read_text())
        rows=[json.loads(x) for x in (run/'requests.jsonl').read_text().splitlines()]
        rows=[r for r in rows if r.get('stage')=='stage0']
        small=[r for r in rows if r.get('class')!='longprefill']
        long=[r for r in rows if r.get('class')=='longprefill']
        eligible=summary['counts']=={'completed':21} and not summary['aborted'] and not summary['faults'] and len(long)==1
        row={'run':run.name,'eligible':eligible,'boot_nonce':identity['boot_nonce'],
             'binary_sha256':identity.get('sha256',identity.get('binary_sha256')),
             'small_n':len(small),'small_p95_ttft_s':percentile([r['ttft_s'] for r in small if r.get('ttft_s') is not None]),
             'small_max_ttft_s':max((r.get('ttft_s') or 0 for r in small),default=0),
             'partial_p95_ttft_s':summary.get('classes',{}).get('1066-partial-target',{}).get('p95_ttft_s'),
             'long_ttft_s':long[0].get('ttft_s') if long else None,'long_total_s':long[0].get('e2e_s') if long else None,
             'long_input_tokens':long[0].get('usage',{}).get('prompt_tokens') if long else None,
             'server_delta':summary['server_delta'],'queue_peak':summary['queue_peak'],
             'yield_count':len(re.findall(r'^\[prime-yield\]',log,re.M))}
        result['mixed'].append(row)
        if not eligible:result['failures'].append({'run':run.name,'summary':summary})
    elif (run/'summary.json').exists():
        summary=json.loads((run/'summary.json').read_text())
        if identity.get('args',{}).get('mode')=='chunk':
            # One HTTP request after readiness; discard boot calibration operations.
            relevant=log[log.index('[meter] admit'):]
            phases={}
            for phase,rows,wall in re.findall(r'^\[prime-chunk\] phase=(\S+) rows=(\d+) wall_ms=([\d.]+)',relevant,re.M):
                phases.setdefault(phase,[]).append({'rows':int(rows),'wall_ms':float(wall)})
            parts={}
            for phase,values in phases.items():
                parts[phase]={'rows':sum(v['rows'] for v in values),'chunks':len(values),'all_max_ms':max(v['wall_ms'] for v in values)}
                for part in range(3):
                    subset=values[len(values)*part//3:len(values)*(part+1)//3]
                    walls=[v['wall_ms'] for v in subset]
                    parts[phase][('early','middle','late')[part]]={'n':len(walls),'median_ms':statistics.median(walls) if walls else None,'p95_ms':percentile(walls),'max_ms':max(walls,default=0)}
            result['chunks'].append({'run':run.name,'phases':parts,'requests':summary['rows'],'yield_count':summary['yield_count']})
        else:result['byte_gates'].append({'run':run.name,**summary})
    elif (run/'cleanup.json').exists():
        result['failures'].append({'run':run.name,'error':(run/'error.txt').read_text() if (run/'error.txt').exists() else 'no final summary'})
result['medians']={}
for model in ('ornith','qwen'):
    table={}
    for arm in ('off','on'):
        rows=[r for r in result['mixed'] if r['eligible'] and r['run'].startswith(model+'-r') and re.search('-'+arm+'(?:-v2)?$',r['run'])]
        table[arm]={'boots':len(rows),'nonce_count':len({r['boot_nonce'] for r in rows})}
        for metric in ('small_p95_ttft_s','small_max_ttft_s','partial_p95_ttft_s','long_ttft_s','long_total_s'):
            values=[r[metric] for r in rows if r[metric] is not None]
            table[arm][metric]={'median':statistics.median(values) if values else None,'min':min(values,default=None),'max':max(values,default=None)}
    result['medians'][model]=table
pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['medians'],indent=2))
