#!/usr/bin/env python3
import csv,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
oracle=list(csv.DictReader((R/'oracle.tsv').open(),delimiter='\t'))
gather=list(csv.DictReader((R/'gather.tsv').open(),delimiter='\t'))
summary={'oracle':[],'bench':[],'source':(R/'candidate-source.sha').read_text().strip()}
for p in ('p32k','p128k'):
 for t in (2,4,7):
  rows=[r for r in oracle if r['prompt']==p and int(r['t'])==t]
  gs=[r for r in gather if r['prompt']==p and int(r['t'])==t]
  assert len(rows)==t*64*11 and len(gs)==11
  summary['oracle'].append({'prompt':p,'t':t,'layers':len(gs),'rows':len(rows),'argmax_flips':sum(r['argmax_a']!=r['argmax_b'] for r in rows),'failures':sum(r['pass']!='1' for r in rows),'max_error':max(float(r['max_error']) for r in rows),'max_error_over_band':max(float(r['max_error'])/float(r['band']) for r in rows),'gather_changed':sum(int(r['bits_changed']) for r in gs)})
if (R/'bench.tsv').exists():
 bench=list(csv.DictReader((R/'bench.tsv').open(),delimiter='\t'))
 for p in ('p32k','p128k'):
  for t in (2,4,7):
   for layer in range(-1,11):
    rows=[r for r in bench if r['prompt']==p and int(r['t'])==t and int(r['layer'])==layer]
    assert len(rows)==20
    blocks=[]
    for block in range(5):
     cells=[r for r in rows if int(r['block'])==block]
     assert [r['arm'] for r in cells]==list('ABBA')
     a=statistics.mean(float(r['ms']) for r in cells if r['arm']=='A');b=statistics.mean(float(r['ms']) for r in cells if r['arm']=='B')
     blocks.append((a,b,a-b))
    summary['bench'].append({'prompt':p,'t':t,'layer':layer,'a_ms':statistics.median(x[0] for x in blocks),'b_ms':statistics.median(x[1] for x in blocks),'saving_ms':statistics.median(x[2] for x in blocks),'blocks':blocks})
 summary['weighted_ms']={p:sum(r['saving_ms']*{2:48/162,4:103/162,7:11/162}[r['t']] for r in summary['bench'] if r['prompt']==p and r['layer']==-1) for p in ('p32k','p128k')}
 summary['combined_ms']=statistics.mean(summary['weighted_ms'].values())
 summary['verdict']='KEEP' if summary['combined_ms']>=.5 and not any(r['failures'] for r in summary['oracle']) else 'NEGATIVE'
else:summary['verdict']='NEGATIVE' if any(r['failures'] for r in summary['oracle']) else 'INCOMPLETE'
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='bench'},indent=2))
for r in summary['bench']:
 if r['layer']==-1:print(f"| {r['prompt']} | {r['t']} | {r['a_ms']:.9f} | {r['b_ms']:.9f} | {r['saving_ms']:.9f} |")
