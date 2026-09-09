#!/usr/bin/env python3
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('raw',type=Path);a=p.parse_args()
rows=list(csv.DictReader((a.raw/'bench.tsv').open(),delimiter='\t'))
groups=defaultdict(dict)
for r in rows:
    key=(r['regime'],int(r['t']),r['layer'],int(r['block']))
    order=int(r['order']);assert order not in groups[key]
    assert r['arm']==('current' if order in (0,3) else 'fused')
    groups[key][order]=float(r['us'])
paired=defaultdict(list)
for (regime,t,layer,block),x in groups.items():
    assert set(x)=={0,1,2,3}
    current=(x[0]+x[3])/2;fused=(x[1]+x[2])/2
    paired[(regime,t,layer)].append(dict(block=block,current_us=current,fused_us=fused,saving_us=current-fused))
summary={};per_layer=[]
for (regime,t,layer),blocks in paired.items():
    assert sorted(x['block'] for x in blocks)==list(range(5))
    row=dict(t=t,layer=layer,current_us=statistics.median(x['current_us'] for x in blocks),fused_us=statistics.median(x['fused_us'] for x in blocks),saving_us=statistics.median(x['saving_us'] for x in blocks),blocks=blocks)
    if regime=='rotating':summary[t]=row
    else:per_layer.append(row)
assert len(per_layer)==102 and set(summary)=={2,4,7}
weighted=sum(summary[t]['saving_us']*w for t,w in [(2,48),(4,103),(7,11)])/162/1000
result=dict(regime='34-layer rotation; complete six-projection chains; warmed ABBA x5',widths=summary,weighted_saving_ms=weighted,verdict='KEEP' if weighted>=0.5 else 'NEGATIVE')
(a.raw/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
with (a.raw/'per-layer.tsv').open('w') as f:
    f.write('t\tlayer\tcurrent_us\tfused_us\tpaired_saving_us\n')
    for r in sorted(per_layer,key=lambda r:(r['t'],int(r['layer']))):
        f.write(f'{r["t"]}\t{r["layer"]}\t{r["current_us"]:.9f}\t{r["fused_us"]:.9f}\t{r["saving_us"]:.9f}\n')
print('| t | Current ms/round | Fused ms/round | Paired saving ms/round |')
print('|---|---:|---:|---:|')
for t,r in sorted(summary.items()):print(f'| {t} | {r["current_us"]/1000:.9f} | {r["fused_us"]/1000:.9f} | {r["saving_us"]/1000:.9f} |')
print(f'Weighted saving: {weighted:.9f} ms/round. {result["verdict"]}.')
