#!/usr/bin/env python3
"""Derive tables from completed HTTP records and their exact server log windows."""
import collections,json,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
round_re=re.compile(r'\[rootcause-round\] round=(\d+) k=(\d+) drafted=(\d+) accepted=(\d+) verify_ms=([\d.]+) wall_ms=([\d.]+)')
rows=[];groups=collections.defaultdict(list);widths=collections.defaultdict(list);agreements=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
for p in sorted((R/'raw').glob('*/result.json')):
 a=json.loads(p.read_text());d=p.parent
 if not (d/'server.log').exists():continue
 log=(d/'server.log').read_text()
 rounds=[dict(zip(['round','k','drafted','accepted','verify_ms','wall_ms'],[int(x) if i<4 else float(x) for i,x in enumerate(m)])) for m in round_re.findall(log)]
 usage=a['usage'];n=usage['completion_tokens'];spec=usage.get('spec',{});count=len(rounds)
 if a['k'] != 0:assert count>0 and spec, a['label']
 words=(d/'output.txt').read_text().split();grams=collections.Counter(tuple(words[i:i+12]) for i in range(max(0,len(words)-11)))
 loop=max(grams.values(),default=0)>=4
 a.update(n=n,cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0),tok_s=n/a['wall_s'],loop_suspect=loop,rounds=count,drafted=sum(x['drafted'] for x in rounds),accepted=sum(x['accepted'] for x in rounds))
 if count:
  assert count==spec['rounds'],(a['label'],count,spec)
  assert a['drafted']==spec['drafted'] and a['accepted']==spec['accepted'],(a['label'],a,spec)
  a.update(drafted_per_round=a['drafted']/count,accepted_per_round=a['accepted']/count,acceptance=a['accepted']/a['drafted'],wall_ms_per_round=1000*a['wall_s']/count,engine_ms_per_round=statistics.mean(x['wall_ms'] for x in rounds),verify_ms_per_round=statistics.mean(x['verify_ms'] for x in rounds))
 if a.get('ttft_s') is not None:
  a['decode_tok_s']=(n-1)/(a['wall_s']-a['ttft_s'])
 rows.append(a)
 if a['label'].startswith('timing-') and not loop:
  sampler=a['label'].split('-')[2];groups[(sampler,str(a['k']))].append(a)
  for x in rounds:widths[x['drafted']+1].append(x['verify_ms'])
 if a['mode']=='agreement' and a['label'].startswith('agreement-'):
  sampler=a['label'].split('-')[1];arm=f"{sampler}-K{a['k']}"
  for line in log.splitlines():
   if '[rootcause-agreement]' in line or '[rootcause-greedy]' in line:
    arr={name:json.loads(val) for name,val in re.findall(r'(drafts|argmax|sampled|p|q|u)=(\[[^\]]*\])',line)}
    drafts=arr['drafts'];argmax=arr['argmax'];sampled=arr.get('sampled',argmax)
    accepted=int(re.search(r'accepted=(\d+)',line).group(1)) if 'accepted=' in line else next((i for i,(x,y) in enumerate(zip(drafts,argmax)) if x!=y),len(drafts))
    if 'p' in arr:
     replay=next((i for i,(p0,q0,u0) in enumerate(zip(arr['p'],arr['q'],arr['u'])) if u0*q0>=p0),len(drafts))
     assert replay==accepted,(arm,replay,accepted)
    for i,(x,y,z) in enumerate(zip(drafts,argmax,sampled)):
     c=agreements[arm][i+1];c.update(n=1,argmax=int(x==y),sampled=int(x==z),accepted=int(i<accepted),reached=int(i<=accepted),argmax_rejected=int(x==y and i==accepted),reached_argmax=int(x==y and i<=accepted),reached_sampled=int(x==z and i<=accepted))
     if 'p' in arr:c.update(p=arr['p'][i],ratio=min(1,arr['p'][i]/arr['q'][i]))
summary=[]
for (sampler,k),items in groups.items():
 row={'sampler':sampler,'k':k,'n':len(items)}
 for key in ['tok_s','decode_tok_s','ttft_s','cached','drafted_per_round','accepted_per_round','acceptance','wall_ms_per_round','engine_ms_per_round','verify_ms_per_round']:
  vals=[x[key] for x in items if x.get(key) is not None]
  if vals:row[key]=statistics.median(vals)
 summary.append(row)
result={'requests':rows,'medians':summary,'verify_widths':{k:{'n':len(v),'median_ms':statistics.median(v),'mean_ms':statistics.mean(v)} for k,v in widths.items()},'agreement':{arm:{str(pos):dict(c) for pos,c in ps.items()} for arm,ps in agreements.items()}}
(R/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('| sampler | K | N | drafted/round | accepted/round | acceptance | wall ms/round | verify ms/round | tok/s |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for a in summary:
 print('| '+' | '.join(str(a.get(k,'-')) if k in ('sampler','k','n') else f'{a[k]:.3f}' if k in a else '-' for k in ['sampler','k','n','drafted_per_round','accepted_per_round','acceptance','wall_ms_per_round','verify_ms_per_round','tok_s'])+' |')
print('loops:',[a['label'] for a in rows if a['loop_suspect']])
