#!/usr/bin/env python3
"""Reconcile all requests, derive medians and paired empirical PMIN histograms."""
import collections,hashlib,itertools,json,math,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
rr=re.compile(r'\[rootcause-round\] round=(\d+) k=(\d+) drafted=(\d+) accepted=(\d+) verify_ms=([\d.]+) wall_ms=([\d.]+)')
rows=[];groups=collections.defaultdict(list);greedy={};hists=collections.defaultdict(lambda: {'accepted_ids':collections.Counter(),'accepted_lengths':collections.Counter(),'output_ids':collections.Counter()});oracles=[]
for p in sorted((R/'raw').glob('*/result.json')):
 a=json.loads(p.read_text());d=p.parent
 if not (d/'server.log').exists():continue
 log=(d/'server.log').read_text()
 rounds=[dict(zip(['round','k','drafted','accepted','verify_ms','wall_ms'],[int(x) if i<4 else float(x) for i,x in enumerate(m)])) for m in rr.findall(log)]
 usage=a['usage'];n=usage['completion_tokens'];spec=usage.get('spec',{});count=len(rounds)
 assert a['done'] and a['http']==200
 words=(d/'output.txt').read_text().split();grams=collections.Counter(tuple(words[i:i+12]) for i in range(max(0,len(words)-11)))
 a.update(n=n,cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0),tok_s=n/a['wall_s'],loop_suspect=max(grams.values(),default=0)>=4,rounds=count,drafted=sum(x['drafted'] for x in rounds),accepted=sum(x['accepted'] for x in rounds))
 if a['k'] != 0:
  assert count==spec['rounds'] and a['drafted']==spec['drafted'] and a['accepted']==spec['accepted'],a
  assert a['drafted']>0
  a.update(drafted_per_round=a['drafted']/count,accepted_per_round=a['accepted']/count,acceptance=a['accepted']/a['drafted'],wall_ms_per_round=1000*a['wall_s']/count,engine_ms_per_round=statistics.mean(x['wall_ms'] for x in rounds),verify_ms_per_round=statistics.mean(x['verify_ms'] for x in rounds))
 else:assert not spec and not rounds
 ticks=re.findall(r'\[tick\].*?priming=(\d+).*?prefill_ms=([\d.]+) decode_ms=([\d.]+)',log)
 decode=sum(float(v) for prim,pre,v in ticks if int(prim)==0 and float(pre)==0)
 spec_ticks=re.findall(r'\[tick-spec\].*?wall_ms=([\d.]+) generated=(\d+)->(\d+)',log)
 if a['k'] != 0:
  assert spec_ticks,(a['label'],'missing tick-spec')
  decode=sum(float(ms) for ms,before,after in spec_ticks)
 a['server_tick_ms_per_token']=decode/n
 a['decode_tok_s']=(n-1)/(a['wall_s']-a['ttft_s'])
 dispatch=re.findall(r'\[compose-dispatch\] flags=(\d+) kda=(\d+)',log)
 assert dispatch and int(dispatch[-1][0])==a['flags'],(a['label'],dispatch)
 a['kda_dispatch_cumulative']=int(dispatch[-1][1])
 ids=re.findall(r'\[compose-output-ids\] tokens=(\[[^\]]*\])',log)
 if ids:
  ids=json.loads(ids[-1]);assert len(ids)==n,(a['label'],len(ids),n)
  (d/'token-ids.json').write_text(json.dumps(ids)+'\n')
 if a['label'].startswith('greedy-'):
  assert len(ids)==160
  greedy[a['label']]={'n':n,'ids':ids,'token_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'text_sha256':hashlib.sha256((d/'output.txt').read_bytes()).hexdigest()}
 for t,nr,changed,flips,ratio in re.findall(r'\[compose-mla-oracle\] t=(\d+) rows=(\d+) changed=(\d+) flips=(\d+) max_ratio=([\deE.+-]+)',log):
  oracles.append({'label':a['label'],'t':int(t),'rows':int(nr),'changed':int(changed),'flips':int(flips),'max_ratio':float(ratio)})
 if a['label'].startswith('distribution-'):
  arm=a['label'].split('-')[-1];h=hists[arm];h['output_ids'].update(ids)
  accepted=0;hist_rounds=0
  for line in log.splitlines():
   if '[rootcause-agreement]' not in line:continue
   arr={key:json.loads(val) for key,val in re.findall(r'(drafts|argmax|sampled|p|q|u)=(\[[^\]]*\])',line)}
   m=int(re.search(r'accepted=(\d+)',line).group(1))
   replay=next((i for i,(pv,qv,uv) in enumerate(zip(arr['p'],arr['q'],arr['u'])) if uv*qv>=pv),len(arr['drafts']))
   assert m==replay
   h['accepted_ids'].update(arr['drafts'][:m]);h['accepted_lengths'].update([m]);accepted+=m;hist_rounds+=1
  assert accepted==a['accepted'] and hist_rounds==count
 if a['label'].startswith('timing-') and not a['loop_suspect']:groups[a['label'].split('-')[-1]].append(a)
 rows.append(a)
medians=[]
for arm in ['plain','off','kda','mla','both','all','auto']:
 items=groups[arm]
 if not items:continue
 row={'arm':arm,'n':len(items),'k':items[0]['k']}
 for key in ['tok_s','server_tick_ms_per_token','decode_tok_s','ttft_s','cached','drafted_per_round','accepted_per_round','acceptance','wall_ms_per_round','engine_ms_per_round','verify_ms_per_round']:
  vals=[x[key] for x in items if x.get(key) is not None]
  if vals:row[key]=statistics.median(vals)
 medians.append(row)
def divergence(a,b):
 na=sum(a.values());nb=sum(b.values());keys=set(a)|set(b)
 p={k:a[k]/na for k in keys};q={k:b[k]/nb for k in keys};m={k:(p[k]+q[k])/2 for k in keys}
 return {'n_off':na,'n_on':nb,'total_variation':sum(abs(p[k]-q[k]) for k in keys)/2,'jensen_shannon_bits':sum((p[k]*math.log2(p[k]/m[k]) if p[k] else 0)+(q[k]*math.log2(q[k]/m[k]) if q[k] else 0) for k in keys)/2}
divs={k:divergence(hists['pmin_off'][k],hists['pmin_on'][k]) for k in hists['pmin_off']} if len(hists)==2 else {}
pairs=[{'a':a,'b':b,'token_identical':greedy[a]['ids']==greedy[b]['ids'],'text_identical':greedy[a]['text_sha256']==greedy[b]['text_sha256']} for a,b in itertools.combinations(greedy,2)]
result={'requests':rows,'medians':medians,'greedy':greedy,'exactness_pairs':pairs,'mla_oracles':oracles,'histograms':hists,'divergence':divs}
(R/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('| Arm | K | N | Accepted/round | Acceptance | HTTP ms/round | Tick ms/token | HTTP tok/s |')
print('|---|---:|---:|---:|---:|---:|---:|---:|')
for a in medians:
 print('| '+' | '.join(str(a.get(k,'-')) if k in ('arm','k','n') else f'{a[k]:.6f}' if k in a else '-' for k in ['arm','k','n','accepted_per_round','acceptance','wall_ms_per_round','server_tick_ms_per_token','tok_s'])+' |')
print('Exactness:',pairs)
print('MLA:',len(oracles),'calls',sum(x['rows'] for x in oracles),'rows',sum(x['flips'] for x in oracles),'flips',max((x['max_ratio'] for x in oracles),default=0),'max band ratio')
print('Divergence:',divs)
print('Loops:',[a['label'] for a in rows if a['loop_suspect']])
