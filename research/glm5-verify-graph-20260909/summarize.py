"""Reconcile HTTP, usage, server ticks, graph engagement and exact oracle receipts."""
import collections, hashlib, json, re, statistics
from pathlib import Path
R=Path(__file__).resolve().parent
raw=R/'raw';rows=[];groups=collections.defaultdict(list)
rr=re.compile(r'\[verify-cell-round\] round=(\d+) k=(\d+) drafted=(\d+) accepted=(\d+) verify_ms=([\d.]+)')
for p in sorted(raw.glob('*/result.json')):
 a=json.loads(p.read_text());d=p.parent;log=(d/'server.log').read_text()
 rounds=[dict(zip(['round','k','drafted','accepted','verify_ms'],[int(x) if i<4 else float(x) for i,x in enumerate(m)])) for m in rr.findall(log)]
 usage=a['usage'];n=usage['completion_tokens'];spec=usage.get('spec',{});count=len(rounds)
 assert a['done'] and a['http']==200
 events=[json.loads(line[6:]) for line in (d/'response.sse').read_text().splitlines() if line.startswith('data: {')]
 text=''.join(c.get('delta',{}).get(f,'') for e in events for c in e.get('choices',[]) for f in ['content','reasoning_content','reasoning'])
 words=text.split();grams=collections.Counter(tuple(words[i:i+12]) for i in range(max(0,len(words)-11)))
 a.update(n=n,cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0),tok_s=n/a['wall_s'],loop_suspect=max(grams.values(),default=0)>=4,rounds=count)
 if a['k']!=0:
  assert count==spec['rounds'] and sum(x['drafted'] for x in rounds)==spec['drafted'] and sum(x['accepted'] for x in rounds)==spec['accepted']
  a.update(accepted_per_round=spec['accepted']/count,http_ms_per_round=1000*a['wall_s']/count,verify_ms_per_round=statistics.mean(x['verify_ms'] for x in rounds))
  ticks=re.findall(r'\[tick-spec\].*?wall_ms=([\d.]+) generated=(\d+)->(\d+)',log);assert ticks
  a['tick_ms_per_token']=sum(float(ms) for ms,_,_ in ticks)/n
 else:
  assert not spec and not rounds
  ticks=re.findall(r'\[tick\].*?priming=(\d+).*?prefill_ms=([\d.]+) decode_ms=([\d.]+)',log)
  a['tick_ms_per_token']=sum(float(ms) for prim,pre,ms in ticks if int(prim)==0 and float(pre)==0)/n
 a['captures']=len(re.findall(r'\[glm5-verify-graph\] engaged:.*captured:',log))
 a['capture_widths']=sorted({int(t) for t in re.findall(r'\[glm5-verify-graph\] engaged:.* t=(\d+) captured:',log)})
 a['refusals']=re.findall(r'\[glm5-verify-graph\].*(?:refused|FAILED|latched).*',log)
 if a['label'].startswith('timing-'):
  assert not a['loop_suspect'],('looped timing',a['label'])
  groups[a['label'].split('-',2)[2]].append(a)
 rows.append(a)
medians=[]
for arm in ['plain','K6-off','K6-on','auto-on']:
 items=groups[arm];assert len(items)==3,(arm,len(items))
 a={'arm':arm,'n':len(items),'k':items[0]['k']}
 for key in ['accepted_per_round','http_ms_per_round','verify_ms_per_round','tick_ms_per_token','tok_s']:
  vals=[r[key] for r in items if key in r]
  if vals:a[key]=statistics.median(vals)
 medians.append(a)
oracles=[]
for p in sorted(raw.glob('oracle-*.json')):
 a=json.loads(p.read_text());assert a['equal']
 ids=json.loads(a['off']['verify-cell-output'][-1].split('tokens=',1)[1]);assert len(ids)==160
 oracles.append({'kind':a['kind'],'k':a['k'],'equal':a['equal'],'tokens':len(ids),'rounds':len(a['off']['verify-cell-tape']),'logit_rounds':len(a['off']['verify-cell-logits']),'token_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()})
assert len(oracles)==4
plain,off,on,auto=medians
result={'requests':rows,'medians':medians,'oracles':oracles,'gain_vs_plain_pct':100*(on['tok_s']/plain['tok_s']-1),'http_ms_per_round_saved':off['http_ms_per_round']-on['http_ms_per_round'],'verify_ms_per_round_saved':off['verify_ms_per_round']-on['verify_ms_per_round']}
result['verdict']='KEEP' if result['gain_vs_plain_pct']>=3 else 'NEGATIVE' if on['tok_s']<off['tok_s'] else 'FLAT'
(R/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('| Arm | K | N | Accepted/round | HTTP ms/round | Tick ms/token | HTTP tok/s |')
print('|---|---:|---:|---:|---:|---:|---:|')
for a in medians:print('| '+' | '.join(str(a.get(k,'-')) if k in ['arm','k','n'] else f'{a[k]:.6f}' if k in a else '-' for k in ['arm','k','n','accepted_per_round','http_ms_per_round','tick_ms_per_token','tok_s'])+' |')
print(json.dumps({k:v for k,v in result.items() if k not in ['requests','medians']},indent=2))
