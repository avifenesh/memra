#!/usr/bin/env python3
"""Summarize completed paired probes; profiler rows never enter timing medians."""
import collections,hashlib,json,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path(__file__).parent/'receipts/pair-box'
rows=[];pairs=[]
for context in ('128k','1m'):
 for pair in range(1,4):
  ids=[]
  for arm in ('off','on'):
   cell=root/f'pair{pair}-{context}-{arm}'
   if not (cell/'exit').exists() or (cell/'exit').read_text().strip()!='0':raise SystemExit(f'{cell.name}: incomplete or failed')
   r=json.loads((cell/'rows.jsonl').read_text());blob=(cell/'long.ids').read_bytes();ids.append(blob)
   if len(blob.split())!=160 or r['out_tokens']!=160:raise SystemExit(f'{cell.name}: expected160 IDs')
   words=(cell/'long.txt').read_text().split();repeats=max(collections.Counter(tuple(words[i:i+16]) for i in range(max(0,len(words)-15))).values(),default=0)
   r.update(cell=cell.name,context=context,pair=pair,split=arm,ids_sha256=hashlib.sha256(blob).hexdigest(),max_repeated_16_word_span=repeats,timing_eligible=repeats<4)
   rows.append(r)
  if ids[0]!=ids[1]:raise SystemExit(f'{context} pair{pair}: ID DIVERGENCE')
  pairs.append({'context':context,'pair':pair,'ids_identical':True,'sha256':hashlib.sha256(ids[0]).hexdigest()})
medians={}
for context in ('128k','1m'):
 medians[context]={}
 for arm in ('off','on'):
  group=[r for r in rows if r['context']==context and r['split']==arm and r['timing_eligible']]
  medians[context][arm]={'n':len(group),'prime_s':statistics.median(r['prime_s'] for r in group),'decode_tok_s':statistics.median(r['decode_tok_s'] for r in group)}
 a=medians[context]['off'];b=medians[context]['on']
 medians[context]['prime_time_change_pct']=100*(b['prime_s']/a['prime_s']-1)
 medians[context]['decode_rate_change_pct']=100*(b['decode_tok_s']/a['decode_tok_s']-1)
profiles={}
for arm in ('off','on'):
 p=root/f'profile-1m-{arm}'/'tally/summary.json'
 if p.exists():profiles[arm]=json.loads(p.read_text())
summary={'timed_identity':'PASS','rows':rows,'pairs':pairs,'medians':medians,'profiles':profiles,'profile_state':'PASS' if len(profiles)==2 else 'PENDING','main_128k_ids_match':all(r['ids_sha256']=='308075f01e83ff9fce016c5ff5976e9b0578c287de7419edca941852edf7c087' for r in rows if r['context']=='128k')}
(root/'paired-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('| Context | Pair | OFF prime s | ON prime s | OFF decode tok/s | ON decode tok/s | IDs |')
print('|---|---:|---:|---:|---:|---:|---|')
for p in pairs:
 a,b=[next(r for r in rows if r['context']==p['context'] and r['pair']==p['pair'] and r['split']==arm) for arm in ('off','on')]
 print(f"| {p['context']} | {p['pair']} | {a['prime_s']:.4f} | {b['prime_s']:.4f} | {a['decode_tok_s']:.3f} | {b['decode_tok_s']:.3f} | identical |")
print(json.dumps(medians,indent=2))
