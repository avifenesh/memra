#!/usr/bin/env python3
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
s=json.loads((R/'summary.json').read_text())
med=s['medians'];order={'greedy':0,'t06':1,'vendor':2,'p1':3,'k40':4,'plain':5}
med.sort(key=lambda x:(order[x['sampler']],{'2':0,'4':1,'6':2,'auto':3,'0':4}[x['k']]))
def fmt(x,key,places=3):return f'{x[key]:.{places}f}' if key in x else '-'
a=['| Sampling | K | N | Drafted/round | Accepted/round | Acceptance | HTTP ms/round | Engine ms/round | Verify ms/round | HTTP tok/s |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in med:a.append('| '+' | '.join([x['sampler'],x['k'],str(x['n'])]+[fmt(x,k) for k in ['drafted_per_round','accepted_per_round','acceptance','wall_ms_per_round','engine_ms_per_round','verify_ms_per_round','tok_s']])+' |')
(R/'ACCEPTANCE.md').write_text('\n'.join(a)+'\n')
a=['| Arm | Position | N verified | Argmax match | Shadow sample match | N reached | Argmax match/reached | Sample match/reached | Accepted/reached | Argmax rejected at reached slot |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for arm,positions in sorted(s['agreement'].items()):
 for pos,c in sorted(positions.items(),key=lambda x:int(x[0])):
  def pct(key,den):return f"{100*c[key]/c[den]:.2f}%" if c[den] else '-'
  a.append('| '+' | '.join([arm,pos,str(c['n']),pct('argmax','n'),pct('sampled','n'),str(c['reached']),pct('reached_argmax','reached'),pct('reached_sampled','reached'),pct('accepted','reached'),str(c['argmax_rejected'])])+' |')
(R/'AGREEMENT.md').write_text('\n'.join(a)+'\n')
plain=next(x for x in med if x['sampler']=='plain')
a=[f"Plain baseline: {plain['tok_s']:.6f} HTTP tok/s; {plain['decode_tok_s']:.6f} decode-interval tok/s; {1000/plain['decode_tok_s']:.6f} ms per plain token interval (client wall, includes sampler/transport, not isolated GPU).",'', '| Vendor K | Observed accepted/round | HTTP ms/round | Break-even accepted/round | Engine ms/round | Decode break-even accepted/round |','|---|---:|---:|---:|---:|---:|']
for x in med:
 if x['sampler']=='vendor':a.append(f"| {x['k']} | {x['accepted_per_round']:.6f} | {x['wall_ms_per_round']:.6f} | {plain['tok_s']*x['wall_ms_per_round']/1000-1:.6f} | {x['engine_ms_per_round']:.6f} | {plain['decode_tok_s']*x['engine_ms_per_round']/1000-1:.6f} |")
a+=['','Break-even uses a+1 tokens/round and ignores first-anchor/final-cap boundary corrections. The HTTP calculation compares HTTP with HTTP; the decode screening calculation uses engine round wall and the plain client interval. Neither is a hardware ceiling.','','| Verify width | Current synchronized verify median ms | N current rounds | Prior tally GPU span ms | Prior N |','|---|---:|---:|---:|---:|']
prior=json.loads((R/'tally-reference/k6-summary.json').read_text())
for t in ['2','3','4','5','6','7']:
 v=s['verify_widths'].get(t,{})
 a.append(f"| {t} | {fmt(v,'median_ms',6)} | {v.get('n',0)} | {prior[t]['gpu_span_us_per_round']/1000:.6f} | {prior[t]['rounds']} |")
(R/'BREAK-EVEN.md').write_text('\n'.join(a)+'\n')
print('\n'.join(a[:9]))
