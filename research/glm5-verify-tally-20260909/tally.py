#!/usr/bin/env python3
"""Attribute kernels using NVTX host ranges and CUDA launch correlation, never wall overlap."""
import argparse,bisect,collections,json,sqlite3
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
c=sqlite3.connect(a.sqlite);c.row_factory=sqlite3.Row
names=dict(c.execute('SELECT id,value FROM StringIds'))
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
ranges=[]
for row in c.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
    d=dict(row);name=d.get('text') or names.get(d.get('textId'),'')
    if name.startswith('glm5-'):
        d['name']=name;ranges.append(d)
assert ranges,'No GLM NVTX ranges'
verify=sorted([r for r in ranges if r['name'].startswith('glm5-verify:')],key=lambda r:r['start'])
layers=sorted([r for r in ranges if r['name'].startswith('glm5-layer:')],key=lambda r:r['start'])
weights=sorted([r for r in ranges if r['name'].startswith('glm5-weight:')],key=lambda r:r['start'])
def fields(r):
    return {k:int(v) for k,v in (s.split('=') for s in r['name'].split(':')[1:])}
def lookup(rs, starts, stamp, tid):
    i=bisect.bisect_right(starts,stamp)-1
    if i>=0 and stamp<=rs[i]['end'] and tid==rs[i]['globalTid']:
        return rs[i]
vs=[r['start'] for r in verify];ls=[r['start'] for r in layers];ws=[r['start'] for r in weights]
apis=collections.defaultdict(list)
for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
    if table in tables:
        for r in c.execute(f'SELECT * FROM {table}'):
            apis[r['correlationId']].append(dict(r))
rounds={i:dict(fields(r),nvtx_host_us=(r['end']-r['start'])/1000,kernels=[],layers=[],copies=[]) for i,r in enumerate(verify)}
ids={id(r):i for i,r in enumerate(verify)}
for l in layers:
    r=lookup(verify,vs,l['start'],l['globalTid'])
    if r:rounds[ids[id(r)]]['layers'].append(fields(l)['il'])
for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
    candidates=apis[r['correlationId']]
    candidates=[x for x in candidates if 'Launch' in names.get(x.get('nameId'),'')]
    if not candidates:continue
    api=max(candidates,key=lambda x:x['start'])
    vr=lookup(verify,vs,api['start'],api['globalTid'])
    if not vr:continue
    ly=lookup(layers,ls,api['start'],api['globalTid'])
    wt=lookup(weights,ws,api['start'],api['globalTid'])
    name=names.get(r['shortName']) or names.get(r['demangledName'])
    rec=dict(name=name,start=r['start'],end=r['end'],us=(r['end']-r['start'])/1000,api_us=(api['end']-api['start'])/1000,layer=fields(ly)['il'] if ly else None,weight=fields(wt) if wt else None,grid=[r['gridX'],r['gridY'],r['gridZ']],block=[r['blockX'],r['blockY'],r['blockZ']])
    rounds[ids[id(vr)]]['kernels'].append(rec)
for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
    if table not in tables:
        continue
    for r in c.execute(f'SELECT * FROM {table} ORDER BY start'):
        candidates=apis[r['correlationId']]
        if not candidates:
            continue
        api=max(candidates,key=lambda x:x['start'])
        vr=lookup(verify,vs,api['start'],api['globalTid'])
        if vr:
            rounds[ids[id(vr)]]['copies'].append(dict(start=r['start'],end=r['end'],bytes=r['bytes'],kind=table))
buckets=collections.defaultdict(list)
for rr in rounds.values():
    assert sorted(rr['layers'])==list(range(45)),f'incomplete verify round: {rr["round"]}, {rr["layers"]}'
    assert rr['kernels']
    buckets[rr['t']].append(rr)
summary={}
for t,rs in sorted(buckets.items()):
    kernels=collections.defaultdict(list)
    gaps=[];spans=[];trunk=[];idle=[];copy_us=[]
    for rr in rs:
        kk=sorted(rr['kernels'],key=lambda k:k['start'])
        end=kk[0]['end'];gap=0
        for k in kk[1:]:
            gap+=max(0,k['start']-end);end=max(end,k['end'])
        gaps.append(gap/1000);spans.append((end-kk[0]['start'])/1000)
        intervals=sorted([(k['start'],k['end']) for k in kk]+[(k['start'],k['end']) for k in rr['copies']])
        occupied=0;lo,hi=intervals[0]
        for start,end in intervals[1:]:
            if start>hi:
                occupied+=hi-lo;lo=start
            hi=max(hi,end)
        occupied+=hi-lo
        idle.append((hi-intervals[0][0]-occupied)/1000)
        copy_us.append(sum((k['end']-k['start'])/1000 for k in rr['copies']))
        trunk.append(sum(k['layer'] is not None for k in kk))
        for k in kk:kernels[k['name']].append(k)
    rows=[]
    for name,ks in kernels.items():
        total=sum(k['us'] for k in ks);row=dict(kernel=name,launches_per_round=len(ks)/len(rs),total_us_per_round=total/len(rs),mean_us=total/len(ks))
        # Logical streamed weight bytes. qmatvec batched kernels read each weight once,
        # non-batched grid.y repeats the plane for each input row. This is effective BW,
        # not hardware-counter HBM traffic (L2 hits and repeated sectors are unknown).
        if name.startswith('qmatvec_') and all(k['weight'] for k in ks):
            b=sum(k['weight']['bytes']*(1 if '_b' in name else k['grid'][1]) for k in ks)
            row.update(weight_bytes_per_round=b/len(rs),effective_GBs=b/total/1000,bw_pct=b/total/1000/8000*100,saving_to_70pct_us_per_round=max(0,total/len(rs)-b/len(rs)/5600/1000))
        if name in ('moe_gate_up_preclamp8_q8_rows_ilp', 'moe_down8_fma_q8_rows_ilp'):
            planes = 2 if 'gate_up' in name else 1
            b = len(ks) * planes * t * 8 * 4096 * 2048 * 36 // 64
            row.update(weight_bytes_per_round=b/len(rs), effective_GBs=b/total/1000,
                       bw_pct=b/total/1000/8000*100,
                       saving_to_70pct_us_per_round=max(0,total/len(rs)-b/len(rs)/5600/1000),
                       byte_model='logical top8 visits; interleaved NVFP4, including repeat experts')
        if name == 'hc_mixes_gemv_f32':
            b = len(ks) * 24 * 16384 * 4
            row.update(weight_bytes_per_round=b/len(rs), effective_GBs=b/total/1000,
                       bw_pct=b/total/1000/8000*100,
                       saving_to_70pct_us_per_round=max(0,total/len(rs)-b/len(rs)/5600/1000),
                       byte_model='24 by 16384 f32, repeated per token row')
        rows.append(row)
    rows.sort(key=lambda x:-x['total_us_per_round'])
    summary[t]=dict(rounds=len(rs),total_launches_per_round=sum(len(r['kernels']) for r in rs)/len(rs),trunk_launches_per_layer=sum(trunk)/len(rs)/45,total_kernel_us_per_round=sum(r['total_us_per_round'] for r in rows),gpu_span_us_per_round=sum(spans)/len(rs),gpu_gaps_us_per_round=sum(gaps)/len(rs),device_idle_us_per_round=sum(idle)/len(rs),copy_us_per_round=sum(copy_us)/len(rs),nvtx_host_us_per_round=sum(r['nvtx_host_us'] for r in rs)/len(rs),ranked=rows)
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(a.out/'rounds.json').write_text(json.dumps(list(rounds.values()))+'\n')
print(json.dumps(summary,indent=2))
