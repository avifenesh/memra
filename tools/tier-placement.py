#!/usr/bin/env python3
"""07 §3 opaque byte census arithmetic, NOT a model adapter or admission decision.

Both PP candidates; physical owners are explicit, never KV/TP. Record-byte changes
are sizing diagnostics only, never a supported-format substitution. Decimal GB.
"""
import argparse
import json

CAPACITY = 96_000_000_000
CLASSES = ('routed bank', 'attention/indexer ESTIMATE', 'shared expert', 'norm/router ESTIMATE',
           'host-table projections', 'embedding/head', 'draft module', 'vision reservation')
SPLITS = ((10,10,10,10), (9,11,9,11))


def weight_table(layers):
    return [[n*7_219_445_760,
             n*126_743_040 + (2,1,3,2)[i]*10_229_376,
             n*35_424_000, n*7_888_088 + (40_960 if i==3 else 0),
             157_521_920 if i<2 else 0,
             1_323_827_200 if i in (0,3) else 0,
             7_932_874_632 if i==3 else 0, 970_506_240 if i==0 else 0]
            for i,n in enumerate(layers)]


def row(layers, record_bytes, context, requests, reserve=0):
    if record_bytes <= 0 or context <= 0 or requests <= 0 or reserve < 0:
        raise ValueError('invalid geometry')
    weights = [sum(classes) for classes in weight_table(layers)]
    # Owner streams [2,8,14,20], first three half-rate; no hidden replicas.
    records = [2*(context//2), context//2, context, 0]
    floors = [(n+(3 if i==3 else 0))*128*528 for i,n in enumerate(layers)]
    global_kv = [requests*n*record_bytes for n in records]
    kv = [g+requests*f for g,f in zip(global_kv,floors)]
    remaining = [CAPACITY-w-k-reserve for w,k in zip(weights,kv)]
    return {'record_bytes':record_bytes,'context_tokens':context,'requests':requests,
            'global_bytes':sum(global_kv),'kv_bytes':kv,'remaining_bytes':remaining,
            'minimum_remaining_bytes':min(remaining)}


def frontier(layers, record_bytes, reserve=0):
    # Exact monotone arithmetic, including floor(T/2); no linear approximation of odd tails.
    lo, hi = 1, 1
    while row(layers,record_bytes,1_048_576,hi,reserve)['minimum_remaining_bytes'] >= 0:
        hi *= 2
    while lo < hi:
        mid=(lo+hi)//2
        if row(layers,record_bytes,1_048_576,mid,reserve)['minimum_remaining_bytes'] < 0: hi=mid
        else: lo=mid+1
    n=lo
    r=row(layers,record_bytes,1_048_576,n,reserve)
    device=next(i for i,v in enumerate(r['remaining_bytes']) if v<0)
    lo,hi=1,1_048_576
    while lo<hi:
        mid=(lo+hi)//2
        if row(layers,record_bytes,mid,n,reserve)['minimum_remaining_bytes']<0: hi=mid
        else: lo=mid+1
    return {'first_N':n,'device':device,'first_context':lo,'first_8192_step':((lo+8191)//8192)*8192}


def tables(record_bytes=None):
    modes = [record_bytes] if record_bytes is not None else [652,356]
    if any(type(b) is not int or b <= 0 for b in modes): raise ValueError('positive --record-bytes required')
    result=[]
    for layers in SPLITS:
        result.append({'split':layers,'class_names':CLASSES,'weight_classes_per_device':weight_table(layers),
                       'weights':[sum(c) for c in weight_table(layers)],
                       'host_bytes':202_758_032_400+32_000_000_000+8_000_000_000,
                       'grid':[row(layers,b,t,n) for b in modes for t in (131_072,1_048_576) for n in (1,4,16)],
                       'frontiers':[{'record_bytes':b,'reserve_per_device':r,**frontier(layers,b,r)} for b in modes for r in (0,4_000_000_000)]})
    return {'schema_version':1,'status':'ESTIMATE: payload arithmetic, not allocated peaks or admission',
            'excluded':'Unmeasured scratch, staging, loader peaks, candidate buffers, compression tails, graphs; no hidden full-cache replicas',
            'source':'07 §3 ALLOC-4 census rules; attention/indexer residual and norm/router allocation are estimates',
            'candidates':result}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record-bytes',type=int)
    p.add_argument('--json',action='store_true')
    a=p.parse_args()
    report=tables(a.record_bytes)
    if a.json:
        print(json.dumps(report,indent=2)); return
    print(report['status']); print(report['source']); print('Excluded: '+report['excluded'])
    for c in report['candidates']:
        print('\nPP split '+ '/'.join(map(str,c['split'])))
        print('| Weight class (bytes) | GPU0 | GPU1 | GPU2 | GPU3 | Total |\n|---|---:|---:|---:|---:|---:|')
        for i,name in enumerate(CLASSES):
            values=[w[i] for w in c['weight_classes_per_device']]
            print('| '+name+' | '+' | '.join(map(str,values+[sum(values)]))+' |')
        print('| Total | '+' | '.join(map(str,c['weights']+[sum(c['weights'])]))+' |')
        print('| Payload headroom | '+' | '.join(map(str,[CAPACITY-w for w in c['weights']]))+' | |')
        print(f"Host envelope ESTIMATE: {c['host_bytes']} B (one table copy +32GB OS/loader +8GB staging)")
        print('| B/record | Context | N | Global GB | GPU0 KV GB | GPU1 KV GB | GPU2 KV GB | GPU3 KV GB | Min remaining GB |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in c['grid']:
            print('| '+' | '.join(map(str,[r['record_bytes'],r['context_tokens'],r['requests']]))+' | '+' | '.join(f'{v/1e9:.6f}' for v in [r['global_bytes'],*r['kv_bytes'],r['minimum_remaining_bytes']])+' |')
        print('| B/record | Reserve/card B | First N at 1M | Device | First context | First8192-step |\n|---:|---:|---:|---:|---:|---:|')
        for f in c['frontiers']:
            print('| '+' | '.join(str(f[k]) for k in ('record_bytes','reserve_per_device','first_N','device','first_context','first_8192_step'))+' |')


if __name__=='__main__':
    main()
