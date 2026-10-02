#!/usr/bin/env python3
"""Validate the finite component cell and aggregate times using the actual slot census."""
import argparse,csv,json,pathlib,statistics
ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--out',required=True);args=ap.parse_args()
p=pathlib.Path(args.run)
def table(name):
    with (p/name).open() as f:return list(csv.DictReader(f,delimiter='\t'))
census=table('census.tsv'); measurements=table('measurements.tsv'); checks=table('checks.tsv')
assert len(census)==10 and sum(int(x['slots']) for x in census)==400
assert sum(int(x['slots']) for x in census if x['input_group']=='true')==272
classes={x['class']:x for x in census};assert len(classes)==10
by={}
for x in measurements:
    key=(x['class'],int(x['m']),int(x['round']),x['arm'])
    assert key not in by and x['arm'] in ('int8','fp4')
    assert int(x['launches'])==64 and 0<=key[2]<10
    assert x['order']==('AB' if key[2]%2==0 else 'BA')
    assert int(x['k'])==int(classes[key[0]]['k']) and int(x['n'])==int(classes[key[0]]['n'])
    assert float(x['gpu_ms'])>0 and float(x['wall_ms'])>0
    by[key]=x
expected={(c,m,r,arm) for c in classes for m in [512,2048,4096] for r in range(10) for arm in ['int8','fp4']}
assert set(by)==expected and len(checks)==30
assert {(x['class'],int(x['m'])) for x in checks}=={(c,m) for c in classes for m in [512,2048,4096]}
for x in checks:
    assert x['repeat_int8']==x['repeat_fp4']==x['rp']=='true'
    assert int(x['bad_scale_rc'])==2902
    assert x['int8_sha256']!=x['fp4_sha256']
    for k in ['input_sha256','int8_sha256','fp4_sha256']:assert len(x[k])==64

per_class=[]
for m in [512,2048,4096]:
    for c,shape in classes.items():
        arms={arm:[float(by[c,m,r,arm]['gpu_ms']) for r in range(10)] for arm in ['int8','fp4']}
        stats={arm:dict(median=statistics.median(v),minimum=min(v),maximum=max(v),spread_percent=100*(max(v)-min(v))/statistics.median(v)) for arm,v in arms.items()}
        ratios=[arms['int8'][r]/arms['fp4'][r] for r in range(10)]
        per_class.append(dict(projection=c,rows=m,slots=int(shape['slots']),input_group=shape['input_group']=='true',k=int(shape['k']),n=int(shape['n']),gpu_ms=stats,
            ratio_of_medians=stats['int8']['median']/stats['fp4']['median'],paired_ratio_median=statistics.median(ratios),paired_ratio_min=min(ratios),paired_ratio_max=max(ratios)))
aggregates=[]
for m in [512,2048,4096]:
    for group in ['input_272','all_400']:
        selected=[x for x in per_class if x['rows']==m and (group=='all_400' or x['input_group'])]
        sums={arm:sum(x['slots']*x['gpu_ms'][arm]['median'] for x in selected) for arm in ['int8','fp4']}
        lower=sum(x['slots']*x['gpu_ms']['int8']['minimum'] for x in selected)/sum(x['slots']*x['gpu_ms']['fp4']['maximum'] for x in selected)
        upper=sum(x['slots']*x['gpu_ms']['int8']['maximum'] for x in selected)/sum(x['slots']*x['gpu_ms']['fp4']['minimum'] for x in selected)
        orders={}
        for order,rounds in [('AB',range(0,10,2)),('BA',range(1,10,2))]:
            totals={arm:sum(x['slots']*statistics.median(float(by[x['projection'],m,r,arm]['gpu_ms']) for r in rounds) for x in selected) for arm in ['int8','fp4']}
            orders[order]=totals['int8']/totals['fp4']
        aggregates.append(dict(rows=m,group=group,slot_count=sum(x['slots'] for x in selected),weighted_gpu_ms=sums,ratio=sums['int8']/sums['fp4'],observed_extreme_ratio_bounds=[lower,upper],order_ratios=orders))
macs={group:sum(int(x['slots'])*int(x['k'])*int(x['n']) for x in census if group=='all_400' or x['input_group']=='true') for group in ['input_272','all_400']}
result=dict(schema='a4-component-cost-v1',measurement_scope='Resident real weights and synthetic activations; quantization plus GEMM; no model quality or wall-prefill claim',samples=len(by),rounds_per_arm=10,rounds_per_order=5,launches_per_sample=64,macs_per_token=macs,input_mac_fraction=macs['input_272']/macs['all_400'],per_class=per_class,aggregates=aggregates)
pathlib.Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for x in aggregates:print(json.dumps(x))
