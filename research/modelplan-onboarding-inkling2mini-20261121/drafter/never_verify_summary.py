
import json,sys,collections
d=sys.argv[1]
for cls in ['declared','nvfp4-rig']:
  for rep in [1,2]:
    agg=collections.defaultdict(lambda:[0.0,0])
    for l in open(f'{d}/{cls}-r{rep}.jsonl'):
        r=json.loads(l)
        m='plain' if r['mode']=='plain' else ('never' if r['k']==99 else 'adapt')
        for kk in (r['kind'],'all'):
            a=agg[(m,kk)]; a[0]+=r['decode_s']; a[1]+=r['generated']-1
    out=[]
    for kk in ['chat','code','agentic','all']:
        p=agg[('plain',kk)]; n=agg[('never',kk)]; a=agg[('adapt',kk)]
        pr=p[1]/p[0]; out.append(f"{kk}: plain {pr:.1f} never {n[1]/n[0]/pr:.3f} adapt {a[1]/a[0]/pr:.3f}")
    print(cls,rep,' | '.join(out))
