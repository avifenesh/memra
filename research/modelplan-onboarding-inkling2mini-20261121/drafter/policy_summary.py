
import json,sys,collections,os
d=sys.argv[1]
for cls in ['declared','nvfp4-rig']:
  for pol in ['','-cautious']:
    for rep in [1,2]:
      f=f'{d}/{cls}{pol}-r{rep}.jsonl'
      if not os.path.exists(f): continue
      agg=collections.defaultdict(lambda:[0.0,0,0.0,0.0])
      for l in open(f):
        r=json.loads(l)
        m='plain' if r['mode']=='plain' else ('never' if r['k']==99 else 'adapt')
        for kk in (r['kind'],'all'):
            a=agg[(m,kk)]; a[0]+=r['decode_s']; a[1]+=r['generated']-1; a[2]+=r.get('plain_s',0); a[3]+=r.get('spec_s',0)
      out=[]
      for kk in ['chat','code','agentic','all']:
        p=agg[('plain',kk)]; n=agg[('never',kk)]; a=agg[('adapt',kk)]
        pr=p[1]/p[0]; out.append(f"{kk} {n[1]/n[0]/pr:.3f}/{a[1]/a[0]/pr:.3f}")
      a=agg[('adapt','chat')]
      print(f"{cls}{pol or '-default'} r{rep}: never/adapt vs plain: "+'  '.join(out)+f"  | chat adapt split plain_s {a[2]:.2f} spec_s {a[3]:.2f} loop {a[0]-a[2]-a[3]:.2f} of {a[0]:.2f}")
