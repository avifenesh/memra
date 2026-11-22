
import json,statistics as s,sys,glob
d=sys.argv[1]
print('class\trun\tplain\tverify rows 1..8 (median over 12 prompts, ms)\tper extra row (LSQ rows 2..8)\tdraft K1..7')
for cls in ['declared','nvfp4-rig']:
  for rep in [1,2]:
    rs=[json.loads(l)['bench'] for l in open(f'{d}/bench-{cls}-r{rep}.jsonl')]
    v=[s.median(r['verify_ms_by_rows'][i] for r in rs) for i in range(8)]
    dr=[s.median(r['draft_ms_by_k'][i] for r in rs) for i in range(7)]
    xs=list(range(2,9)); ys=v[1:]
    mx=s.mean(xs); my=s.mean(ys); slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
    print(f"{cls}\t{rep}\t{s.median(r['plain_step_ms'] for r in rs):.2f}\t{' '.join(f'{x:.1f}' for x in v)}\t{slope:.2f}\t{' '.join(f'{x:.2f}' for x in dr)}")
