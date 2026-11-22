import sqlite3,collections,sys
import os; P=os.environ.get('PR','/home/avifenesh/.cache/inkling2mini/perf/prof-rows/')
for n in sys.argv[1:]:
    db=sqlite3.connect(P+n+'.sqlite')
    rows=db.execute('select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start').fetchall()
    embeds=[i for i,r in enumerate(rows) if 'embed' in r[2]]
    if not embeds: print(n,'no embed', collections.Counter(r[2] for r in rows).most_common(3)); continue
    calls=[]
    for a,b in zip(embeds, embeds[1:]+[len(rows)]):
        seg=rows[a:b]; R=seg[0][3]
        calls.append((R, sum((r[1]-r[0]) for r in seg)/1e3, (seg[-1][1]-seg[0][0])/1e3, seg))
    by=collections.defaultdict(list)
    for R,k,w,seg in calls: by[R].append((k,w,seg))
    print('==',n, {R:len(v) for R,v in by.items()})
    for R,v in sorted(by.items()):
        ks=sorted(x[0] for x in v); ws=sorted(x[1] for x in v)
        agg=collections.Counter()
        for _,_,seg in v:
            for r in seg: agg[r[2]]+=(r[1]-r[0])/1e3/len(v)
        print('  R=%d kernel us/call median %.0f, wall %.0f; top: %s'%(R, ks[len(ks)//2], ws[len(ws)//2], ', '.join('%s %.0f'%(k[:22],t) for k,t in agg.most_common(8))))
