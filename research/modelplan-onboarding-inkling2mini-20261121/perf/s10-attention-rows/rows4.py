import sqlite3,collections,sys,os
P=os.environ.get('PR')
for n in sys.argv[1:]:
    db=sqlite3.connect(P+n+'.sqlite')
    rows=db.execute("select k.start,k.end,k.gridX,k.registersPerThread,k.dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where s.value like '%gemv_rows%' order by k.start").fetchall()
    by=collections.defaultdict(list)
    for a,b,g,rg,sm in rows: by[(g,rg,sm)].append((b-a)/1e3)
    print('==',n)
    for k,v in sorted(by.items()):
        v.sort(); print('  grid=%d regs=%d smem=%d n=%d median=%.1f us'%(k[0],k[1],k[2],len(v),v[len(v)//2]))
