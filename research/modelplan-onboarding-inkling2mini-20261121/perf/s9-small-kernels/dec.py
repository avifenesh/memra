import sqlite3,sys,collections
db=sqlite3.connect(sys.argv[1]); ntok=int(sys.argv[2])
rows=db.execute('select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start').fetchall()
# decode tokens: find the last ntok occurrences of the unembed / last kernel per token; approximate by taking the tail after the last big prefill sgemm
last=max(i for i,r in enumerate(rows) if 'sgemm' in r[2])
tail=rows[last+1:]
span=(tail[-1][1]-tail[0][0])/1e6
agg=collections.defaultdict(lambda:[0,0.0,set()])
for r in tail:
    a=agg[r[2]]; a[0]+=1; a[1]+=(r[1]-r[0])/1e3; a[2].add((r[3],r[4],r[5],r[6]))
print('decode kernels',len(tail),'span ms',round(span,1))
tot=sum(a[1] for a in agg.values())
for k,a in sorted(agg.items(),key=lambda x:-x[1][1])[:16]:
    print('%-34s n=%5d total=%8.1f us  avg=%7.1f  grids=%s'%(k[:34],a[0],a[1],a[1]/a[0],sorted(a[2])[:3]))
print('kernel total ms',tot/1e3)
