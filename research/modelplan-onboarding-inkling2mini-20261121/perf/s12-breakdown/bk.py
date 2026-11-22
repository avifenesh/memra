import sqlite3,collections,sys,os
P='/home/avifenesh/.cache/inkling2mini/perf/prof-bk/'
n=sys.argv[1]; last=int(sys.argv[2])
db=sqlite3.connect(P+n+'.sqlite')
rows=db.execute('select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start').fetchall()
emb=[i for i,r in enumerate(rows) if 'embed' in r[2]]
calls=[(a,b) for a,b in zip(emb, emb[1:]+[len(rows)])]
dec=calls[-last-1:-1]  # skip the very last (may be partial)
agg=collections.Counter(); cnt=collections.Counter(); wall=0; busy=0
for a,b in dec:
    seg=rows[a:b]; wall+=(rows[b][0] if b<len(rows) else seg[-1][1])-seg[0][0]
    for r in seg: agg[r[2]]+=(r[1]-r[0]); cnt[r[2]]+=1; busy+=r[1]-r[0]
N=len(dec)
print('calls',N,'wall/token %.3f ms'%(wall/N/1e6),'kernel busy/token %.3f ms'%(busy/N/1e6),'gaps/token %.3f ms'%((wall-busy)/N/1e6),'launches/token %.0f'%(sum(cnt.values())/N))
for k,v in agg.most_common(30): print('%-36s %8.1f us  x%5.1f'%(k[:36], v/N/1e3, cnt[k]/N))
