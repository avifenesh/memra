import sqlite3,sys,collections
db=sqlite3.connect(sys.argv[1]); pat=sys.argv[2]
rows=db.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where s.value like ? order by k.start",('%'+pat+'%',)).fetchall()
d=collections.defaultdict(list)
for a,b,n,gx,gy,gz,bx in rows: d[(n[:34],gx,gy,gz,bx)].append((b-a)/1e3)
for k,v in sorted(d.items(), key=lambda kv:-sum(kv[1])):
    v.sort(); print(k, 'n=%d'%len(v), 'median %.2f us'%v[len(v)//2], 'total %.0f us'%sum(v))
