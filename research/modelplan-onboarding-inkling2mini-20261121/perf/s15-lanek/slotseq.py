import sqlite3,sys
db=sqlite3.connect(sys.argv[1]); tok=int(sys.argv[2]) if len(sys.argv)>2 else 20; slot=int(sys.argv[3]) if len(sys.argv)>3 else 10
rows=db.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
emb=[i for i,r in enumerate(rows) if 'embed' in r[2]]
a,b=emb[tok],emb[tok+1]
seg=rows[a:b]
# slots split at the w2 expert gemm: every second expert gemm
ex=[i for i,r in enumerate(seg) if 'expert_gemm' in r[2]]
s0=ex[2*slot-1]+1 if slot>0 else 0; s1=ex[2*slot+1]+1
prev=None
for r in seg[s0:s1]:
    gap=(r[0]-prev)/1e3 if prev else 0
    print('%-32s grid(%d,%d,%d)x%-4d st%d dur %6.2f us  gap %6.2f'%(r[2][:32],r[3],r[4],r[5],r[6],r[7],(r[1]-r[0])/1e3,gap)); prev=r[1]
print('slot wall %.1f us'%((seg[s1-1][1]-seg[s0][0])/1e3))
