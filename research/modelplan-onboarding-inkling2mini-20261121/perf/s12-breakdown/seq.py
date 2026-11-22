import sqlite3,sys
P='/home/avifenesh/.cache/inkling2mini/perf/prof-bk/'
db=sqlite3.connect(P+sys.argv[1]+'.sqlite')
rows=db.execute('select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start').fetchall()
emb=[i for i,r in enumerate(rows) if 'embed' in r[2]]
a=emb[-3]; b=emb[-2]
seg=rows[a:b]
# print first ~80 kernels (about 2-3 slots)
for r in seg[:int(sys.argv[2])]:
    print('%-32s %6.1f us grid=(%d,%d,%d) blk=%d gap_before=%.1f'%(r[2][:32],(r[1]-r[0])/1e3,r[3],r[4],r[5],r[6],0))
