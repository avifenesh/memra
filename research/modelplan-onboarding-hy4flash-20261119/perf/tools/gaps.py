import csv, sys, statistics, re
from collections import defaultdict
rows=[]
for r in csv.DictReader(open(sys.argv[1])):
    rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), re.sub(r"\(.*","",r["Name"])[:30]))
rows.sort()
st=[i for i,r in enumerate(rows) if "r_embed_kernel" in r[2]]
calls=[rows[st[i]:st[i+1]+1] for i in range(len(st)-1)]
agg=defaultdict(list)
for c in calls[3:]:
    for a,b in zip(c,c[1:]):
        g=b[0]-(a[0]+a[1])
        if g>30000: agg[(a[2],b[2])].append(g)
tot=0
for k,v in sorted(agg.items(), key=lambda kv:-sum(kv[1])):
    n=len(calls[3:])
    print(f"{k[0]:30s} -> {k[1]:30s} {sum(v)/n/1e6:7.3f} ms/step  n/step {len(v)/n:5.1f}  med {statistics.median(v)/1e3:7.1f} us")
    tot+=sum(v)/n
print("total gaps >30us per step ms", tot/1e6)
