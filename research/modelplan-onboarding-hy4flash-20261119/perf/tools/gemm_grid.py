import csv, statistics, re, sys
from collections import defaultdict
rows=[]
for r in csv.DictReader(open(sys.argv[1])):
    rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"], r.get("GrdX"), r.get("GrdY"), r.get("GrdZ")))
rows.sort()
st=[i for i,r in enumerate(rows) if "r_embed_kernel" in r[2]]
calls=[rows[st[i]:st[i+1]] for i in range(len(st)-1)]
dec=calls[3:]
agg=defaultdict(list)
for c in dec:
    per=defaultdict(lambda:[0,0])
    for r in c:
        if "gemm" in r[2] or "gemv" in r[2]:
            k=re.sub(r"\(.*","",r[2])[:40]+f" {r[3]}x{r[4]}x{r[5]}"
            per[k][0]+=r[1]; per[k][1]+=1
    for k,v in per.items(): agg[k].append(v)
tot=0
for k,v in sorted(agg.items(), key=lambda kv:-statistics.median([x[0] for x in kv[1]])):
    t=statistics.median([x[0] for x in v]); n=statistics.median([x[1] for x in v]); tot+=t
    print(f"{k:60s} {t/1e6:8.3f} ms {n:4.0f}x  {t/n/1e3:8.1f} us each")
print(tot/1e6)
