#!/usr/bin/env python3
"""PDL-on trace: per-kernel exclusive time a decode step (end - max(prev end, start)). usage: excl.py A.csv [B.csv]"""
import csv,re,statistics,sys
from collections import defaultdict
def load(path):
    rows=[]
    for r in csv.DictReader(open(path)):
        rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), re.sub(r"\(.*","",r["Name"]).replace("void ","")[:30], r.get("GrdX"), r.get("GrdY"), r.get("GrdZ")))
    rows.sort()
    st=[i for i,r in enumerate(rows) if "r_embed_kernel" in r[2]]
    agg=defaultdict(list); cnt=defaultdict(list)
    for ci in range(3,len(st)-1):
        c=rows[st[ci]:st[ci+1]]
        per=defaultdict(int); pc=defaultdict(int); pe=c[0][0]
        for r in c:
            e=r[0]+r[1]
            k=r[2]+(f" {r[3]}x{r[4]}x{r[5]}" if 'gem' in r[2] else '')
            per[k]+=max(0,e-max(pe,r[0])); per['~idle']+=max(0,r[0]-pe); pc[k]+=1
            pe=max(pe,e)
        for k,v in per.items(): agg[k].append(v); cnt[k].append(pc[k])
    return {k:statistics.median(v)/1e6 for k,v in agg.items()}, {k:statistics.median(v) for k,v in cnt.items()}
a,ca=load(sys.argv[1])
b,cb=load(sys.argv[2]) if len(sys.argv)>2 else ({},{})
keys=sorted(set(a)|set(b), key=lambda k:-max(a.get(k,0),b.get(k,0)))
for k in keys[:int(sys.argv[3]) if len(sys.argv)>3 else 45]:
    print(f"{k:44s} {a.get(k,0):7.3f} {b.get(k,0):7.3f}  n {ca.get(k,0):4.0f} {cb.get(k,0):4.0f}")
print(f"{'total':44s} {sum(a.values()):7.3f} {sum(b.values()):7.3f}")
