#!/usr/bin/env python3
"""DAY83 section 0: the dispatch clock per prefetched block and per generated token in the generate phase (generate minus gate) and the window (window minus warm), medians over each clocked arm's 10 runs, from mirrored card cells. usage: card-clocks.py <label> <glob> [...]"""
CLOCK=re.compile(r"\[moe-cache\] dispatch-clock phase=(\w+) (.*)")
def read(path):
    ph={}
    for line in open(path,errors='replace'):
        m=CLOCK.search(line)
        if m: ph[m.group(1)]={k:int(v) for k,v in re.findall(r"(\w+)=(\d+)",m.group(2))}
    return ph
keys=['prefetch_ns','pf_demand_ns','pf_resident_ns','pf_retire_ns','pf_stage_ns','pf_reserve_ns','dispatch_ns','prefetch_issued']
for label,pat in [(a,b) for a,b in zip(sys.argv[1::2],sys.argv[2::2])]:
    rows=[]
    for f in sorted(glob.glob(pat)):
        ph=read(f)
        if 'generate' not in ph or 'gate' not in ph: continue
        d={k:ph['generate'][k]-ph['gate'][k] for k in keys}
        w={k:ph['window'][k]-ph['warm'][k] for k in keys} if 'window' in ph and 'warm' in ph else None
        rows.append((d,w))
    if not rows: print(label,'none'); continue
    med=lambda xs: statistics.median(xs)
    g={k:med([r[0][k] for r in rows]) for k in keys}
    iss=g['prefetch_issued']
    print(f"{label} N={len(rows)} generate: issued={iss:.0f} per token={iss/32:.1f} | per token us: "+" ".join(f"{k[:-3]}={g[k]/32/1000:.1f}" for k in keys[:-1])+" | per issued us: "+" ".join(f"{k[:-3]}={g[k]/iss/1000:.2f}" for k in keys[1:5]))
    if rows[0][1]:
        w={k:med([r[1][k] for r in rows]) for k in keys}; iw=w['prefetch_issued']
        print(f"{label} window: issued={iw:.0f} per token={iw/32:.1f} | per token us: "+" ".join(f"{k[:-3]}={w[k]/32/1000:.1f}" for k in keys[:-1]))
