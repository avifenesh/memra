import csv,sys,collections
rows=[r for r in csv.reader(open(sys.argv[1])) if r]
i0=[i for i,r in enumerate(rows) if "Kernel Name" in r][0]; rows=rows[i0:]; h=rows[0]; idx={k:i for i,k in enumerate(h)}
agg=collections.OrderedDict()
for r in rows[2:]:
    if len(r)<len(h): continue
    name=r[idx['Kernel Name']].split('(')[0].replace('<unnamed>::','')[:34]
    def g(k):
        try: return float(r[idx[k]].replace(',',''))
        except: return float('nan')
    agg.setdefault(name,[]).append([g(k) for k in ['gpu__time_duration.sum','sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','launch__grid_size','launch__block_size']])
for k,v in agg.items():
    n=len(v); m=lambda i: sum(x[i] for x in v)/n
    print(f"{k:34s} n={n:3d} t={m(0):8.1f}us fp64pipe%={m(1):5.1f} sm%={m(2):5.1f} warps%={m(3):5.1f} grid={m(4):7.0f} blk={m(5):4.0f}")
