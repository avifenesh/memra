
# cut the 250 ms log by #MARK phases; flag loaded samples at P2 or memory 11001 MHz (void)
import csv, sys, re, os, datetime as dt, statistics as st
d = sys.argv[1]
rows = []
for r in list(csv.reader(open(os.path.join(d, 'smi.csv'))))[1:]:
    try:
        t = dt.datetime.strptime(r[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
        rows.append((t, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6]), r[7].strip(), r[8].strip()))
    except Exception:
        pass
marks = {}
for f in os.listdir(d):
    if not f.endswith('.txt'): continue
    for l in open(os.path.join(d, f)):
        m = re.match(r'#MARK ([0-9.]+) (\S+) (start|end)$', l.strip())
        if m: marks.setdefault(m.group(2), {})[m.group(3)] = float(m.group(1))
print('phase,samples,loaded,sm_med,sm_min,mem_med,mem_min,power_med,temp_max,pstates,void')
bad = 0
for name, v in sorted(marks.items(), key=lambda kv: kv[1].get('start', 0)):
    if 'start' not in v or 'end' not in v: continue
    s = [r for r in rows if v['start'] - 0.25 <= r[0] <= v['end'] + 0.25]
    ld = [r for r in s if r[5] > 0]
    void = any(r[7] == 'P2' or r[2] == 11001 for r in ld)
    bad += void
    if not s: print(name + ',0'); continue
    print(f"{name},{len(s)},{len(ld)},{st.median(r[1] for r in s):.0f},{min(r[1] for r in s):.0f},{st.median(r[2] for r in s):.0f},{min(r[2] for r in s):.0f},{st.median(r[3] for r in s):.1f},{max(r[4] for r in s):.0f},{'|'.join(sorted(set(r[7] for r in s)))},{'VOID' if void else 'ok'}")
print('void phases:', bad)
