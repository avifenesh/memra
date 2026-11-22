import json,sys,glob,os
for f in sorted(glob.glob('/home/avifenesh/.cache/inkling2mini/perf/set4/*.json')):
    j=json.load(open(f))
    for a in j['arms']:
        m=a['arrays']; kl=m['kl']; n=len(kl)
        rr=sum(m.get('route_rows',[0])); rd=sum(m.get('route_rows_differ',[0]))
        lab=[i for i in range(n) if m['has_label'][i]>0]
        dn=sum(m['nll_arm'][i]-m['nll_ref'][i] for i in lab)/max(1,len(lab))
        print('%-10s %-60s kl=%.3e top1=%.5f flips/100k=%.1f dnll=%+.2e'%(os.path.basename(f)[:-5], a['spec'], sum(kl)/n, sum(m['top1'])/n, 1e5*rd/max(1,rr), dn))
