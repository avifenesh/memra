import json,random,sys,glob,os
random.seed(7)
D=sys.argv[1]
def boot_seq(m, seqs):
    by={}
    for i in range(len(m['kl'])):
        if m['has_label'][i]>0: by.setdefault(int(m['seq'][i]),[]).append(m['nll_arm'][i]-m['nll_ref'][i])
    keys=list(by)
    if len(keys)==1:
        d=by[keys[0]]; B=64; units=[d[i:i+B] for i in range(0,len(d),B)]
    else: units=[by[k] for k in keys]
    allv=[x for u in units for x in u]; mean=sum(allv)/len(allv); bs=[]
    for _ in range(1000):
        s=[x for _ in units for x in random.choice(units)]; bs.append(sum(s)/len(s))
    bs.sort(); return mean,bs[25],bs[974]
for f in sorted(glob.glob(D+'/*.json')):
    j=json.load(open(f))
    for a in j['arms']:
        m=a['arrays']; n=len(m['kl'])
        rr=sum(m.get('route_rows',[0])); rd=sum(m.get('route_rows_differ',[0]))
        mean,lo,hi=boot_seq(m,None)
        print('%-9s %-28s kl=%.2e top1=%.5f flips/100k=%7.1f dnll=%+.2e [%+.2e, %+.2e]%s'%(os.path.basename(f)[:-5],a['spec'],sum(m['kl'])/n,sum(m['top1'])/n,1e5*rd/max(1,rr),mean,lo,hi,'  WORSE' if lo>0 else ''))
