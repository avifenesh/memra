import json,random,sys
random.seed(7)
for f in sys.argv[1:]:
    j=json.load(open(f))
    for a in j['arms']:
        m=a['arrays']; n=len(m['kl'])
        d=[m['nll_arm'][i]-m['nll_ref'][i] for i in range(n) if m['has_label'][i]>0]
        B=64; blocks=[d[i:i+B] for i in range(0,len(d),B)]
        mean=sum(d)/len(d); bs=[]
        for _ in range(2000):
            s=[x for _ in blocks for x in random.choice(blocks)]
            bs.append(sum(s)/len(s))
        bs.sort()
        print(f.split('/')[-1], a['spec'], 'dnll=%+.3e ci=[%+.3e, %+.3e]'%(mean, bs[50], bs[1949]), 'n=%d'%len(d))
