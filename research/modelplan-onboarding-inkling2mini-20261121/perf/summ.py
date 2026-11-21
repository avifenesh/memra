import json,sys,csv,statistics,datetime
d=sys.argv[1]
log=open(sys.argv[2]).read().splitlines() if len(sys.argv)>2 else []
marks={}
for l in log:
    if l.startswith('MARK '):
        _,name,kind,t=l.split()
        marks.setdefault(name,{})[kind]=t
rows=list(csv.reader(open(d+'/smi.csv')))[1:]
def tsec(s):
    h,m,x=s.split(':'); return int(h)*3600+int(m)*60+float(x)
smi=[]
for r in rows:
    try:
        ts=r[0].strip().split(' ')[1]; smi.append((tsec(ts),r[1].strip(),int(r[3].split()[0]),float(r[4].split()[0]),int(r[2].split()[0])))
    except Exception: pass
res={}
for name,m in sorted(marks.items()):
    arm,i=name.rsplit('-',1)
    g=json.load(open(f"{d}/gen-{arm}-{i}.json"))['gen']
    a,b=tsec(m['start']),tsec(m['end'])
    loaded=[s for s in smi if a<=s[0]<=b and s[3]>60]
    bad=[s for s in loaded if s[1]=='P2' or s[2]==11001]
    sm=[s[4] for s in loaded]
    res.setdefault(arm,[]).append(dict(tok_s=round(g['decode_tok_s'],3),p50_ms=round(g['decode_step_p50_ms'],3),launches=g['launches_per_token'],ids=g['ids'],loaded=len(loaded),void=len(bad),sm=(min(sm),max(sm)) if sm else None))
out={}
for arm,v in res.items():
    ok=[x for x in v if x['void']==0]
    out[arm]=dict(takes=v,median_tok_s=statistics.median([x['tok_s'] for x in ok]) if ok else None)
ids={arm:[tuple(x['ids']) for x in v] for arm,v in res.items()}
allids=set(i for v in ids.values() for i in v)
out['greedy_ids_equal_across_all_takes']=len(allids)==1
out['ntokens']=len(next(iter(allids)))
print(json.dumps(out,indent=1))
