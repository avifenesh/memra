import json,sys,csv,statistics
d=sys.argv[1]; log=open(sys.argv[2]).read().splitlines(); pre=sys.argv[3] if len(sys.argv)>3 else 'gen-'
marks={}
for l in log:
    if l.startswith('MARK '):
        _,name,kind,t=l.split(); marks.setdefault(name,{})[kind]=t
def tsec(s):
    h,m,x=s.split(':'); return int(h)*3600+int(m)*60+float(x)
smi=[]
for r in list(csv.reader(open(d+'/smi.csv')))[1:]:
    try: smi.append((tsec(r[0].strip().split(' ')[1]),r[1].strip(),int(r[3].split()[0]),float(r[4].split()[0])))
    except Exception: pass
res={}
for name,m in marks.items():
    arm,i=name.rsplit('-',1)
    try: j=json.load(open(f"{d}/{pre}{arm}-{i}.json"))
    except Exception as e:
        res.setdefault(arm,[]).append(dict(error=str(e))); continue
    g=j.get('gen',{})
    a,b=tsec(m['start']),tsec(m['end'])
    loaded=[s for s in smi if a<=s[0]<=b and s[3]>60]
    bad=[s for s in loaded if s[1]=='P2' or s[2]==11001]
    res.setdefault(arm,[]).append(dict(tok_s=round(g.get('decode_tok_s',0),3),prefill_tok_s=round(g.get('prefill_tok_s',0),1),p50_ms=round(g.get('decode_step_p50_ms') or 0,3),launches=g.get('launches_per_token'),graph_steps=g.get('graph_steps'),ids=g.get('ids'),loaded=len(loaded),void=len(bad)))
out={}
for arm,v in sorted(res.items()):
    ok=[x for x in v if x.get('void',1)==0]
    out[arm]=dict(tok_s=[x.get('tok_s') for x in v],prefill_tok_s=[x.get('prefill_tok_s') for x in v],median_tok_s=statistics.median([x['tok_s'] for x in ok]) if ok else None,median_prefill_tok_s=statistics.median([x['prefill_tok_s'] for x in ok]) if ok else None,p50_ms=[x.get('p50_ms') for x in v],launches=v[0].get('launches'),voids=[x.get('void') for x in v],loaded=[x.get('loaded') for x in v])
ids=set(tuple(x['ids']) for v in res.values() for x in v if x.get('ids'))
out['greedy_ids_equal_across_all_takes']=len(ids)==1
print(json.dumps(out,indent=1))
