
import json,sys
def load(p):
    out={}
    for l in open(p):
        r=json.loads(l)
        kinds={}
        for q in r['prompts']:
            a=kinds.setdefault(q['kind'],[0,0]); a[0]+=q['accepted']; a[1]+=q['steps']
        tot=[sum(v[0] for v in kinds.values()), sum(v[1] for v in kinds.values())]
        eq=[q['equal_plain'] for q in r['prompts']]
        out[r['comp']]=dict(draft=r['draft_ms_by_k'], kinds={k:v[0]/max(v[1],1) for k,v in kinds.items()}, all=tot[0]/max(tot[1],1), eq=sum(1 for e in eq if e) if eq[0] is not None else None)
    return out
g=load(sys.argv[1]); s=load(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2]!='-' else {}
f=load(sys.argv[3]) if len(sys.argv)>3 else {}
base=g['none']; d0=base['draft'][-1]
print('draft ms at K=7, none:', round(d0,3), ' plain-equal greedy:', base['eq'])
print('comp\tdraft7_ms\tsaved_ms\tacc_all\td_acc\tchat\tcode\tagentic\tS_acc\tS_d\tfp8_saved\tfp8_d_acc')
sb=s.get('none'); 
for c,v in g.items():
    sv=s.get(c); fv=f.get(c)
    row=[c, round(v['draft'][-1],3), round(d0-v['draft'][-1],3), round(v['all'],3), round(v['all']-base['all'],3)]
    row+= [round(v['kinds'].get(k,0),3) for k in ('chat','code','agentic')]
    row+= [round(sv['all'],3), round(sv['all']-sb['all'],3)] if sv and sb else ['','']
    row+= [round(d0-fv['draft'][-1],3), round(fv['all']-base['all'],3)] if fv else ['','']
    print('\t'.join(map(str,row)))
