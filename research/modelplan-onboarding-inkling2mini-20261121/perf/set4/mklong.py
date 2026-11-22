import json
docs=json.load(open('/home/avifenesh/.cache/inkling2mini/klsens/corpus/corpus.json'))
codey=[d for d in docs if d.get('cat') in ('code','doc','docchat_code','docchat_doc')]
multi=[d for d in docs if d.get('cat') in ('multi','docchat_multi','chat','tools')]
def build(name, src, target=8448):
    ids=[]; samp=[]; used=[]
    for d in src:
        if len(ids)>=target: break
        ids+=d['ids']; samp+=d.get('sampled',[0]*len(d['ids'])); used.append(d['name'])
    ids=ids[:target]; samp=samp[:target]
    return {'name':name,'ids':ids,'sampled':samp,'parts':used}
a=build('L1_code_docs', codey); b=build('L2_multi_chat', multi)
for x in (a,b): print(x['name'], len(x['ids']), len(x['parts']), sum(x['sampled']))
json.dump([a],open('/home/avifenesh/.cache/inkling2mini/perf/corpus-L1.json','w'))
json.dump([b],open('/home/avifenesh/.cache/inkling2mini/perf/corpus-L2.json','w'))
