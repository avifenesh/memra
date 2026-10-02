#!/usr/bin/env python3
"""Compare captured token identity and raw top-five values, without a quality claim."""
import argparse,json,pathlib
ap=argparse.ArgumentParser();ap.add_argument('cell');a=ap.parse_args()
p=pathlib.Path(a.cell)
def rows(name):return [json.loads(x) for x in (p/name).read_text().splitlines()]
result=[]
for arm, mode in [('http','aligned'),('http','split4'),('http-unsplit','aligned'),('http-unsplit','output-limit'),('http-unsplit','common-init')]:
    http=rows(arm+'.jsonl');logits=rows(arm+'-logits.jsonl')
    assert len(http)==len(logits)>0
    direct=rows(mode+'.jsonl');assert len(direct)==len(http)
    for h,l,d in zip(http,logits,direct):
        assert h['i']==d['i']
        assert h['response']['tokens'] == [l['top5'][0]['id']], 'HTTP sampler output differs from recorded raw argmax'
        assert h['response']['tokens_predicted'] == 1, 'HTTP generated the wrong output count'
        ht=h['tokens']['tokens'];dt=d['tokens']
        lt={x['id']:x['logit'] for x in l['top5']};rt={x['id']:x['logit'] for x in d['top5']}
        common=lt.keys()&rt.keys()
        row=dict(i=h['i'],http_arm=arm,mode=mode,tokens_equal=ht==dt,n_tokens=len(ht),http_top1=l['top5'][0]['id'],
                 api_top1=d['top5'][0]['id'],top5_exact=l['top5']==d['top5'],top5_ids_equal=lt.keys()==rt.keys(),
                 max_shared_top5_abs_diff=max(abs(lt[t]-rt[t]) for t in common),shared_top5_count=len(common))
        assert row['tokens_equal'],row
        result.append(row)
print(json.dumps(result,indent=2))
