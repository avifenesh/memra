#!/usr/bin/env python3
"""Last-row comparisons for reciprocal split controls; preserve actual callback scope."""
import argparse, hashlib, json, math, pathlib, struct
ap=argparse.ArgumentParser();ap.add_argument('mono');ap.add_argument('split');a=ap.parse_args()
def load(folder,chunk):
    folder=pathlib.Path(folder)
    rows=[json.loads(x) for x in (folder/'index.jsonl').read_text().splitlines()]
    out={}
    for row in rows:
        if row['chunk']!=chunk:continue
        key=row['i'],row['name']
        assert key not in out, f'duplicate trace stage {key}'
        raw=(folder/row['file']).read_bytes()
        assert len(raw)==row['width']*4
        out[key]=(struct.unpack('<'+'f'*row['width'],raw),hashlib.sha256(raw).hexdigest())
    return out
left,right=load(a.mono,0),load(a.split,1)
assert left.keys()==right.keys(),(left.keys()-right.keys(),right.keys()-left.keys())
result=[]
for key,(x,xhash) in left.items():
    y,yhash=right[key];assert len(x)==len(y)
    assert all(math.isfinite(v) for v in x+y)
    delta=[u-v for u,v in zip(x,y)]
    result.append(dict(i=key[0],stage=key[1],n=len(x),byte_equal=xhash==yhash,
                       max_abs=max(map(abs,delta)),rmse=math.sqrt(sum(v*v for v in delta)/len(x)),
                       relative_l2=math.sqrt(sum(v*v for v in delta)/max(sum(v*v for v in x),1e-30)),
                       mono_sha256=xhash,split_sha256=yhash))
print(json.dumps(result,indent=2))
