#!/usr/bin/env python3
"""Full-vocabulary comparison. A different suppression mask is reported, never hidden."""
import argparse, hashlib, json, math, pathlib, struct
ap=argparse.ArgumentParser();ap.add_argument('reference');ap.add_argument('candidate');a=ap.parse_args()
def read(p):
    raw=pathlib.Path(p).read_bytes();assert len(raw)%4==0
    values=struct.unpack('<'+'f'*(len(raw)//4),raw)
    assert all(not math.isnan(v) and v!=math.inf for v in values)
    assert any(math.isfinite(v) for v in values)
    return raw,values
def logsum(values):
    m=max(values)
    return m+math.log(sum(math.exp(v-m) for v in values))
rb,r=read(a.reference);cb,c=read(a.candidate);assert len(r)==len(c)>0
rlog,clog=logsum(r),logsum(c)
missing=[i for i,(x,y) in enumerate(zip(r,c)) if math.isfinite(x) and y==-math.inf]
shared=[i for i,(x,y) in enumerate(zip(r,c)) if math.isfinite(x) and math.isfinite(y)]
assert shared
kl='Infinity' if missing else sum(math.exp(r[i]-rlog)*((r[i]-rlog)-(c[i]-clog)) for i in shared)
rshared=logsum([r[i] for i in shared]);cshared=logsum([c[i] for i in shared])
conditional_kl=sum(math.exp(r[i]-rshared)*((r[i]-rshared)-(c[i]-cshared)) for i in shared)
ri=max(range(len(r)),key=r.__getitem__);ci=max(range(len(c)),key=c.__getitem__)
d=[r[i]-c[i] for i in shared]
print(json.dumps(dict(n=len(r),byte_equal=rb==cb,reference_top1=ri,candidate_top1=ci,top1_equal=ri==ci,
    kl_reference_candidate=kl,reference_only_support_count=len(missing),
    reference_mass_suppressed_by_candidate=sum(math.exp(r[i]-rlog) for i in missing),
    candidate_only_support_count=sum(x==-math.inf and math.isfinite(y) for x,y in zip(r,c)),
    shared_support_count=len(shared),kl_conditioned_on_shared_support=conditional_kl,
    shared_support_max_abs=max(map(abs,d)),shared_support_rmse=math.sqrt(sum(x*x for x in d)/len(d)),
    reference_sha256=hashlib.sha256(rb).hexdigest(),candidate_sha256=hashlib.sha256(cb).hexdigest()),indent=2,allow_nan=False))
