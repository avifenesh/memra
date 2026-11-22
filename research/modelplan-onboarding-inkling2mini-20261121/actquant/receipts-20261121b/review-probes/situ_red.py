
import numpy as np, math
d='/home/avifenesh/.cache/inkling2mini/review-1396/out/situ_g1'
gu=np.fromfile(d+'/x.bin',np.uint16); K=4352; gu=gu.reshape(K,2*K)
def f(b): return (b.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
g=f(gu[:,0::2]); u=f(gu[:,1::2])
G=4*np.tanh(g/4)*(1/(1+np.exp(-g))); U=25*np.tanh(u/25)
z=G*U
r=z.astype(np.float32); rd=r.astype(np.float64); b=r.view(np.uint32).copy()
inex=(rd!=z)&np.isfinite(r); tw=np.abs(rd)>np.abs(z)
b[inex&tw]-=1; b[inex]|=1
b=b.astype(np.uint64); y=((b+0x7fff+((b>>16)&1))>>16).astype(np.uint16)
for tag in ('lib','apx'):
    dev=np.fromfile(d+'/y.'+tag,np.uint16).reshape(K,K)
    print(tag,'old kernel vs the new f64 reference: differ', int((dev!=y).sum()),'of',y.size)
