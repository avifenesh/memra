import struct,sys,math
def load(p):
    d={}
    for line in open(p):
        f=line.rstrip('\n').split('\t')
        if f[0]!='stage': continue
        d[(f[1],int(f[2]))]=[struct.unpack('>f',bytes.fromhex(h))[0] for h in f[4].split(',')]
    return d
a=load(sys.argv[1]); b=load(sys.argv[2])
order=[]
for k in a:
    if k in b: order.append(k)
def key(k):
    st,l=k
    rank={'expand':0,'mixer':1,'attn':2,'router':3,'route':4,'routed':5,'ffn':6,'layer':7,'collapse':8}[st]
    return (l if l>=0 else (-1 if st=='expand' else 999), rank)
order.sort(key=key)
print(f"{'stage':9s} {'L':>3s} {'n':>6s} {'cos':>9s} {'|ref|':>10s} {'|eng|':>10s} {'ratio':>8s} {'maxabs':>10s}")
for k in order:
    x,y=a[k],b[k]
    if len(x)!=len(y): print(k,'LEN MISMATCH',len(x),len(y)); continue
    na=math.sqrt(sum(v*v for v in x)); nb=math.sqrt(sum(v*v for v in y))
    num=sum(p*q for p,q in zip(x,y))
    cos=num/(na*nb) if na>0 and nb>0 else float('nan')
    mx=max(abs(p-q) for p,q in zip(x,y))
    print(f"{k[0]:9s} {k[1]:3d} {len(x):6d} {cos:9.6f} {na:10.4g} {nb:10.4g} {nb/na if na>0 else 0:8.4f} {mx:10.4g}")
