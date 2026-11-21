
import numpy as np, json, sys, os
OUT='/home/avifenesh/.cache/inkling2mini/review-1396/out'
meta=json.load(open(OUT+'/meta.json'))
def dec(code):
    E=(code>>3)&15; m=code&7; s=-1 if code&0x80 else 1
    if E==0: return s*m, -9, -6          # mant, exp of lsb, field exponent
    return s*(8+m), E-7-3, E-7
def model(av, ae, wv, we, K, guard=25, zero_mode='skip', norm=False):
    acc_m, acc_e = 0, 0   # acc = acc_m * 2^acc_e exactly (fp32 value)
    for s0 in range(0, K, 32):
        sa = int(ae[s0//32]) - 127 if ae[s0//32] else -127
        sw = int(we[s0//32]) - 127 if we[s0//32] else -127
        prods=[]; emax=None
        for i in range(s0, s0+32):
            ma, la, fa = dec(int(av[i])); mw, lw, fw = dec(int(wv[i]))
            if norm:
                # normalised exponent of the value
                import math
                fa = (math.floor(math.log2(abs(ma)))+la) if ma else None
                fw = (math.floor(math.log2(abs(mw)))+lw) if mw else None
            p = ma*mw; pe = la+lw+sa+sw
            prods.append((p,pe))
            nz = (ma!=0 and mw!=0)
            if nz or zero_mode=='count':
                if fa is None or fw is None: continue
                e = fa+fw+sa+sw
                emax = e if emax is None else max(emax,e)
        if acc_m!=0:
            ea = acc_m.bit_length()-1+acc_e if acc_m>0 else (-acc_m).bit_length()-1+acc_e
            emax = ea if emax is None else max(emax, ea)
        if emax is None: continue
        sh = emax-guard
        def tr(m,e):
            # trunc(m*2^e / 2^sh) toward zero
            d=e-sh
            if d>=0: return m<<d
            a=abs(m)>>(-d); return a if m>=0 else -a
        tot = tr(acc_m,acc_e) + sum(tr(p,pe) for p,pe in prods)
        mag=abs(tot); sc=sh; b=mag.bit_length()
        if b>24: mag >>= (b-24); sc += b-24
        acc_m = -mag if tot<0 else mag; acc_e = sc
    return float(acc_m)*2.0**acc_e if acc_m else 0.0
res={}
for name in sorted(os.listdir(OUT)):
    if not name.startswith('g8_'): continue
    d=OUT+'/'+name; m=meta.get(name) or {}
    N=m.get('n',32); K=m.get('k'); 
    act=np.fromfile(d+'/act.bin',np.uint8); actsc=np.fromfile(d+'/actsc.bin',np.uint8)
    w=np.fromfile(d+'/w.bin',np.uint8); wsc=np.fromfile(d+'/wsc.bin',np.uint8); ex=np.fromfile(d+'/expert.bin',np.uint32)
    if K is None: K = len(act)//len(ex)
    R=len(act)//K
    acc=np.fromfile(d+'/acc.nb1w0',np.float32).reshape(R,N)
    grouped = len(ex) - (m.get('extra_rows',0) or 0)
    cnt={'field_skip':0,'field_count':0,'norm_skip':0}; tot=0
    for r in range(grouped):
        e=int(ex[r]); av=act[r*K:(r+1)*K]; ae=actsc[r*K//32:(r+1)*K//32]
        for c in range(N):
            row=e*N+c; wv=w[row*K:(row+1)*K]; we=wsc[row*K//32:(row+1)*K//32]
            got=float(acc[r,c]); tot+=1
            for k,(zm,nm) in {'field_skip':('skip',False),'field_count':('count',False),'norm_skip':('skip',True)}.items():
                v=np.float32(model(av,ae,wv,we,K,zero_mode=zm,norm=nm))
                if v.tobytes()!=np.float32(got).tobytes(): cnt[k]+=1
    res[name]=(tot,cnt); print(name,tot,cnt,flush=True)
