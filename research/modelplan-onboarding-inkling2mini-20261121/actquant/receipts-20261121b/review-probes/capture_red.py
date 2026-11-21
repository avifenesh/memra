
# the first round's dot_chain_trunc (normalised exponents) vs the field-exponent model on the committed
# device capture's W8A8 records
import sys, struct, numpy as np
sys.path.insert(0, '/home/avifenesh/.cache/inkling2mini/actq-fix')
exec(open('/home/avifenesh/.cache/inkling2mini/actq-fix/trunc_check.py').read().split('res={}')[0])
def parse(p):
    b = open(p, 'rb').read(); at = 8; n = struct.unpack('<I', b[at:at+4])[0]; at += 4; out = {}
    for _ in range(n):
        l = struct.unpack('<H', b[at:at+2])[0]; at += 2; nm = b[at:at+l].decode(); at += l
        nb = struct.unpack('<I', b[at:at+4])[0]; at += 4; rec = {}
        for _ in range(nb):
            l = struct.unpack('<H', b[at:at+2])[0]; at += 2; tg = b[at:at+l].decode(); at += l
            sz = struct.unpack('<I', b[at:at+4])[0]; at += 4; rec[tg] = b[at:at+sz]; at += sz
        out[nm] = rec
    return out
import torch
sys.path.insert(0, '/home/avifenesh/.cache/inkling2mini/wt-actq/research/modelplan-onboarding-inkling2mini-20261121/actquant')
import gen_goldens as G
g = parse(sys.argv[1]); c = parse(sys.argv[2])
for name in sorted(c):
    if not name.startswith('w8a8/'): continue
    r = g[name]; d = np.frombuffer(r['dims'], np.uint32); e, n, k, rows = map(int, d)
    x = G.from_bits16(np.frombuffer(r['x'], np.uint16)).reshape(rows, k)
    v, s = G.mxfp8_quantize(x); v = v.numpy(); s = s.numpy()
    w = np.frombuffer(r['w'], np.uint8); ws = np.frombuffer(r['wexps'], np.uint8); ex = np.frombuffer(r['expert'], np.uint32)
    acc = np.frombuffer(c[name]['acc'], np.float32).reshape(rows, n)
    bad = {'field': 0, 'normalised': 0}
    for i in range(rows):
        for col in range(n):
            row = int(ex[i]) * n + col
            for key, norm in (('field', False), ('normalised', True)):
                m = np.float32(model(v[i], s[i], w[row*k:(row+1)*k], ws[row*k//32:(row+1)*k//32], k, norm=norm))
                bad[key] += int(m.tobytes() != acc[i, col].tobytes())
    print(name, rows * n, 'accumulators; model != device:', bad, flush=True)
