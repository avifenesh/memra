
# W8A8 program comparison inputs: real layer 20-23 MXFP8 experts, real MoE inputs (lane E tape p2_window.actq_f32).
import json, struct, sys, os
import numpy as np, torch
sys.path.insert(0, '/home/avifenesh/.cache/inkling2mini/wt-actq/research/modelplan-onboarding-inkling2mini-20261121/actquant')
import gen_goldens as G
F32 = torch.float32
CK = '/home/avifenesh/models/inkling2-mini/model-00001-of-00001.safetensors'
TAPE = '/home/avifenesh/.cache/inkling2mini/oracle/tapes/p2_window.actq_f32'
OUT = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 24
def st(path):
    f = open(path, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); return f, h, 8 + n
def read(f, h, base, name, start=0, length=None):
    o = h[name]['data_offsets']; length = (o[1] - o[0] - start) if length is None else length
    f.seek(base + o[0] + start); return np.frombuffer(f.read(length), np.uint8).copy()
def tensor(path, name):
    f, h, base = st(path); d = h[name]; b = read(f, h, base, name)
    dt = {'F32': np.float32, 'I32': np.int32, 'I64': np.int64}[d['dtype']]
    return b.view(dt).reshape(d['shape'])
moe_in = None
f, h, base = st(CK)
def e4m3_f64(codes):
    c = codes.astype(np.int32); E = (c >> 3) & 15; m = c & 7
    mag = np.where(E == 0, m * 2.0**-9, (8 + m) * np.power(2.0, E - 10))
    return np.where(c & 0x80, -mag, mag)
def scale_f64(e):
    e = e.astype(np.int32); return np.where(e == 0, 2.0**-127, np.power(2.0, e - 127.0))
def bf16_exact(v):
    # bf16 bits of f64 v when v is exactly representable; flag otherwise
    t = torch.tensor(v, dtype=torch.float64)
    b = t.to(torch.bfloat16); ok = (b.to(torch.float64) == t)
    return b.view(torch.int16).numpy().view(np.uint16), ok.numpy()
pos_all = None
router_ids = tensor(TAPE + '/router.safetensors', 'route_ids'); router_w = tensor(TAPE + '/router.safetensors', 'route_weights')
fm, hm, bm = st(TAPE + '/moe_in.safetensors'); shp = hm['moe_in']['shape']
P = shp[1]; tok = np.linspace(8, P - 1, T).astype(int)
summary = {}
for L in (20, 21, 22, 23):
    slot = L
    rowbytes = P * 2304 * 4
    x = np.stack([read(fm, hm, bm, 'moe_in', (slot * P + p) * 2304 * 4, 2304 * 4).view(np.float32) for p in tok])
    xb = G.to_bf16(torch.tensor(x))
    ids = router_ids[slot, tok]; gam = router_w[slot, tok].astype(np.float32)
    distinct = sorted(set(ids.flatten().tolist())); remap = {e: i for i, e in enumerate(distinct)}
    for proj, K, N in (('w13', 2304, 2304), ('w2', 1152, 2304)):
        pre = 'model.llm.layers.%d.mlp.experts.%s' % (L, proj)
        W = np.concatenate([read(f, h, base, pre + '_weight', e * N * K, N * K) for e in distinct])
        WS = np.concatenate([read(f, h, base, pre + '_weight_scale', e * N * (K // 32), N * (K // 32)) for e in distinct])
        if proj == 'w13':
            act_rows = xb; expert = []; rows = []
            # one group per (token, expert): row = token
            groups = []
            for t in range(T):
                for j in range(8):
                    groups += [remap[int(ids[t, j])], 1, t] + [0] * 7
            R = T
        else:
            # w2 input: SiTU of the gate_up from the exact w13 sums, times the routing weight
            rowsl = []
            groups = []
            for t in range(T):
                v, s = G.mxfp8_quantize(xb[t:t + 1])
                a = (e4m3_f64(v.numpy()[0]) * np.repeat(scale_f64(s.numpy()[0]), 32))
                for j in range(8):
                    e = int(ids[t, j]); ei = remap[e]
                    w13 = read(f, h, base, 'model.llm.layers.%d.mlp.experts.w13_weight' % L, e * 2304 * 2304, 2304 * 2304).reshape(2304, 2304)
                    w13s = read(f, h, base, 'model.llm.layers.%d.mlp.experts.w13_weight_scale' % L, e * 2304 * 72, 2304 * 72).reshape(2304, 72)
                    wd = e4m3_f64(w13) * np.repeat(scale_f64(w13s), 32, axis=1)
                    gu = G.to_bf16(torch.tensor((wd @ a).astype(np.float32)))
                    y = G.situ(gu[None, :], torch.tensor([gam[t, j]], dtype=F32))
                    rowsl.append(y[0])
                    groups += [ei, 1, t * 8 + j] + [0] * 7
            act_rows = torch.stack(rowsl); R = act_rows.shape[0]
        v, s = G.mxfp8_quantize(act_rows)
        v = v.numpy(); s = s.numpy()
        ad = e4m3_f64(v) * np.repeat(scale_f64(s), 32, axis=1)          # [R][K] f64, exact
        wdeq = None
        d = os.path.join(OUT, 'L%d_%s' % (L, proj)); os.makedirs(d, exist_ok=True)
        v.tofile(d + '/act.bin'); s.tofile(d + '/actsc.bin'); np.zeros(R, np.float32).tofile(d + '/actamax.bin')
        W.tofile(d + '/w.bin'); WS.tofile(d + '/wsc.bin'); np.ones(len(distinct), np.float32).tofile(d + '/wamax.bin')
        g = np.array(groups, np.uint32); g.tofile(d + '/groups1.bin'); ng = len(groups) // 10
        abf, aok = bf16_exact(ad); abf.tofile(d + '/abf.bin')
        exact = np.zeros((ng, N)); mag = np.zeros((ng, N)); wbad = 0
        wbf_all = np.empty((len(distinct), N, K), np.uint16)
        for ei in range(len(distinct)):
            wd = e4m3_f64(W[ei * N * K:(ei + 1) * N * K].reshape(N, K)) * np.repeat(scale_f64(WS[ei * N * K // 32:(ei + 1) * N * K // 32].reshape(N, K // 32)), 32, axis=1)
            b, ok = bf16_exact(wd); wbf_all[ei] = b; wbad += int((~ok).sum())
            for gi in range(ng):
                if g[gi * 10] == ei:
                    r = g[gi * 10 + 2]
                    exact[gi] = wd @ ad[r]; mag[gi] = np.abs(wd) @ np.abs(ad[r])
        wbf_all.tofile(d + '/wbf.bin'); exact.tofile(d + '/exact.bin'); mag.tofile(d + '/mag.bin')
        meta = dict(N=N, K=K, R=int(R), ng=int(ng), E=len(distinct), act_bf16_inexact=int((~aok).sum()), w_bf16_inexact=wbad,
                    w_subnormal_codes=int((((W & 0x78) == 0) & ((W & 0x7f) != 0)).sum()), w_codes=int(W.size),
                    act_subnormal_codes=int((((v & 0x78) == 0) & ((v & 0x7f) != 0)).sum()), act_codes=int(v.size))
        json.dump(meta, open(d + '/meta.json', 'w')); summary[os.path.basename(d)] = meta; print(os.path.basename(d), meta, flush=True)
json.dump(dict(tape=TAPE, tokens=tok.tolist(), cases=summary), open(OUT + '/summary.json', 'w'), indent=1)
