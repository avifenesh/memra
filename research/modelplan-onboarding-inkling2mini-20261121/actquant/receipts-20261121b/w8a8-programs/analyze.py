
import numpy as np, json, os, sys
D = sys.argv[1]
def bf16(x):
    b = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7fff + ((b >> 16) & 1)) >> 16
    return b.astype(np.uint32)
def steps(a, b):
    k = lambda u: np.where(u & 0x8000, -(u & 0x7fff).astype(np.int64), u.astype(np.int64))
    return np.abs(k(a) - k(b))
res = {}
for c in sorted(os.listdir(D)):
    d = os.path.join(D, c)
    if not os.path.exists(d + '/acc.bf16mma'): continue
    m = json.load(open(d + '/meta.json')); N = m['N']; ng = m['ng']
    ex = np.fromfile(d + '/exact.bin').reshape(ng, N); mag = np.fromfile(d + '/mag.bin').reshape(ng, N)
    r = {}
    for p in ('mxf8', 'bf16mma'):
        acc = np.fromfile(d + '/acc.' + p, np.float32).reshape(ng, N).astype(np.float64)
        err = np.abs(acc - ex); rel = err / np.maximum(mag, 1e-300); rex = err / np.maximum(np.abs(ex), 1e-300)
        # rounding the exact sum to fp32 once: the best an fp32 accumulator can do
        st = steps(bf16(acc), bf16(ex))
        r[p] = dict(rel_mag_max=float(rel.max()), rel_mag_mean=float(rel.mean()), rel_mag_p99=float(np.quantile(rel, .99)),
                    rel_exact_median=float(np.median(rex)), rms_err=float(np.sqrt((err**2).mean())),
                    bf16_differs_from_bf16_of_exact=float((st != 0).mean()), bf16_max_steps=int(st.max()))
    a = np.fromfile(d + '/acc.mxf8', np.float32); b = np.fromfile(d + '/acc.bf16mma', np.float32)
    r['bf16_outputs_differ_between_programs'] = float((bf16(a) != bf16(b)).mean())
    r['elements'] = int(ng * N)
    f32once = np.abs(ex.astype(np.float32).astype(np.float64) - ex) / np.maximum(mag, 1e-300)
    r['fp32_once_rel_mag_mean'] = float(f32once.mean())
    res[c] = r; print(c, json.dumps(r))
json.dump(res, open(D + '/../w8-compare.json', 'w'), indent=1)
